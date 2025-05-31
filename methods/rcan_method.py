import numpy as np 
import torch
import torch.nn as nn
import os
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import math

# --- Start of common.py content (adapted) ---
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) # Official uses rgb_range here
        self.bias.data.div_(std)
        self.requires_grad = False

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)
# --- End of common.py content ---


# --- Start of rcan.py content (adapted) ---
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x # Official implementation uses res += x, not mul(res_scale)
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size)) # This conv is outside the loop in official code
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class RCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args['n_resgroups']
        n_resblocks = args['n_resblocks']
        n_feats = args['n_feats']
        kernel_size = 3
        reduction = args['reduction'] 
        scale = args['scale']
        act = nn.ReLU(True)
        
        # RGB mean and std from official code
        rgb_mean = (0.4488, 0.4371, 0.4040) # Default for DIV2K
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args['rgb_range'], rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args['n_colors'], n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args['res_scale'], n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args['n_colors'], kernel_size)]

        self.add_mean = MeanShift(args['rgb_range'], rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        x = x + res # Official implementation uses x + res

        x = self.tail(x)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, ' \
                                           'whose dimensions in the model are {} and ' \
                                           'whose dimensions in the checkpoint are {}.' \
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


def rcan_upscale(lr_image_path, scale_factor=4):
    model_path = 'model/RCAN_BIX4.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型参数，这些参数需要与预训练模型匹配
    # 官方默认参数 (RCAN_BIX4_G10R20P16)
    args = {
        'scale': scale_factor,
        'n_resgroups': 10,  # 10 Residual Groups
        'n_resblocks': 20,  # 20 RCABs in each Residual Group (official uses n_resblocks for RCAB count)
        'n_feats': 64,      # 64 feature maps
        'reduction': 16,    # Channel Attention reduction factor
        'rgb_range': 255,   # RGB range
        'n_colors': 3,      # Number of color channels
        'res_scale': 1      # Residual scaling (official default is 1)
    }
    
    model = RCAN(args).to(device)
    
    if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
    
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False) # Set strict=False for potential minor mismatches
        model.eval()
    else:
        print(f"Warning: RCAN model weights not found at {model_path}. Using uninitialized model.")
        raise FileNotFoundError(f"RCAN model weights not found at {model_path}. Please download the pre-trained model.")

    lr_image = Image.fromarray(lr_image_path).convert('RGB')
    lr_tensor = ToTensor()(lr_image).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_image = ToPILImage()(sr_tensor.squeeze(0).cpu())
    # 将PIL Image转换为NumPy数组
    sr_image_np = np.array(sr_image)
    return sr_image_np