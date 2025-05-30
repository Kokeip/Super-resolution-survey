
---

## 📊 方法总结型综述表格：超分辨率模型对比（推荐格式）

|类别|模型|核心机制|是否开源|推荐数据集|评估指标|GitHub链接|
|---|---|---|---|---|---|---|
|**基准模型**|Bicubic|三次插值|✅（内置）|Set5 / DIV2K|PSNR / SSIM|-|
||SRCNN|三层卷积|✅|Set5 / DIV2K|PSNR / SSIM|[GitHub](https://github.com/yjn870/SRCNN-pytorch)|
||FSRCNN|后上采样 + 轻量结构|✅|Set5 / DIV2K|PSNR / SSIM|[GitHub](https://github.com/yjn870/FSRCNN-pytorch)|
||ESPCN|Sub-pixel layer|✅|Set5 / DIV2K|PSNR / SSIM|[GitHub](https://github.com/twhui/DCSCN-Tensorflow)|
|**残差网络**|VDSR|残差学习 + 深层CNN|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/twtygqyy/pytorch-vdsr)|
||RDN|残差密集连接|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/yulunzhang/RCAN)|
||RCAN|残差通道注意力网络|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/yulunzhang/RCAN)|
||DRLN|多级残差 + 拉普拉斯注意力|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/IVIPLab/DRLN)|
|**注意力机制**|SwinIR|Swin Transformer + LayerNorm|✅|DIV2K|PSNR / SSIM / LPIPS|[GitHub](https://github.com/JingyunLiang/SwinIR)|
||HAN|多层注意力融合|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/Meituan-AutoML/HAN)|
|**轻量模型**|IMDN|信息蒸馏模块|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/zeyuanzhang/IMDN-pytorch)|
||CARN|轻量残差块|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/nmhkahn/CARN-pytorch)|
|**GAN类**|SRGAN|感知损失 + 对抗训练|✅|DIV2K|PSNR / SSIM / LPIPS / MOS|[GitHub](https://github.com/tensorlayer/srgan)|
||ESRGAN|Residual-in-Residual + GAN改进|✅|DIV2K+Flickr2K|PSNR / SSIM / LPIPS / MOS|[GitHub](https://github.com/xinntao/ESRGAN)|
|**扩散模型**|SR3|DDPM + 迭代去噪|✅|FFHQ / DIV2K|LPIPS / MOS|[GitHub](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)|
|**无监督模型**|ZSSR|图像自监督训练|✅|自建图像|PSNR / SSIM|[GitHub](https://github.com/assafshocher/ZSSR)|

---

## ✅ 你应在综述中补充/报告的指标说明

|指标|含义|是否客观|备注|
|---|---|---|---|
|**PSNR**|峰值信噪比|✅|测试图像越接近原图，数值越高|
|**SSIM**|结构相似度|✅|更能反映感知质量|
|**LPIPS**|感知距离|✅（需模型提取特征）|越低越好，需额外库|
|**MOS**|主观评分|❌|若你做主观测试（可选）|

---

## 🔧 补充工具建议

| 工具             | 用途            | 推荐                    |
| -------------- | ------------- | --------------------- |
| `scikit-image` | 计算PSNR / SSIM | ✅                     |
| `lpips`        | 安装LPIPS指标     | ✅ `pip install lpips` |
| `matplotlib`   | 结果可视化对比       | ✅                     |

---

## 📎 附件：可参考标题建议

- **A Practical Survey of Deep Super-Resolution Models: From Bicubic to Swin Transformer**
    
- **Benchmarking Deep Learning-Based Super-Resolution Models Under a Unified Framework**
    
- **An Empirical Review of Super-Resolution Techniques with Open Source Implementations**




## 🚀 如何使用本框架及扩展

本框架旨在提供一个统一的超分辨率模型评估和测试环境。目前已集成了 Bicubic 插值方法，并为 SRCNN 预留了接口。你可以按照以下步骤使用本框架，并轻松扩展以测试其他超分辨率模型。

### 1. 环境准备

确保你已安装所有必要的 Python 库。可以通过以下命令安装：

```bash
pip install opencv-python numpy scikit-image tqdm pandas matplotlib lpips
```
### 2. 数据集准备

将你的高分辨率（HR）和低分辨率（LR）图像数据集放置在项目根目录下的相应文件夹中。例如，对于 Set5 数据集，确保其结构如下：

```
Super resolution/
├── Set5/
│   ├── HR/
│   │   ├── baby.png
│   │   └── ...
│   └── LR_bicubic/
│       └── X4/
│           ├── babyx4.png
│           └── ...
└── ...
```

### 3. 运行测试

你可以通过运行 `test.py` 文件来对已实现的模型进行测试。在 `test.py` 中，你可以配置要测试的模型、数据集和上采样比例。

```python:d:\pycharm\Super resolution\test.py
# ... existing code ...

if __name__ == '__main__':
    # 配置要测试的模型和数据集
    # MODELS = ['Bicubic', 'SRCNN'] # 示例：可以添加更多模型名称
    # DATASETS = ['Set5', 'DIV2K'] # 示例：可以添加更多数据集名称

    # 运行测试
    # test_model_on_dataset(MODELS, DATASETS, scale=4)

    # 示例：仅测试 Bicubic 在 Set5 上的表现
    test_model_on_dataset(['Bicubic'], ['Set5'], scale=4)

    # 示例：如果你实现了 SRCNN，可以这样测试
    # test_model_on_dataset(['SRCNN'], ['Set5'], scale=4)

# ... existing code ...
```

### 4. 实现新的超分辨率方法

要添加新的超分辨率模型，请按照以下步骤操作：

1.  **在 `methods` 文件夹中创建新的 Python 文件：**

    在 `d:\pycharm\Super resolution\methods` 目录下，为你的新模型创建一个独立的 `.py` 文件，例如 `your_model_method.py`。在这个文件中，定义你的超分辨率函数，该函数应接受低分辨率图像和上采样比例作为输入，并返回超分辨率图像。

    ```python:d:\pycharm\Super resolution\methods\your_model_method.py
    import cv2
    import numpy as np
    # 导入你的模型所需的其他库，例如 torch, tensorflow 等

def your_model_upscale(lr_img, scale=4):
    """你的模型上采样实现"""
    # 在这里加载你的模型，并执行推理
    # sr_img = your_model.predict(lr_img)
    # return sr_img
    # 示例：这里只是一个占位符，你需要替换为你的模型逻辑
    return cv2.resize(
        lr_img,
        (int(lr_img.shape[1]*scale), int(lr_img.shape[0]*scale)),
        interpolation=cv2.INTER_CUBIC
    )
    ```

2.  **在 `test.py` 中导入并集成你的方法：**

    在 `test.py` 文件的顶部，导入你新创建的方法。然后，在 `apply_sr_model` 函数中添加一个 `elif` 分支来调用你的新方法。

    ```python:d:\pycharm\Super resolution\test.py
    import cv2
    # ... existing code ...
    from methods.bicubic_method import bicubic_upscale
    # 导入你的新方法
    from methods.your_model_method import your_model_upscale

    # ... existing code ...

def apply_sr_model(lr_img, model_name, scale):
    if model_name == 'Bicubic':
        return bicubic_upscale(lr_img, scale)
    elif model_name == 'SRCNN':
        raise NotImplementedError(f"Model {model_name} is not yet implemented.")
    # 添加你的新模型分支
    elif model_name == 'YourModel':
        return your_model_upscale(lr_img, scale)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # ... existing code ...
    ```

3.  **更新 `MODELS` 列表：**

    在 `test.py` 的 `if __name__ == '__main__':` 块中，将你的新模型名称添加到 `MODELS` 列表中，以便在测试时包含它。

    ```python:d:\pycharm\Super resolution\test.py
    # ... existing code ...

if __name__ == '__main__':
    MODELS = ['Bicubic', 'SRCNN', 'YourModel'] # 添加你的新模型名称
    DATASETS = ['Set5', 'DIV2K']

    test_model_on_dataset(MODELS, DATASETS, scale=4)

    # ... existing code ...
    ```

### 5. 评估指标

框架会自动计算 PSNR、SSIM 和 LPIPS 等指标。确保你的模型输出图像的格式和数据类型与框架要求一致（通常是 NumPy 数组，像素值范围在 0-255 或 0-1 之间）。

### 6. 结果保存

超分辨率后的图像和评估结果将保存在 `results` 文件夹中。你可以根据需要调整保存逻辑。