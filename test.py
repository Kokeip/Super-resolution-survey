import cv2
import numpy as np
import os
import lpips
import glob
import time
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt

# 从单独的文件中导入上采样方法
from methods.bicubic_method import bicubic_upscale
from methods.rcan_method import rcan_upscale

# 初始化LPIPS模型
loss_fn_alex = lpips.LPIPS(net='alex')  # 使用AlexNet作为特征提取器

# ========================
# 超参数配置
# ========================
SCALE = 4  # 超分倍数
MODELS = [
    'RCAN', # 暂时只保留Bicubic，其他模型需要单独实现并导入
    #'Bicubic' # 添加Bicubic模型，以便进行对比
    # 'SRCNN', 'FSRCNN', 'ESPCN', 'VDSR', 
    # 'RDN', 'RCAN', 'DRLN', 'SwinIR', 'HAN', 
    # 'IMDN', 'CARN', 'SRGAN', 'ESRGAN', 'SR3', 'ZSSR'
]
DATASETS = {
    'DIV2K': {
        'HR': 'DIV2K_valid_HR',
        'LR': 'DIV2K_valid_LR_bicubic/X4'
    },
    'Set5': {
        'HR': 'Set5/HR',
        'LR': 'Set5/LR_bicubic/X4'  # 需要预先生成
    }
}

# ========================
# 辅助函数
# ========================
def prepare_set5_lr(hr_path, lr_path, scale=4):
    """为Set5生成LR图像"""
    os.makedirs(lr_path, exist_ok=True)
    hr_files = glob.glob(os.path.join(hr_path, '*.png'))
    
    for hr_file in tqdm(hr_files, desc="Preparing Set5 LR images"):
        hr_img = cv2.imread(hr_file)
        lr_img = cv2.resize(
            hr_img, 
            (int(hr_img.shape[1]/scale), int(hr_img.shape[0]/scale)),
            interpolation=cv2.INTER_CUBIC
        )
        lr_file = os.path.join(lr_path, os.path.basename(hr_file))
        cv2.imwrite(lr_file, lr_img)

def calculate_metrics(hr_img, sr_img):
    """计算PSNR, SSIM, LPIPS指标"""
    # 确保图像尺寸匹配
    if hr_img.shape != sr_img.shape:
        hr_img = cv2.resize(hr_img, (sr_img.shape[1], sr_img.shape[0]))
    
    # PSNR (Y通道)
    hr_y = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    sr_y = cv2.cvtColor(sr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    psnr_value = psnr(hr_y, sr_y, data_range=255)
    
    # SSIM (Y通道)
    ssim_value = ssim(hr_y, sr_y, data_range=255)
    
    # LPIPS (RGB空间)
    hr_tensor = lpips.im2tensor(hr_img).float()/127.5 - 1
    sr_tensor = lpips.im2tensor(sr_img).float()/127.5 - 1
    lpips_value = loss_fn_alex(hr_tensor, sr_tensor).item()
    
    return psnr_value, ssim_value, lpips_value

def visualize_comparison(hr_img, sr_img, model_name, save_path):
    """可视化对比结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # HR图像
    axes[0].imshow(cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('HR Image')
    axes[0].axis('off')
    
    # SR图像
    axes[1].imshow(cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'SR Image ({model_name})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ========================
# 模型推理接口
# ========================
# bicubic_upscale 函数已移至 methods/bicubic_method.py

# srcnn_upscale 函数已移除，因为它只是一个临时替代，未来应单独实现
# def srcnn_upscale(lr_img, scale=4):
#     """SRCNN模型上采样 - 需要实现模型加载和推理"""
#     # 伪代码：实际需要加载预训练模型
#     # model = load_srcnn_model()
#     # sr_img = model.predict(lr_img)
#     return bicubic_upscale(lr_img, scale)  # 临时替代

# 其他模型的上采样函数类似实现...
# 实际使用时需要为每个模型实现具体的推理逻辑

# ========================
# 主测试函数
# ========================
def test_model_on_dataset(model_name, dataset_name, scale=4, save_visuals=False):
    """在指定数据集上测试单个模型"""
    hr_dir = DATASETS[dataset_name]['HR']
    lr_dir = DATASETS[dataset_name]['LR']
    
    # 获取文件列表
    hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
    
    # 确保文件数量匹配
    if len(hr_files) != len(lr_files):
        print(f"警告: {dataset_name}数据集HR({len(hr_files)})和LR({len(lr_files)})文件数量不匹配")
        min_files = min(len(hr_files), len(lr_files))
        hr_files = hr_files[:min_files]
        lr_files = lr_files[:min_files]
    
    results = []
    times = []
    
    # 定义图像大小阈值和 patch 大小
    IMAGE_SIZE_THRESHOLD = 200
    PATCH_SIZE = 128
    OVERLAP_SIZE = 16 # 示例重叠大小，可以根据需要调整
    
    # 创建可视化目录
    if save_visuals:
        visual_dir = f"results/visuals/{dataset_name}/{model_name}"
        os.makedirs(visual_dir, exist_ok=True)
    
    for hr_path, lr_path in tqdm(zip(hr_files, lr_files), 
                                desc=f"{model_name} on {dataset_name}", 
                                total=len(hr_files)):
        # 读取图像
        hr_img = cv2.imread(hr_path)
        lr_img = cv2.imread(lr_path)
        
        # 获取图像尺寸
        h, w, _ = lr_img.shape
        
        # 模型推理
        start_time = time.time()
        
        # 根据图像大小选择处理方式
        if h < IMAGE_SIZE_THRESHOLD and w < IMAGE_SIZE_THRESHOLD:
            # 整图处理
            print(f"处理图像: {os.path.basename(lr_path)} (整图)")
            sr_img = apply_sr_model(lr_img, model_name, scale)
        else:
            # Patch 处理
            print(f"处理图像: {os.path.basename(lr_path)} (Patch)")
            sr_img = process_image_with_patches(lr_img, model_name, scale, PATCH_SIZE, OVERLAP_SIZE)
            
        inference_time = time.time() - start_time
        times.append(inference_time)
        
        # 计算指标
        metrics = calculate_metrics(hr_img, sr_img)
        results.append(metrics)
        
        # 保存可视化结果（每10张保存1张），Set都存
        if save_visuals and ((len(results) % 10 == 1) or dataset_name == 'Set5'):
            save_path = os.path.join(visual_dir, os.path.basename(hr_path))
            visualize_comparison(hr_img, sr_img, model_name, save_path)
    
    # 计算平均指标
    avg_metrics = np.mean(results, axis=0)
    avg_time = np.mean(times)
    
    return {
        'psnr': avg_metrics[0],
        'ssim': avg_metrics[1],
        'lpips': avg_metrics[2],
        'time': avg_time,
        'results': results
    }

def apply_sr_model(img, model_name, scale):
    """根据模型名称应用超分辨率模型"""
    if model_name == 'Bicubic':
        return bicubic_upscale(img, scale)
    elif model_name == 'RCAN':
        return rcan_upscale(img, scale)
    elif model_name == 'SRCNN':
        raise NotImplementedError("SRCNN model is not yet implemented. Please implement it in a separate file.")
    # 添加其他模型的分支...
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Only 'Bicubic' and 'RCAN' are supported for now.")

def process_image_with_patches(lr_img, model_name, scale, patch_size, overlap_size):
    """使用 patch 处理图像并进行重叠融合"""
    h_lr, w_lr, _ = lr_img.shape
    h_sr, w_sr = h_lr * scale, w_lr * scale

    # 初始化最终的SR图像和权重图
    sr_img_final = np.zeros((h_sr, w_sr, 3), dtype=np.float32)
    weight_map = np.zeros((h_sr, w_sr, 1), dtype=np.float32)

    # 计算步长
    stride = patch_size - overlap_size

    # 生成一个 patch 的权重图
    # 这里使用一个简单的线性衰减权重，中心最大，边缘最小
    patch_weight = np.zeros((patch_size * scale, patch_size * scale, 1), dtype=np.float32)
    for i in range(patch_size * scale):
        for j in range(patch_size * scale):
            dist_x = min(i, patch_size * scale - 1 - i)
            dist_y = min(j, patch_size * scale - 1 - j)
            patch_weight[i, j] = min(dist_x, dist_y) + 1 # 距离边缘越近，权重越小，但至少为1
    patch_weight = patch_weight / np.max(patch_weight) # 归一化到0-1

    # 遍历所有 patch
    for y in range(0, h_lr, stride):
        for x in range(0, w_lr, stride):
            # 确定当前 patch 的区域
            y_end = min(y + patch_size, h_lr)
            x_end = min(x + patch_size, w_lr)
            
            # 提取 LR patch
            lr_patch = lr_img[y:y_end, x:x_end, :]

            # 如果 patch 小于指定大小，进行 padding
            padded_lr_patch = np.zeros((patch_size, patch_size, 3), dtype=lr_img.dtype)
            padded_lr_patch[:lr_patch.shape[0], :lr_patch.shape[1], :] = lr_patch

            # 对 patch 进行超分
            sr_patch = apply_sr_model(padded_lr_patch, model_name, scale)

            # 计算 SR patch 在最终图像中的位置
            sr_y_start = y * scale
            sr_x_start = x * scale
            sr_y_end = sr_y_start + sr_patch.shape[0]
            sr_x_end = sr_x_start + sr_patch.shape[1]

            # 将 SR patch 及其权重叠加到最终图像和权重图中
            # 注意：这里需要处理 sr_patch 实际大小可能小于 (patch_size * scale) 的情况
            current_sr_patch_h = min(sr_patch.shape[0], h_sr - sr_y_start)
            current_sr_patch_w = min(sr_patch.shape[1], w_sr - sr_x_start)

            sr_img_final[sr_y_start:sr_y_start + current_sr_patch_h,
                         sr_x_start:sr_x_start + current_sr_patch_w, :] += \
                sr_patch[:current_sr_patch_h, :current_sr_patch_w, :] * \
                patch_weight[:current_sr_patch_h, :current_sr_patch_w, :]

            weight_map[sr_y_start:sr_y_start + current_sr_patch_h,
                       sr_x_start:sr_x_start + current_sr_patch_w, :] += \
                patch_weight[:current_sr_patch_h, :current_sr_patch_w, :]

    # 避免除以零，将权重图中小于一个很小的值的地方设置为1
    weight_map[weight_map == 0] = 1e-6

    # 加权平均融合
    sr_img_final = sr_img_final / weight_map
    
    # 裁剪到原始 HR 图像的尺寸，因为 padding 可能会导致尺寸略大
    sr_img_final = sr_img_final[:h_sr, :w_sr, :]

    return sr_img_final.astype(np.uint8)


def run_full_experiment(save_csv=True, save_visuals=False):
    """运行完整实验：所有模型在所有数据集上测试"""
    # 准备结果目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visuals', exist_ok=True)
    
    # 为Set5生成LR图像（如果不存在）
    if not os.path.exists(DATASETS['Set5']['LR']):
        print("为Set5生成LR图像...")
        prepare_set5_lr(
            DATASETS['Set5']['HR'], 
            DATASETS['Set5']['LR'], 
            SCALE
        )
    
    # 结果数据结构
    all_results = {ds: {model: None for model in MODELS} for ds in DATASETS}
    summary_df = pd.DataFrame(columns=['Model', 'Dataset', 'PSNR', 'SSIM', 'LPIPS', 'Time(s)'])
    
    # 遍历所有模型和数据集
    for dataset in DATASETS:
        for model in MODELS:
            print(f"\n{'='*50}")
            print(f"测试模型: {model} | 数据集: {dataset}")
            print(f"{'='*50}")
            
            # 运行测试
            results = test_model_on_dataset(
                model_name=model,
                dataset_name=dataset,
                scale=SCALE,
                save_visuals=save_visuals
            )
            
            # 存储结果
            all_results[dataset][model] = results
            
            # 添加到汇总表
            new_row = {
                'Model': model,
                'Dataset': dataset,
                'PSNR': results['psnr'],
                'SSIM': results['ssim'],
                'LPIPS': results['lpips'],
                'Time(s)': results['time']
            }
            new_row_df = pd.DataFrame([new_row]) # 将字典转换为DataFrame
            summary_df = pd.concat([summary_df, new_row_df], ignore_index=True) # 使用concat连接
            
            # 打印当前结果
            print(f"结果 - PSNR: {results['psnr']:.2f} dB | "
                  f"SSIM: {results['ssim']:.4f} | "
                  f"LPIPS: {results['lpips']:.4f} | "
                  f"时间: {results['time']:.2f}s")
    
    # 保存结果
    if save_csv:
        summary_df.to_csv('results/sr_model_comparison.csv', index=False)
        print("结果已保存至: results/sr_model_comparison.csv")
    
    # 生成可视化报告
    generate_report(all_results)
    
    return all_results

def generate_report(results):
    """生成可视化报告"""
    # 创建图表目录
    os.makedirs('results/plots', exist_ok=True)
    
    # 为每个数据集创建指标对比图
    for dataset in DATASETS:
        models = []
        psnr_values = []
        ssim_values = []
        lpips_values = []
        times = []
        
        for model in MODELS:
            models.append(model)
            psnr_values.append(results[dataset][model]['psnr'])
            ssim_values.append(results[dataset][model]['ssim'])
            lpips_values.append(results[dataset][model]['lpips'])
            times.append(results[dataset][model]['time'])
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # PSNR对比
        plt.subplot(2, 2, 1)
        plt.bar(models, psnr_values, color='skyblue')
        plt.title(f'{dataset} - PSNR Comparison')
        plt.ylabel('PSNR (dB)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # SSIM对比
        plt.subplot(2, 2, 2)
        plt.bar(models, ssim_values, color='lightgreen')
        plt.title(f'{dataset} - SSIM Comparison')
        plt.ylabel('SSIM')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # LPIPS对比
        plt.subplot(2, 2, 3)
        plt.bar(models, lpips_values, color='salmon')
        plt.title(f'{dataset} - LPIPS Comparison')
        plt.ylabel('LPIPS (越低越好)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 时间对比
        plt.subplot(2, 2, 4)
        plt.bar(models, times, color='gold')
        plt.title(f'{dataset} - Inference Time Comparison')
        plt.ylabel('Time (s)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{dataset}_metrics_comparison.png', dpi=150)
        plt.close()
    
    print("可视化报告已生成在 results/plots/ 目录")

# ========================
# 主程序入口
# ========================
if __name__ == "__main__":
    # 运行完整实验
    full_results = run_full_experiment(
        save_csv=True,
        save_visuals=True  # 设置为True保存可视化对比图
    )
    
    # 打印最终结果摘要
    print("\n\n实验完成! 结果摘要:")
    print(pd.read_csv('results/sr_model_comparison.csv'))