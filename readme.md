
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
    
