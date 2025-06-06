
---

## 📊 方法总结型综述表格：超分辨率模型对比（推荐格式）

|类别|模型|核心机制|是否开源|推荐数据集|评估指标|GitHub链接|
|---|---|---|---|---|---|---|
|**基准模型**|Bicubic(1)|三次插值|✅（内置）|Set5 / DIV2K|PSNR / SSIM|-|
||SRCNN(2)|三层卷积|✅|Set5 / DIV2K|PSNR / SSIM|[GitHub](https://github.com/yjn870/SRCNN-pytorch)|
||FSRCNN(2)|后上采样 + 轻量结构|✅|Set5 / DIV2K|PSNR / SSIM|[GitHub](https://github.com/yjn870/FSRCNN-pytorch)|
||ESPCN(2)|Sub-pixel layer|✅|Set5 / DIV2K|PSNR / SSIM|[GitHub](https://github.com/twhui/DCSCN-Tensorflow)|
|**残差网络**|VDSR(2)|残差学习 + 深层CNN|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/twtygqyy/pytorch-vdsr)|
||RDN(2)|残差密集连接|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/yulunzhang/RCAN)|
||RCAN(1)|残差通道注意力网络|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/yulunzhang/RCAN)|
||DRLN(1)|多级残差 + 拉普拉斯注意力|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/IVIPLab/DRLN)|
|**注意力机制**|SwinIR(3)|Swin Transformer + LayerNorm|✅|DIV2K|PSNR / SSIM / LPIPS|[GitHub](https://github.com/JingyunLiang/SwinIR)|
||HAN(3)|多层注意力融合|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/Meituan-AutoML/HAN)|
|**轻量模型**|IMDN(1)|信息蒸馏模块|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/zeyuanzhang/IMDN-pytorch)|
||CARN(1)|轻量残差块|✅|DIV2K|PSNR / SSIM|[GitHub](https://github.com/nmhkahn/CARN-pytorch)|
|**GAN类**|SRGAN(1)|感知损失 + 对抗训练|✅|DIV2K|PSNR / SSIM / LPIPS / MOS|[GitHub](https://github.com/tensorlayer/srgan)|
||ESRGAN(1)|Residual-in-Residual + GAN改进|✅|DIV2K+Flickr2K|PSNR / SSIM / LPIPS / MOS|[GitHub](https://github.com/xinntao/ESRGAN)|
|**扩散模型**|SR3(3)|DDPM + 迭代去噪|✅|FFHQ / DIV2K|LPIPS / MOS|[GitHub](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)|
|**无监督模型**|ZSSR(1)|图像自监督训练|✅|自建图像|PSNR / SSIM|[GitHub](https://github.com/assafshocher/ZSSR)|

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



##参考GPT回答

---

## 🌱 1️⃣ 最早期阶段：插值 & 传统机器学习

最早做超分（SISR）的时候，**没有深度学习**，主要靠：

* **双线性（bilinear）**、\*\*双三次（bicubic）\*\*插值 → 画质差，容易模糊
* **基于稀疏编码、字典学习** → 比如 SRC（Sparse Representation based SR）

这是 SISR 领域最早的阶段，完全没有CNN、Attention这些概念，靠手工设计特征 + 小模型做重建，效果有限。

👉 这类现在通常作为 baseline 比如 "bicubic"。

---

## 🚀 2️⃣ 第一波深度学习 SR：简单CNN

2014年 **SRCNN** 出来了，标志着 **SISR 进入深度学习时代**：

* **SRCNN**: 第一个深度网络SR，结构非常简单，3层CNN，输入是低分辨率图像插值到高分辨率后，做修复。
* 后续 **FSRCNN** → 改进速度
* **VDSR** → 改深度更高（20层），直接学习 LR→HR 的残差。

👉 这类可以叫 **simple CNN SR**，是“深度学习起步阶段”的 SR。

---

## 🏗️ 3️⃣ 第二波：残差 & 递归 网络（更深更稳）

随着 **ResNet** 爆火，SISR 也开始用残差学习：

* **EDSR**: 去掉BN，纯残差block堆叠，效果非常强。
* **DRRN**: 残差 + 递归结构，模型参数量少，但可以无限加深（用递归）。
* **DRLN**: 加 Laplacian 层做多尺度残差增强。

👉 这阶段的 SR 方法大体目标是：**做更深的网络 → 提升细节还原能力**，典型特点是 **Residual Learning**。

---

## ✨ 4️⃣ Attention-based CNN（局部注意力机制）

CNN虽然很强，但 **固定卷积核**限制了信息建模能力。
于是大家开始在CNN中加入 **注意力机制（attention）**：

* **RCAN**: 引入了 **Channel Attention（通道注意力）**，提升了 CNN 的细节建模能力。
* **SAN**: 不仅用 channel attention，还加了 **non-local block**，做长距离依赖建模。
* **HAN**、**CSNLN** 等进一步把 attention 用得更花。

👉 这些方法还是CNN为主，只不过 **在block中加 attention 提升效果**，可以叫 **Attention-based CNN**。

---

## 🪶 5️⃣ 轻量化模型（轻便模型，适合移动端）

深的网络效果好，但计算量大。很多人希望 **跑在手机上、嵌入式设备上**：

* **IMDN**: 信息多蒸馏网络，核心是轻量化 + 高效attention。
* **CARN**: Cascading Residual Network，轻量版的 RCAN 思路。
* **ESPCN**: 最早一批轻量 SR，直接学习 sub-pixel shift。
* **MobileNet/ShuffleNet SR** 也是这个流派的延伸。

👉 这一流派叫 **轻量化模型 / Mobile-friendly SR**，追求速度快 + 模型小。

---

## 🎨 6️⃣ GAN-based SR（以感知质量为目标）

之前CNN和attention都是追求PSNR、SSIM，但画面会偏平滑。
**GAN（生成对抗网络）** 引入后，目标改成 **“生成逼真好看的高分图像”**：

* **SRGAN**: 第一个 GAN-based SR，图像质量大幅提升（虽然PSNR低）。
* **ESRGAN**: 更强的 GAN + 更好感知loss，成为业界“画质最好”之一。
* **Real-ESRGAN**: 关注真实退化图像，能处理噪声、压缩 artifacts。

👉 这个流派 **不是追求PSNR**，追求 **主观画质好**，适合影视、游戏等场景。

---

## 🧠 7️⃣ Transformer-based SR（跨时代新主流）

CNN attention 结构好归好，但 **局部 receptive field** 限制很大，**MSA（Multi-head Self-Attention）** 可以跨全局建模信息。

* **SwinIR**: 第一批做得很好的 **基于 Swin Transformer** 的 SR。
* **HAT**: Hierarchical Attention Transformer，进一步提升性能。
* **Restormer**: 适合做各种图像复原，包括 SR。

👉 Transformer-based SR 是现在的**新主流**，效果优于CNN，感知能力好。

---

## 🌈 8️⃣ Diffusion-based SR（最新趋势）

2023年后，**扩散模型（Diffusion models）** 开始进军 SR：

* **SR3**（Google）: 用扩散模型一步步噪声逆向采样生成超分结果，画质非常棒。
* **StableSR**、**DDPMSR**: 把扩散用到更大scale、可控超分。
* **Hitchhiker’s Guide** 指出 diffusion 很可能是未来SR趋势之一。

👉 扩散SR目前还是 **研究阶段为主**，工业界部分用在高端场景（艺术风格、大分辨率SR）。



## 🌟 结语

所以你会发现：

* **CNN流派**（Residual, Attention, Lightweight）是一路发展出来的。
* **GAN流派**是目标转向“感知质量”。
* **Transformer / Diffusion** 是**新架构革命**，可以处理全局依赖，更强。
* **轻量化模型**和 Attention 有交集，但侧重点不同，确实应该单独放。
* **Diffusion 是目前“未来”趋势，GAN和Diffusion目前“分流”。**

---

