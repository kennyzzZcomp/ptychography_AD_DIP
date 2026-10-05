# DeePIE：对接 ProPtyNet 仿真的独立复现

入口：`simulations/DeePIE.py`。模型：`functions/paperrepro/deepie_model.py`。
训练器：`functions/paperrepro/deepie_solver.py`。

这是根据正文与补充材料编写的独立实现，并适配本仓库的 Fresnel 仿真条件。
不是作者代码，不声称逐数值复现原文表格。`run` 的完成只表示指定更新次数执行完毕，
不表示收敛、论文指标复现成功或方法优劣已得到验证。

## 来源与实现对应

- 正文：Z. Zhang et al., *X-ray ptychography using physics-enhanced implicit neural
  representations*, Optics Letters 50, 7159–7162 (2025),
  https://doi.org/10.1364/OL.578784 。本地输入 `xray_ptychoINR.pdf`。
- Supplement 1（用户提供的 `7672073.pdf`）：
  https://doi.org/10.6084/m9.figshare.30454637 。以下 PDF 页码包含封面。

| 依据 | 实现 | 复现边界 |
|---|---|---|
| 正文 Eq.4 | 振幅、相位分别由两个独立坐标网络产生；O=A exp(i 2π f_phase) | Eq.5 省略了 2π；默认按 Eq.4，可用 `--phase-scale` 做明确消融 |
| 正文 Eq.3/5 | 衍射振幅 L2，联合优化物体网络和像素复数探针 | 用均值而非求和，明确记录；统一物理条件下用现有 Fresnel 替换原文 Fraunhofer 前向 |
| Sup PDF 2–3 页 | 三个隐藏层、每层 256、线性输出，NeRF 式位置编码 | 编码精确排列、坐标区间等未给出，见下文 |
| Sup S3–S5 | W=ΛB；B 为固定余弦基，训练 Λ | 所有仿射层含输出层均重参数化；具体应用层范围存在作者实现不确定性 |
| Sup PDF 3 页 | 128 高频、128 低频、32 相位、ω₀=30 | 按 S4 的角频率解释，不暗加 2π |
| 正文训练段 | Adam，初始学习率 1e-4，step decay | 探针独立 LR、衰减节点、因子未公开；当前默认均 1e-4，每 100 次更新乘 0.5 |
| 正文训练段 | 网络梯度平衡、按扫描强度加权 | 作者没有给完整公式，当前实现为明确声明的替代规则，不能称为作者完整 GradNorm |

## 必须披露的实现选择

1. **激活公式尚未确定。** Sup 给了 ω₀=30 与 α=0.05，但没有明确 α 怎样进入激活。
   默认 `--activation sine` 使用 sin(ω₀x)，α 不参与计算并在 manifest 标出。
   `--activation leaky-sine --alpha 0.05` 是 leaky_relu(sin(ω₀x), α) 的可选假设，
   并非从论文确认的公式。二者均不得标为作者逐层一致实现。
2. **位置编码约定。** 全局物体坐标 x、y 各在 [-1,1] 上包含两端点均匀取样。
   保留原坐标，拼接 sin(π 2^k x)、cos(π 2^k x) 及 y 项。
   `encoding_side=256` 时取 2^k <= (256-1)/2，共 7 个频带，编码后 30 维。
   这是对补充材料“NeRF + Nyquist”的一种明确实现，不把它解释为作者确定约定。
3. **权重 Fourier 基。** z=linspace(0,1,input_width)，低频为 1/128,…,1，
   高频为 1,…,128；32 相位均匀覆盖 [0,2π)，不重复终点。频率 1 按文意保留两次。
   B 不做归一化；改变基幅度会改变 Adam 对 Λ 的优化行为。
4. **初始化。** Λ 独立零均值高斯，按 B 列能量匹配 SIREN 式有效 W 的平均方差；
   不声称有效 W 的各元素独立。隐藏层偏置零；输出权重额外乘 `head_scale=1e-3`，
   振幅输出偏置 1、相位偏置 0，使初始物体接近 1。初始化不使用真值。
5. **探针与测量尺度。** 初始形状来自现有 `scene.P0`，默认全 1。
   默认仅在训练前用测量总能量/初始预测总能量的平方根缩放 P0 一次；不使用真值探针，
   不做逐帧归一化，不做逐次最优尺度拟合。`--probe-scale none` 关闭。
   这一缩放适配原仿真全局强度归一化，属于新增且显式披露的初始化选择。
6. **扫描加权。** `--scan-weight intensity`：w_i=mean(I_i)/mean(I_all)，
   L=mean_i[w_i mean_pixels((abs(U_i)-sqrt(I_i))²)]。
   `--scan-weight uniform` 恢复无权振幅 MSE。两者都只使用测量，不访问干净真值。
7. **梯度平衡。** `--balance equal-norm`：分别计算振幅/相位网络参数梯度 L2 范数 g_A,g_P，
   以 sqrt(g_A*g_P) 为目标，乘数裁到 [0.1,10]；探针梯度不缩放。
   不包括 GradNorm 的可学习任务权重或相对训练速率，不能称为完整 GradNorm。
   Adam 的归一化可能减弱整体梯度缩放的效果；此策略有效性尚需消融。
   `--balance none` 关闭。无权损失、不平衡梯度版本有独立命令如下。

**参数计数。** M=(128+128)×32=8192。当前所有仿射层都训练 Λ 时，
两个物体网络合计 12,600,834 个可训练实数参数；探针另有 2N² 个。
Sup Table S2 的 132,609 是普通单网络 `2→256→256→256→1` 的稠密权重计数，
不等于本实现的训练参数量，也没有计入 PE 的输入维度。
训练后的 W 可以折叠用于推理，但训练期间不能把 Λ 换成自由 W，否则改变优化路径。

## 与你的 ProPtyNet 数据保持一致

直接使用未修改的 `Cfg`、`build_scene()`、`forward_field()`、`evaluate()` 和 `probe_relerr()`。
共用物体/探针真值、位置、噪声、全局归一化、Fresnel 二次相位及 FFT 约定。
不切换到 DeePIE 论文的 Fermat spiral、不改变你的波长、不加入真值支撑掩膜或 TGV。
GT 只用于仿真和已有评价口径，不参与网络初始化、梯度、学习率选择或提前停止。

场景参数相同、素材文件相同、软件/设备计算路径相同时，打印的 scene 指纹应一致。
网络随机种子 `--network-seed` 与场景 `--seed` 分离，改变网络种子不改变数据。
跨设备/库版本的 FFT 或图像缩放可能产生浮点差异，因此不承诺跨平台位级相同。
精确审计时加 `--save-scene` 保存实际 Im/Icl/P0/positions/Q（paper 尺寸文件可能较大）。

读取已有 `paper_result.npz` / `net_result.npz` / `ad_result.npz` 时，用 `--scene-config`：
只继承场景和评价字段，不继承原算法的网络、损失或优化超参数。
这会重新生成场景，并不把 result.npz 当成包含测量数据的文件。
旧结果通常没有保存动态 `quad_sign`：程序会提示，默认 -1；若原实验使用 +1 必须显式传入。
默认沿用源文件的 preset；显式 `--preset` 覆盖源文件中的 preset 几何，
其余显式场景参数优先级最高。素材路径相对当前工作目录解析。

评价沿用原仓库：统一 ROI、全局复因子对齐、相位均值对齐；不额外给 DeePIE
做平移/相位斜坡优化。ROI 的确定可能使用仿真真值照明，但只用于报告，不参与优化。
共用评价函数的矩形 ROI/中心 PSNR 裁剪习惯也未改动。

## 显存与迭代含义

每一步均使用全部扫描点，只做一次 Adam 更新：

1. 从 Λ、B 计算有效 W，在无梯度模式分块生成整个物体场。
2. 将物体场视为叶子张量，扫描分块，逐块累积光学损失对物体和探针的梯度。
3. 用物体场梯度对坐标网络分块做 VJP，累积 dL/dW。
4. 使用 dL/dΛ=(dL/dW)Bᵀ，得到真实训练参数的梯度，随后平衡梯度并更新。

这是链式法则的等价计算；没有使用陈旧场、截断梯度、减少测量或改变扫描优化顺序。
`--scan-chunk` 和 `--coordinate-chunk` 控制内存，不是随机 mini-batch。
测试将复杂场、探针、所有网络参数梯度与直接 autograd 对照。
浮点求和顺序不同可能带来微小数值差异。

**一次更新不等于原论文一次 iteration/epoch 的已确认定义。** 原文没有完全规定其更新调度。
与其他方法比较，应同时报告全测量访问次数、时间和收敛曲线，不能只比较同名 iterations。

## 运行（PowerShell，从 INNM_Code 根目录）

本机已有 `D:\miniconda\envs\flatnet\python.exe`，PyTorch 2.5.1+cu121 可使用 GPU。
不需要安装 tiny-cuda-nn、CUDA 编译器或 PtyLab；后者的物理/数据职责由共用框架承担。
环境依赖为 torch、numpy、scipy、matplotlib、Pillow；图像读取沿用环境中 OpenCV/PIL 的既有选择。

```powershell
$env:PYTHONIOENCODING='utf-8'
$py = 'D:\miniconda\envs\flatnet\python.exe'

# 只核对场景，不训练
& $py simulations/DeePIE.py check --preset paper --device cuda

# 小尺寸冒烟；不能用于论文定量结论
& $py simulations/DeePIE.py run --preset smoke --device cuda --iters 2 --eval-every 1 --outdir results_paper/deepie_smoke

# 完整几何的初步重建；500 为数百次更新的实现选择，不是已验证最优值
& $py simulations/DeePIE.py run --preset paper --device cuda --iters 500 --eval-every 25 --outdir results_paper/deepie_paper_s0

# 建议先做的保守诊断：只降低物体网络 LR，保留探针 LR=1e-4
# 这不是论文原始 LR，也未经过长程收敛验证
& $py simulations/DeePIE.py run --preset paper --device cuda --lr 1e-6 --iters 100 --eval-every 10 --outdir results_paper/deepie_pilot_s0

# 换成你实际的 baseline result 路径；网络种子独立于场景种子
& $py simulations/DeePIE.py run --scene-config results_paper/YOUR_RUN/net_result.npz --device cuda --network-seed 0 --iters 500 --outdir results_paper/deepie_matched_s0

# 明确的无权振幅 L2 对照，去掉尚未确定的梯度策略
& $py simulations/DeePIE.py run --preset paper --device cuda --scan-weight uniform --balance none --iters 500 --outdir results_paper/deepie_uniform_s0

# 测试，不进行长时间重建
& $py -m unittest discover -s simulations -p test_deepie.py -v
```

学习率 1e-4 来自正文，**但作者的 Λ 初始化、基尺度与梯度策略不明，因此不能保证此值
在独立实现中稳定或最优**。如果损失持续增长，先看实际保存的曲线；应明确记录并在
所有评价场景前固定调参协议，不能用真值为每个测试实例挑最有利结果。
可单独改变 `--lr`（物体网络）和 `--probe-lr`（像素探针），其余网络结构保持不变。

现阶段只做短验证，未执行 500 次更新，未确认重建质量或复现原文指标。
遵循仓库 `CLAUDE.md`，不自动启动长时间仿真。

## 2026-10-05 本机验证记录

8 项 unittest 全部通过：Fourier 基/参数、分块与直接 autograd 的复梯度、
零残差梯度、只依赖测量的初始标定、场景逐元素一致、旧结果配置导入和参数覆盖、
梯度平衡、更新后指标/NPZ/checkpoint 重载一致性。
测试仅使用很小场景、至多两次更新。

最终代码在 RTX 3060 Laptop 6 GB、PyTorch 2.5.1+cu121 上的默认 paper 几何：
物体 612²，探针/探测器 512²，100 个扫描点；数据指纹 `9f181b7d2a66`。

| 运行 | 更新数 | 循环耗时（含评价） | 峰值 allocated memory | 加权振幅损失 |
|---|---:|---:|---:|---|
| 论文默认 LR=1e-4 | 1 | 1.06 s | 619 MiB | 9.75067e-4 → 1.19023e-2 |
| 网络 LR=1e-6、探针 LR=1e-4 | 2 | 1.78 s | 624 MiB | 9.75067e-4 → 9.82189e-4 → 9.72891e-4 |

小 LR 诊断的更新后无权振幅 MSE 分别为 9.68937e-4、9.59585e-4。
这些结果只确认功能与显存可行性；默认 LR 首步明显跳升，小 LR 也不是逐步单调下降。
**尚未得到收敛的物体/探针，不能据此报告 DeePIE 的最终精度、优劣或完整运行时间。**
极短计时不用于推算论文规模长程训练速度，GPU 环境及评价频率都会影响结果。
本地原始日志/图/配置/checkpoint 在 `tmp/deepie_validation/paper_final/`
和 `tmp/deepie_validation/paper_low_lr/`；这些临时运行产物不作为源码提交。

## 输出与连续坐标推理

- `deepie_manifest.json`：来源、实现选择、场景指纹、源码 SHA256、模型/训练参数、
  实际参数量、状态、耗时和 GPU 峰值 allocated memory。
- `deepie_result.npz`：与原保存布局兼容的 obj_rec/obj_gt/probe_rec/probe_gt/roi/positions/hist/cfg，
  额外保存 model_config/training_config/scene_fingerprint/initial_probe_scale。
- `deepie_result.png`、`deepie_convergence.png`：共用布局的结果与曲线。
- `deepie_history.json`：每个评价节点的**更新后**损失与指标、更新前损失、梯度范数和 LR。
- `deepie_checkpoint.pt`：全部模型参数和固定基、探针、Adam 和 scheduler 状态。
  当前入口不提供 resume；保存这些状态用于审计及后续扩展。
- `deepie_scene.npz`：仅 `--save-scene` 时生成。

同目录已有 manifest 会拒绝覆盖，请为新实验选择新目录。
训练计时包含循环内评价，不含生成数据、初始尺度校准和最终存盘/画图。
GPU allocated memory 不含驱动、桌面或其他程序占用，不能当作 nvidia-smi 总显存。

加载自己生成的 checkpoint 后可任意取样，无需再次优化：

```python
import torch
from functions.paperrepro.deepie_model import DeePIEObject, ModelConfig, coordinate_grid

state = torch.load('results_paper/deepie_paper_s0/deepie_checkpoint.pt',
                   map_location='cpu', weights_only=False)
model = DeePIEObject(ModelConfig(**state['model_config'])).eval()
model.load_state_dict(state['model'])
side = state['scene_config']['obj_size'] * 2
coords = coordinate_grid(side, 'cpu')  # same field of view at twice the sampling density
field = model.field(coords, side, chunk=4096, effective=model.materialize())
```

连续插值不等于数据已经支持额外空间分辨率，不把此输出自动称为真实超分辨。
