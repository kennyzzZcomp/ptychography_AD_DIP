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

## 诊断“停在常数物体”的顺序

新增 `simulations/DeePIE_diagnose.py`。以下诊断不修改原训练入口。
诊断测试已加入 `simulations/test_deepie.py`，当前合计 12 项测试通过。
以下假设待查的 Colab 实验目录是截图中的 `results_paper/deepie_4`。
同步新增诊断脚本后，在 Colab 项目根目录运行；每次换一个新的诊断输出目录。

### A. 审计已有结果：不训练

```python
!python simulations/DeePIE_diagnose.py audit \
  --result results_paper/deepie_4/deepie_result.npz \
  --outdir results_paper/deepie_4_audit
```

直接使用保存的 obj_rec/obj_gt/roi，不重新仿真，也不加载 pickle checkpoint。
输出 `diagnostic.json` 和 `constant_comparison.png`：

- `gain_over_constant_db`：比同一 ROI 的常数物体高多少 dB。
- `object_statistics.relative_distance_to_constant`：
  ||O-mean(O)||/||O||，越接近 0 越接近空间常数，不受非零全局复尺度影响。
- `amplitude_cv`：振幅标准差/平均值，配合真值统计看对比度是否丢失。
- 原始训练配置，以及已有 history 中实际使用的 LR。

PSNR 接近常数基线并不能单独证明物体恒定；必须结合空间变化统计或重建图。
这些统计也不能单独证明损失完全由探针下降，需要探针冻结/物体冻结对照。

### B. 绕开衍射，直接拟合已知物体

```python
!python simulations/DeePIE_diagnose.py fit-object \
  --from-run results_paper/deepie_4 \
  --device cuda --lr 1e-6 --iters 100 --eval-every 10 \
  --outdir results_paper/deepie_4_fit_object
```

继承原 manifest 中的网络配置和网络种子，但从相同随机初始化重新开始，
不载入已经停滞的网络权重。直接最小化振幅 MSE + 相位弧度 MSE，
默认只拟合评价 ROI 内的坐标（保留原全局坐标，不重新放大局部坐标）。
`--fit-domain full` 可切换到整幅画布。该损失对 wrapped GT phase 的跳变并不平滑；
当前默认 +/-0.8 rad 场景没有这种跨 +/-pi 跳变。

诊断默认显式关闭 LR 衰减、扫描加权、梯度平衡，避免未确认启发式干扰；
CLI 仍可覆盖这些值（扫描加权对直接拟合模式不生效）。
`--lr` 是网络 LR；`--probe-lr`、`--probe-scale` 不控制这两个诊断实验。
记录初始状态、第 1 步、指定间隔和最终状态；同时记录每层有效 W 的原始 RMS、
更新 RMS 和相对更新量。特别小的初始 W 会放大相对更新量，必须同时看绝对 RMS。

如果直接拟合也不能降低目标损失、学出结构，应先检查网络参数化和优化条件，
不应归因于衍射前向或盲探针。100 步失败不证明网络缺乏表达能力，可能是预算/LR不合适。
ROI 直接拟合通过也不等于全幅拟合或盲重建已经通过。

### C. 固定正确探针，只从衍射图重建物体

```python
!python simulations/DeePIE_diagnose.py known-probe \
  --from-run results_paper/deepie_4 \
  --device cuda --lr 1e-6 --iters 100 --eval-every 10 \
  --outdir results_paper/deepie_4_known_probe
```

使用与原仿真一致的真值探针，并用真值前向的全局最大强度恢复其正确幅度尺度；
探针不参与优化。这是明确使用真值的非盲诊断，不是可用于正式对照的 baseline。
`truth_clean_amplitude_mse` 检查真值对干净数据的前向一致性，应接近数值精度。
有噪声时训练使用原含噪数据，不能要求真值在含噪数据上达到零损失。

若 B 通过、C 停滞，再查衍射损失下的网络优化、尺度和数据可辨识性；
若 C 明显有效、原盲重建仍停滞，优先检查联合探针优化。
这些诊断为了降低难度更改了训练规则，不构成对某一个启发式的单变量归因；
定位范围后应逐项恢复原设置验证。

B/C 都输出 `diagnostic.json`、`history.json`、`diagnostic_fields.npz`，
并标记 `not_a_blind_baseline=true`。默认会根据原配置重新生成仿真，
打印新旧指纹；如果不同，应先核查素材、参数覆盖和软件/设备差异。
本地只完成小场景单步测试，没有自动执行以上两项 100 步实验。

### D. 固定正确探针，直接优化复数物体像素

当 C 的损失下降而图像指标不改善时，使用这一对照：

```python
!python /content/ptychography_AD_DIP/simulations/DeePIE_diagnose.py pixel-known-probe \
  --from-run results_paper/deepie_7 \
  --device cuda --lr 1e-2 --iters 100 --eval-every 10 \
  --outdir results_paper/deepie_7_pixel_known_probe
```

D 用相同配置、种子的网络生成与 C 一致的初始复数物体，然后移除网络，
用 Adam 直接优化每个复数像素；固定探针、数据、前向、损失、评价口径与 C 相同。
像素 LR 默认 1e-2，不继承网络系数的 LR；显式 `--lr` 可覆盖。
两种参数的尺度不同，此实验比较优化路径是否可行，不用于等步数速度排名。
梯度平衡仅适用于网络，D 不使用它。仍然是使用真值探针的诊断，不是盲重建。

如果 D 能明显恢复结构而 C 不行，优先检查网络参数化及优化设置；
若 D 也不行，仍需检查优化预算/步长、测量约束及评价对齐，不能仅凭 100 步失败断言前向错误。
新版 C/D 在 history 中保存 `relative_amplitude_mse`：
未加权振幅残差均方除以测量振幅均方，避免把小绝对损失误认为充分拟合。
`truth_clean_amplitude_mse` 同时打印，用于核验真值是否满足模拟的干净测量。
本地只运行小场景单步测试，100 步实验由用户在 Colab 执行。

## 可选小窗口、广域扫描：ptyinr-layout

`--preset ptyinr-layout` 仅增加在 DeePIE 入口，不修改 ProPtyNet 默认配置。
它借用官方 PtyINR 默认示例的数组和扫描布局：物体 241²、窗口 64²、步长 3、
60×60=3600 个位置。仍使用共享场景的 USAF/siemens、圆孔纹理探针和负号 Fresnel 前向，
不是 PtyINR 官方 X-ray 数据/探针，也不是 DeePIE 论文原始仿真。

默认保留 λ=632 nm、z=0.165 m、探测器像素 15.04 µm、探针直径 800 µm。
因此物面像素由约 13.54 µm 变成 108.34 µm，探针直径约 7.38 像素。
窗口线性重叠 95.31%，不能当作光斑重叠（直径口径约 59.4%）。
USAF/siemens 会重新采样到新物体尺寸；新实验同时改变视野、采样和扫描数，
是场景敏感性实验，不是单变量消融，也不能直接排名原场景 PSNR/速度。
网络配置、位置编码和训练默认参数不会随该 preset 自动改变。

Colab 项目根目录运行（先同步 DeePIE.py）：

```python
!python simulations/DeePIE.py check --preset ptyinr-layout --device cuda

!python simulations/DeePIE_diagnose.py known-probe \
  --from-run results_paper/deepie_7 --preset ptyinr-layout \
  --device cuda --lr 1e-6 --scan-chunk 64 \
  --iters 100 --eval-every 10 \
  --outdir results_paper/deepie_small64_known
```

`--from-run` 继承旧网络/种子但重新初始化，新 preset 覆盖几何参数；
原实验与新实验的指纹不同是预期现象。known-probe 的正确探针来自新场景，
不加载旧探针；诊断照常关闭衰减、扫描加权、梯度平衡。
如果旧目录不存在，删掉 `--from-run results_paper/deepie_7`，使用当前默认网络。

盲重建入口同样支持该 preset，例如沿用先前 1e-6/1e-3 学习率组合：

```python
!python simulations/DeePIE.py run --preset ptyinr-layout \
  --device cuda --lr 1e-6 --probe-lr 1e-3 --scan-chunk 64 \
  --iters 500 --eval-every 25 --outdir results_paper/deepie_small64_blind
```

可选 `--sample-pixel-um 13.542012965425533` 保持原物面像素尺度，
此时程序推导探测器像素约 120.32 µm，光斑恢复约 59 像素。
这会改变探测器采样，不能继续使用旧测量；本入口会重新生成数据。
不能同时指定 `--sample-pixel-um` 与 `--det-pixel`。manifest 会保存推导后的实际光学参数。
两个尺度不是谁更正确：前者保留原探测器采样，后者保留原物面采样。

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
