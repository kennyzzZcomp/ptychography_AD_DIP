# ptydata_33：真实数据两阶段重建

独立实验入口，不调用或修改原仿真脚本。**第一阶段 U-Net16＋TGV；第二阶段 U-Net32 复数加法残差＋覆盖加权 NLM，不加 TGV。** 前向采用指定的夫琅禾费模型，两个阶段均使用振幅损失。

## 可选消融：第一阶段内部特征归一化（2026-10-19）

新增 `--stage1-feature-norm none|pixel-rms`，默认`none`保留既有行为。显式选`pixel-rms`，只在S1最后一个解码块之后、振幅/相位头之前，对每个位置的通道特征做RMS归一化，固定RMS下限1e-6。没有新增可训练参数；初始O/P、物理测量、g=1、FFT、坐标、学习率和正则规则不变。S2仍为原网络，不启用该归一化。

先同步本目录的`run.py`和`core.py`到Colab，保留原`initialization.py`，在原PyTorch GPU环境运行。以下保留用户最近的500+200、TGV=0、NLM=0配置，只打开此候选：

```python
!python -u /content/ptychography_AD_DIP/real_ptycho/run.py \
  --data "/content/drive/MyDrive/ptyco_data/ptydata_33.h5" \
  --num-frames 225 --detector-crop 1024 --probe-size 0 \
  --initialization mpie --object-sampling mpie-integer \
  --stage1-feature-norm pixel-rms \
  --stage1-steps 500 --stage2-steps 200 \
  --stage1-lr-net 0.0005 --stage1-lr-probe 0.005 \
  --stage2-lr-net 0.0005 --stage2-lr-probe 0.005 \
  --tgv 0 --nlm-ratio 0 --seed 0 --chunk 1 --eval-every 50
```

关闭对照只将上面的`pixel-rms`换成`none`，其他选项完全相同。每次自动新建输出目录；已有同配置的none结果可直接作为参照，不需覆盖。启动日志必须出现`[feature norm] S1=pixel-rms; S2=none`；配置/权重文件均记录模式。

23项小尺寸测试通过，含候选与原诊断hook的输出/梯度一致性、初始化保持、零特征向量梯度有限，以及mpie/ones各4+2次合成更新。完整500+200重建需用户在Colab手动运行；目前只有前4步诊断中峰值受抑制的证据，且早期loss下降较慢，不能据此保证最终USAF更清晰。

## 当前消融：mPIE 整数切片，保留 g=1 和原初值（2026-10-18）

默认新增 `--object-sampling mpie-integer`：毫米转米、按坐标轴选项换轴/翻转、减全部所选坐标的均值、除以物面像元，使用 `np.round` **先取整，再加** PtyLab 的 `No//2-Np//2`。物体局部场直接切片，不做双线性插值。`--object-sampling bilinear` 可明确复现上一版。

为不改变网络大小和初始数组，仍保留 1440² 网络画布及原先从 mPIE 2454² 初始物体裁取的数组。当前数据的坐标映射是 `mPIE位置 = 网络位置 + (510,510)`；并非把分数位置直接 `round`。全部225个位置与成功 mPIE 文件一致，初始物体/探针与用户上传的旧 `initialization.npz` 逐元素相同，225个初始局部块与参考大画布切片逐元素相同。参考位置和偏移写入 `config.json` 的 `sampling_alignment`；NPZ 的 `positions_px` 始终对应其自身物体数组。

**复跑用户最近一次上传结果的配置**（500+200，TGV=0、NLM=0，两阶段网络0.0005、探针0.005），只改变取样：

```python
!python -u /content/ptychography_AD_DIP/real_ptycho/run.py \
  --data "/content/drive/MyDrive/ptyco_data/ptydata_33.h5" \
  --num-frames 225 --detector-crop 1024 --probe-size 0 \
  --initialization mpie --object-sampling mpie-integer \
  --stage1-steps 500 --stage2-steps 200 \
  --stage1-lr-net 0.0005 --stage1-lr-probe 0.005 \
  --stage2-lr-net 0.0005 --stage2-lr-probe 0.005 \
  --tgv 0 --nlm-ratio 0 --seed 0 --chunk 1 --eval-every 50
```

先同步更新 `core.py` 和 `run.py` 到 Colab，保留原 `initialization.py`。使用原 PyTorch GPU 环境，不需要 PtyLab 环境。启动日志应出现 `[sampling] mpie-integer` 以及 `[loss] ... g=1 FIXED`。本次不改原有训练参数默认值，所以不用上面显式参数时仍会运行默认500+500、TGV=0.01和NLM ratio=0.5；这不等同于用户最近的无正则对照。21项小尺寸测试通过，尚未运行完整实验训练。

## 上一次消融：固定振幅增益 g=1（2026-10-18，历史说明）

两阶段训练、评估以及 NLM 梯度比校准统一使用 `mean((A-B)**2)`，其中 `B=sqrt(I_normalized)`；不再拟合增益。启动日志打印 `[loss] ... g=1 FIXED`，配置保存 `amplitude_gain=1.0`；历史日志保留旧字段 `profile_gain`，其值恒为 1，另记 `gain_mode=fixed`。

只改变此损失尺度处理：保留下面的 mPIE 默认初始化、完整 1024² 探针、双线性亚像素物体提取、数据归一化、全部学习率、TGV、NLM ratio 和 500+500 更新次数。NLM ratio=0.5 及自动校准规则不变，但数据梯度变了，实际校准 lambda 可能随之改变；不固定为旧运行数值。新旧 rRMSE 定义不同，不直接用其大小判定图像质量提升。

同步更新后的 `core.py`、`run.py`（并保留上一版 `initialization.py`）后，运行下面 A 命令即可。每次新建时间戳输出目录，不覆盖旧结果。只做小尺寸测试，未自动运行正式实验重建。

## 初始化对照：mPIE 初值与仿真全 1 初值（2026-10-17）

此前初始化更新设置为：默认 225 点、1024² 中心裁剪、完整 1024² 探针；默认 `--initialization mpie`。不更改网络宽度、分组、前向、亚像素采样、500+500 更新次数、TGV=0.01、NLM ratio=0.5 或其刷新规则。当前有效学习率保持两个阶段网络 0.0005、探针 0.005。以下命令在当前代码上均使用固定 g=1。

同步 `run.py`、`core.py` 和新增的 `initialization.py` 到 Colab，再在原来运行两阶段方法的 **PyTorch 环境**执行。不要用只安装了 PtyLab 的 `/content/ptylab_mpie_env/bin/python`，该环境不包含 PyTorch。若曾在 Colab 主环境混装 CUDA 包，先确认 `torch` GPU 可用；本次不自动安装、卸载或修改环境。

**A：与此前 mPIE 对齐的初值**（默认）：

```python
!python -u /content/ptychography_AD_DIP/real_ptycho/run.py --data "/content/drive/MyDrive/ptyco_data/ptydata_33.h5" --num-frames 225 --detector-crop 1024 --probe-size 0 --initialization mpie
```

**B：沿用仿真的严格全 1 探针初值，作为原方法初始化基线**：

```python
!python -u /content/ptychography_AD_DIP/real_ptycho/run.py --data "/content/drive/MyDrive/ptyco_data/ptydata_33.h5" --num-frames 225 --detector-crop 1024 --probe-size 0 --initialization ones
```

两条命令都从头运行、各自新建时间戳目录；不要并行跑而挤占显存。若此前命令指定了其他阶段学习率或正则参数，两条命令都应追加完全相同的这些参数。本次没有改变这些参数的解析语义。

- `mpie`：复现 PtyLab 0.3.1 的 `ones` 物体与 `circ` 探针，包含其 `1+0.001*uniform(0,1)` **实数振幅微扰**和零相位。圆直径为探针视场的 1/3，在本设置下约 336.596 μm；不做 RMS 重归一化，不使用 mPIE 重建结果或真值。
- 使用与参考 mPIE 相同的 NumPy 随机流顺序：先建立参考大物体，再建立探针，从而能对齐完整探针的随机微扰。网络仍使用自己的 1440² 画布，物体初值从参考 2454² 初始画布按最近整数原点裁取；原点小数残差写入配置。此前保留亚像素取样；当前默认整数切片对齐见文首，初始数组不变。
- 网络的零输出头仍保留；用一个固定、不可训练的振幅 logit 偏移实现上述物体初值，数值精度范围内一致，未增添训练参数。S2 不受此偏移影响，仍是 `O1 + complex residual`。
- `ones`：物体和完整探针都严格为 `1+0j`，不加噪声，不做扰动或归一化。纯远场下此初值可能产生退化的单点傅里叶谱、难以启动优化；即使出现也如实记录，**不会偷偷添加扰动**。这是仿真初始化规则的迁移基线，不代表真实数据与仿真物理条件相同。
- 圆形只定义初值，**不是每步施加的支持域**。两阶段中完整 1024² 探针所有像素都可优化，圆外也可更新。S2 继承 S1 的探针和 Adam 状态，不重新设回初值。
- `legacy-ifft`：仅供明确复现旧入口；可配 `--probe-size 512`，不是默认设置。`mpie/ones` 会拒绝 512² 等缩小支撑的配置，防止误把它当成对齐实验。
- 每次保存 `initialization.npz`：实际网络初始物体、请求的初始物体、完整探针和坐标。`config.json` 保存初值规则、参考画布裁剪、原点残差和实际初始场误差，以便审计。

## 数据放哪里、怎么运行

将整个 `real_ptycho/` 代码文件夹随仓库推到 GitHub。数据、权重和输出已通过本文件夹的 `.gitignore` 排除，**不需要上传到 GitHub**。

在 Colab 克隆仓库后，把 H5 上传到：

```text
<仓库目录>/real_ptycho/data/ptydata_33.h5
```

例如仓库位于 `/content/ptychography_AD_DIP`，则完整数据路径为：

```text
/content/ptychography_AD_DIP/real_ptycho/data/ptydata_33.h5
```

先切换到仓库根目录，选择 GPU runtime（例如 L4）。创建数据文件夹并安装依赖：

```python
!pip install -r real_ptycho/requirements.txt
!mkdir -p real_ptycho/data
```

把H5上传到该文件夹后，再运行：

```python
!python -u real_ptycho/run.py
```

**不需要 REF、仿真 scene H5、无噪声 GT 或其他 NPZ。** 默认读取上述数据路径；本地也会自动查找旧位置 `experiment/ptydata_33.h5`。若文件放在 Drive 等其他地方，直接指定真实路径：

```python
!python -u real_ptycho/run.py --data "/content/drive/MyDrive/ptydata_33.h5"
```

默认使用完整 **15×15＝225 个真实扫描点**，运行 **500＋500 次 optimizer update**，每张测量中心裁剪到 **1024×1024**。结果自动保存在 `real_ptycho/results/<时间戳>/`，不会覆盖旧结果。原生2000²需显式 `--detector-crop 0`。

显式指定全225点（原100点或200点模式仍可分别用 `--num-frames 100`、`--num-frames 200`）：

```python
!python -u real_ptycho/run.py --num-frames 225
```

225点模式保留全部15×15扫描位置，按栅格顺序排列，并保留与原始衍射帧的对应关系及坐标抖动，不排除四角。ABCD按行列奇偶划分，为 **64／56／56／49** 张。网络输入统一64通道（U-Net基础宽度仍是16），短组尾部补零通道；**物理损失只使用真实帧，不重复、不填补虚拟测量**。原100点模式保留原10×10选帧和25通道顺序；200点模式仍是中心最近200点。更改点数后需从头训练，旧25/52通道S1权重不能直接加载为64通道网络。

**原始坐标按用户确认的 mm 读取，不乘10。** 225点、原生2000²测量对应物体画布2800²，裁剪1024²则为1440²。位置单位与换算系数写入`config.json`。现在默认完整探针，不限制512²；以下小探针说明仅为旧配置复现记录。

历史配置复现（不是当前推荐的两组初始化对照）：

```python
!python -u /content/ptychography_AD_DIP/real_ptycho/run.py --data "/content/drive/MyDrive/ptyco_data/ptydata_33.h5" --num-frames 225 --detector-crop 0 --probe-size 512 --initialization legacy-ifft
```

S1/S2都只优化512²探针，外围始终为零，跨阶段保留该小探针的Adam状态。初始化沿用原平均衍射振幅逆FFT及全尺寸RMS归一化，再截取中央512²，**不按小窗口额外重归一化**。分块计算返回全尺寸探针梯度后，仅取中央512²更新参数。`--probe-size 0`可恢复全尺寸探针对照；其他学习率、迭代次数、TGV、NLM和物体画布定义不变。512²硬支撑是本次实验假设，不保证真实探针尾部都在其中。

只检查选帧和坐标、不训练：

```python
!python -u real_ptycho/run.py --prepare-only
```

短流程自检（明确裁剪为256×256，探针64×64、仅4＋2次更新，**不是正式结果**）：

```python
!python -u real_ptycho/run.py --smoke --probe-size 64 --initialization legacy-ifft
```

## 固定配置

| 项目 | 本实验设置 |
|---|---|
| 原数据 | `ptychogramx`: 225×2000×2000 强度；`positionx`: 2×225，直接按mm读取，不乘10 |
| 选帧 | 默认全部225点；`--num-frames 100`恢复中心10×10，`--num-frames 200`为中心最近200点 |
| 本数据选中范围 | 完整15×15，具体位置及原帧号见`selected_frames.csv`，row/col为原栅格的零起编号 |
| ABCD | 225点按原栅格行列奇偶分组，64/56/56/49张；100点保留原局部奇偶分组，每组25张 |
| 光学参数 | λ=632.8 nm；样品到感光面 z=6 mm；探测器像元3.76 μm |
| 坐标轴 | 默认 `positionx[0]→y`、`positionx[1]→x`，这是待核实的映射假设 |
| 前向 | 默认 mPIE 对齐的整数切片物体局部场 × 探针 → 中心化、正交归一化二维 FFT → 模长；可显式选择旧 bilinear |
| 采样 | 默认N=1024；物面像元 λz/(NΔd)=0.986120 μm；物体画布由所选坐标范围动态计算 |
| 探针 | 默认完整1024²自由复数像素，无512硬支撑；两个阶段均优化并跨阶段保留Adam状态 |
| 第一阶段 | 共享 U-Net16，保留内部 concat；每次更新一组，ABCD依次轮换；500次=125轮 |
| 第二阶段 | 新 U-Net32，保留内部 concat；`O = O1 + Δreal + iΔimag`；零残差初值；每次更新使用全部所选帧（默认225张） |
| 学习率 | S1/S2分别配置；保持当前有效默认各为网络0.0005、探针0.005 |
| 第一阶段 TGV | 振幅TGV权重0.01；α0=2、α1=1、ε=0.001；辅助场Adam每次5步、学习率0.01 |
| 第二阶段正则 | **TGV=0；只有覆盖加权复数 NLM 约束** |
| NLM | patch=3，search radius=7，h=0.8σ；σ由O1局部差分MAD估计；每25步刷新目标 |
| NLM强度 | 起始时令正则梯度范数/数据梯度范数=0.5，计算一次λ后固定；**不是直接λ=0.5** |
| 网络输入 | S1：该组真实强度图分别按最大值归一化后居中补零，通道补齐到最大组张数（225点为64）；S2：固定O1振幅、cos相位、sin相位 |
| 损失 | `mean((abs(FFT(P·Opatch)) − sqrt(I))²)`；g固定为1，不拟合增益 |

强度预处理只除以全部所选帧的一个共同平均值，保留各帧之间的相对强度；**不另加噪声，不自动扣背景，不进行逐帧曝光归一化**。网络输入的逐通道归一化不改变用于物理损失的测量。

### 两阶段独立学习率

```python
!python -u real_ptycho/run.py --stage1-lr-net 0.0005 --stage1-lr-probe 0.005 --stage2-lr-net 0.0005 --stage2-lr-probe 0.005
```

四个数值可以分别改；此示例只显式写出原默认值，不代表已调优。第二阶段网络使用新Adam，探针保留Adam历史状态，仅切换到`--stage2-lr-probe`的学习率。两个阶段的实际学习率会打印并写入`config.json`。

注意：当前代码的阶段参数已有非空默认值，因此旧的 `--lr-net/--lr-probe` 实际不会作为回退值生效。**本次为保持其他参数不变，没有修复此既有行为**；请使用上述四个阶段专属参数，实际值以启动日志和 `config.json` 为准。

### 与仿真接口不同、必须说明的适配

- 初值现在由显式 `--initialization` 选择，规则见文首；反FFT不再作为默认初始化。
- 未知真实照明直径，mPIE 圆形直径只是初猜，不是测量事实。S1 TGV及四组输出融合仍使用原**计算窗口覆盖并集**，不是测得的照明范围。NLM覆盖权重仍由S1探针的强度累积估计并固定，亦不是GT。
- 原始毫米坐标仍完整记录；默认按 mPIE 顺序居中、取整并转换到原网络画布，直接切片。只有显式 `--object-sampling bilinear` 保留原亚像素取样。
- 为限制显存，网络使用activation checkpoint，传播按帧分块。当前 g 固定为 1，不再执行拟合增益所需的第一遍 FFT；分块累积物体/探针梯度后，只执行**一次**optimizer更新。不是每帧各更新一次。计算工作量与旧的两遍增益拟合实现不同。
- BatchNorm仍用原训练模式；checkpoint重计算和评估读出不会额外改变BN统计量。数学梯度一致性已有小尺寸单元测试，不能据此保证不同GPU逐位一致。

## 输出看什么

| 文件 | 内容 |
|---|---|
| `initialization.npz` | 实际初始物体、请求初始物体、完整初始探针、坐标和配置；不含GT |
| `config.json` | 所有配置、选帧、坐标、数据指纹、设备、实际NLM系数及假设 |
| `selected_frames.csv` | 所有选中位置对应的原始帧编号（0起）、原栅格行列（0起）及ABCD组 |
| `input_preview.png` | 所选栅格及平均衍射强度 |
| `stage1.npz` | 第一阶段复物体，补零探针`probe_rec`和可学习小探针`probe_compact` |
| `reconstruction.npz` | 两阶段物体、补零探针`probe_rec`、小探针`probe_compact`、坐标和配置，无GT字段 |
| `reconstruction.png` | S1/S2振幅、相位及最终探针，全画布显示 |
| `loss_curve.png/.pdf` | 横轴iteration，纵轴全部所选帧的相对振幅RMSE |
| `history.json`、`summary.json` | 测量误差及分阶段训练时间；不含评估、初始化和保存时间 |
| `stage1_weights.pt`、`stage2_weights.pt` | 网络权重、所选尺寸探针实部/虚部及Adam状态，供审计；不是完整断点续训文件 |
| `intensity_cache.npy` | 选中测量的磁盘缓存，225点原生约3.6 GB，200点约3.2 GB，100点约1.6 GB；结果保留时可单独移除缓存 |

真实数据没有GT，**不计算PSNR，也不把loss下降当作分辨率改善的证据**。正式结果需检查USAF条纹可分辨性、测量残差、探针形状，并与可靠的PtyLab重建核对。显示的振幅上限为各图99.5百分位，仅影响预览；NPZ保存未裁剪的数值，跨阶段振幅尺度也存在盲重建歧义。

## 仍需核实的实验条件

1. **“PtyLab使用Fraunhofer”不等于已证明6 mm传播满足该近似。** 当前按要求接入，未验证光路/原PtyLab配置；收敛不代表物理模型正确。
2. 3.76 μm假设2000×2000来自直接裁剪、没有binning或resize；两者若存在，需改有效像元尺寸。
3. 两行坐标与图像x/y方向、符号需核对。可显式用 `--position-order xy`、`--flip-x`、`--flip-y`，不要为得到好看的图隐式换轴。
4. 未知暗场、背景、饱和掩膜及相机增益；不把当前浮点强度当作已标定光子计数。未知探针直径，也不宣称这份数据的重叠率是80%。

### 显存与测试范围

默认1024²并不等于仿真的64/128。225点模式第一阶段输入64通道，按mm计算物体1440²；若显式选原生2000²，物体则为2800²。**尚未在L4验证这两个新初始化配置的完整500＋500重建**。默认`--chunk 1`且开启checkpoint；OOM时会保留已生成文件并报错，不会悄悄降低分辨率。

`--detector-crop 512`等选项是明确的中心裁剪，**会丢掉外围测量、改变物面采样间隔和可用频率范围，不能视为原生2000²等价的省显存版本**。不要为了跑通默认使用它当正式结果。

开发自检：

```bash
python -m unittest discover -s real_ptycho -v
```

测试覆盖分块损失/梯度与整批的一致性、蛇形扫描选帧、毫米单位与亚像素取样、checkpoint的BN状态/梯度、初始场及平坦TGV、NLM梯度及目标刷新。局部短测结果另见 `VALIDATION.md`。
