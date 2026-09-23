# 渐进测量约束 pilot：高重叠数据，先稀疏再补全

## 10×10 → 先4×4，再全部100张（2026-09-24）

Colab 拉取代码后，可直接运行可编辑配置脚本：

```python
!python /content/ptychography_AD_DIP/simulations/run_paper_progressive.py
```

在 `simulations/run_paper_progressive.py` 顶部修改参数；默认如下：

| 更新次数 | 输入和训练测量 | obj amp TGV | net LR | probe LR |
|---|---|---|---|---|
| 1–1000 | 行、列各取0/3/6/9，4×4=16张 | 0.1 | 0.005 | 0.02 |
| 1001–4000 | 全部10×10=100张 | 0 | 0.0008 | 0.02 |

`TOTAL_ITERS` 是总步数，`SWITCH_AFTER` 是已经完成的更新数；第二阶段从下一次更新开始。
probe 始终为自由像素，不加support；默认 `PROBE_INIT="disk"`，可自行调整。
`STEP_PX=12` 时主数据线性重叠约79.7%，子集实际步长36px，约39.1%，不是精确40%。
10×10/step12需要至少620像素的物体画布，默认624。新旧比较请统一场景、ROI、seed等。
4×4子集包含主网格四个角，扫描中心的外接范围相同，但照明覆盖/冗余仍不同。

等价直接命令：

```python
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net \
    --preset paper --obj-size 624 --grid 10 --step-px 12 \
    --iters 4000 --eval-size 96 --eval-every 25 --seed 0 --device cuda \
    --probe-mode pixel --probe-init disk \
    --measurement-schedule "0:3,1000:1" --measurement-policy fixed \
    --input-policy follow_measurements \
    --tgv-amp 0.1 --tgv-phase 0 --tgv-amp-schedule "1000:0" \
    --lr-net 5e-3 --lr-probe 2e-2 --lr-schedule "1000:8e-4:2e-2" \
    --outdir /content/ov80_grid10_progressive
```

注意事项：

- 不加 `--lr-cosine`；它与显式 `--lr-schedule` 互斥，避免两个调度静默叠加。
- 切换保留网络、probe和Adam状态；新通道的加入仍可能使输出跳变，不保证平滑。
- 第一阶段实际只将16张输入首层卷积，并仅用这16张的测量数据反传。
  全部100张已加载；全量数据仍用于不反传的评价、TGV固定域和归一化尺度。
- TGV为0时停止辅助变量更新及TGV损失计算。日志中该阶段TGV值记0表示未计算，
  不代表当前物体的数学TGV值恰为0；`tgv_amp_active`区分这两种情况。
- 输出仍有 `net_result.npz`、`net_convergence.png` 及训练循环耗时；
  另有 `measurement_schedule.json`、`tgv_amp_schedule.json`、`lr_schedule.json`。
  NPZ的hist保存采样数、实际学习率和TGV状态，便于核对同步切换。
- 不承诺加速或更好重建。训练时间包含循环内评价，排除场景生成和最终存盘/画图。
- 启动脚本拒绝覆盖已有输出目录；重跑请修改 `OUTDIR`，不会删除旧结果。

以下保留之前9×9实验的说明，不是本次10×10默认配置。

这是用户“利用低重叠可恢复能力，加速高重叠训练”想法的第一个隔离实验。
默认**固定全部U-Net输入通道，仅改变物理损失中的测量集合**。因此默认早期并非只看到了
稀疏数据，也不声称减少采集剂量。新增`--input-policy follow_measurements`可让
实际输入通道随测量子集一起渐进，见后文第五组对照。
不改z、场景、GT、探针设置；不是重启网络，补全阶段继续使用已有网络和优化器状态。

## Colab四组实验

先同步代码；不要和旧的40% grid4结果直接比较。这里四组全都使用同一高重叠主数据集：
grid9、step12、obj624。N512+(9-1)*12=608，不超出画布。
名义probe直径约59px，所以主数据约80%重叠；stride3的子集间距36px，约39%重叠。
早期9帧、后期81帧。取0,3,6行列，早期照明范围比全量小；这也是要报告的局限。
不要混用原来grid4/step35的实验作为同数据对照。

```python
import subprocess, sys
from pathlib import Path

script = "/content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py"
root = Path("/content/drive/MyDrive/ProPtyNet/curriculum_pilot")  # 先挂载Drive
experiments = [
    ("full", "0:1", "fixed"),
    ("fixed_to_full", "0:3,600:1", "fixed"),
    ("random_to_full", "0:3,600:1", "random"),
    ("rotating_to_full", "0:3,600:1", "rotate"),
]
for name, schedule, policy in experiments:
    out = root / name
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {out}")
    subprocess.run([
        sys.executable, script, "net", "--preset", "paper",
        "--obj-size", "624", "--grid", "9", "--step-px", "12",
        "--iters", "2000", "--eval-size", "96", "--eval-every", "25",
        "--seed", "0", "--probe-mode", "pixel", "--probe-init", "disk",
        "--tgv-amp", "0.1", "--tgv-phase", "0", "--device", "cuda",
        "--measurement-schedule", schedule,
        "--measurement-policy", policy, "--measurement-seed", "0",
        "--outdir", str(out),
    ], check=True)
```

600步切换只是预设试验值，非已验证最佳值。先比较相同日程的fixed/random/rotate，
不按测试GT为每个方法挑最好切换点。之后再考察不同切换点与仅random minibatch等控制。
所有组都保持相同TGV，第一轮不同时搜索相位权重。

## 参数与目标函数

- 不传 `--measurement-schedule`：旧路径，不新增逐步测量/运行时间记录。
- `0:1`：显式全量基线，记录时间与计数。
- `0:3,600:1`：0-based第0–599步stride3，第600步起stride1。
- 多阶段示例 `0:4,300:2,900:1` 要求grid可被4整除；这不是上面grid9的命令。
- fixed允许stride不整除grid，取range(0, grid, stride)；random/rotate仍要求整除。
  下一stride须整除前一stride，最后必须是1且实际运行到。
- fixed：固定余类0，逐阶段嵌套补全；random：每步同规模均匀无放回抽样；
  rotate：逐步轮换行列余类，stride3每9步覆盖全部81帧。
- measurement-seed只控制选图，不改变场景与网络初始化。原seed仍同时影响场景探针和网络。

## 第五组：输入与物理测量共同渐进（用户原始想法）

在相同Colab目录和主数据集上运行：

```python
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net --preset paper --obj-size 624 --grid 9 --step-px 12 --iters 2000 --eval-size 96 --eval-every 25 --seed 0 --probe-mode pixel --probe-init disk --tgv-amp 0.1 --tgv-phase 0 --device cuda --measurement-schedule "0:3,600:1" --measurement-policy fixed --input-policy follow_measurements --outdir /content/drive/MyDrive/ProPtyNet/curriculum_pilot/joint_fixed_to_full
```

绘图前在`experiments`列表中加入：

```python
experiments.append(("joint_fixed_to_full", "0:3,600:1", "fixed"))
```

`input-policy full`是默认；`follow_measurements`会把输入与首层卷积权重按同一索引
选出后调用较少输入通道的卷积，**不是仍以81通道做稠密卷积的简单置零**。
前600步实际首层输入9通道，之后恢复81通道。其余U-Net层宽度和全图输出不变。
完整数据、全尺寸首层参数和优化器状态仍保留在显存中，不能宣称按输入比例节省总显存。
收集输入/权重张量也有开销，是否快需实测，不从通道比值推导端到端加速比。

所有通道权重在最初按原网络方式初始化，各扫描帧始终对应同一权重通道；切换时不重建
网络、不更换Parameter、不重建Adam。新加入通道会带来新的响应，因此**输出不保证连续**，
BatchNorm统计也可能在切换后调整。这是最朴素的渐进输入基线，不假装已经解决切换冲击。
随机/轮换选图也保持通道身份；非活动通道的数据梯度为零，但过去的Adam动量及非零
weight_decay仍可能影响其权重，不声称整个参数切片完全冻结。

相同初始化下，显式全量`0:1`的follow_measurements走原卷积路径；微型测试验证其
训练数值与full一致。子集卷积输出/梯度与密集置零输入等价，但实际卷积维度更小。
这里early loss的正则缩放仍用全部已采数据mean(I)，评价也使用全部测量（不反传）；
所以它是已采高重叠数据上的计算日程，而不是严格禁止访问剩余数据的在线采集实验。

hist新增`active_input_channels`；collect标签包含input_policy，防止和仅loss渐进混合。
建议先比较full、fixed_to_full与joint_fixed_to_full，再补joint_random/rotate，判断新增
输入调度的收益。若joint不如loss-only，不应将loss-only的收益归因于输入端。

每个阶段使用**当前子集的全局幅度VarPro**标定，数据项取子集像素均值。
这不是全量VarPro目标的无偏随机梯度；不同阶段/政策是不同早期优化路径，最终回到同一全量目标。
同一mini-batch基线也必须采用一致标定定义。TGV域和mean(I)缩放始终由主数据集确定，
不随子集变化，避免把正则域/强度一起改变。
评价用全量预测重新拟合统一标度，不将子集训练loss冒充全数据误差。
这些评价预测不反传，但信息可用于日志；如将其用于自适应切换，需另外明确算法定义。

## 质量–时间曲线

```python
import json
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for name, _, _ in experiments:
    with np.load(root / name / "net_result.npz", allow_pickle=False) as z:
        h = json.loads(z["hist"].item())
    times = [r["elapsed_s"] for r in h]
    axes[0].plot(times, [r["psnr_amp"] for r in h], label=name)
    axes[1].plot(times, [r["ssim_phs"] for r in h], label=name)
axes[0].set_ylabel("Amplitude PSNR (dB)")
axes[1].set_ylabel("Phase SSIM")
for ax in axes:
    ax.set_xlabel("Elapsed reconstruction time (s)")
    ax.grid(alpha=.3)
    ax.legend()
fig.tight_layout()
fig.savefig(root / "quality_vs_time.png", dpi=160)
plt.show()
```

`elapsed_s`包含训练循环、选图、TGV和评价，排除scene生成/初始化及最终保存绘图。
开始计时前同步CUDA，评价处已有CPU取值同步。它不是含加载数据的端到端总运行时间。
保持同GPU、日志/评价频率、精度，并重复计时以控制Colab波动；不要开启分阶段同步计时
`--timing-warmup`后把其耗时直接当作生产吞吐量。
沿用原实现：第it条评价对应该次更新前的预测，耗时包含这一步优化器更新；所有组一致，
指标不是额外更新后再decode的结果。需要更细粒度time-to-quality时应缩小统一评价间隔。

hist新增：active_patterns、measurement_stride、training_patterns_cumulative、
evaluation_patterns_cumulative（仅额外全量评价前向）、full_data_loss、elapsed_s。
训练与评价计数只计传播帧数，不是FLOPs，反向及网络开销仍由墙钟体现。
`measurement_schedule.json`记录每步实际索引与策略。collect标签区分日程与policy，
但不同measurement-seed需单独管理目录，不要误当不同scene/网络seed。

继续条件：最终质量接近全量基线，且达到相同振幅与相位质量更快，并优于随机控制；
如果只胜固定全量却不胜random，最多支持批次优化的收益，不能证明固定跳点特别有效。
如早期偏差导致后期长时间补救，或网络成本支配导致不加速，这是应保留的否定证据。
本地仅运行微型CPU接线/梯度测试，没有执行这里的完整训练，尚无GPU加速结论。
