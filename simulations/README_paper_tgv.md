# Object-amplitude / phase TGV2 for ProPtyNet_paper net

振幅TGV支持 ad（像素优化）和 net（DIP）；相位TGV仅支持 net。振幅与相位有独立开关。
默认 --tgv-amp 0 --tgv-phase 0，沿用原来的损失和优化器更新。

## 原 paper `run`：物体振幅 TGV 计时对照

`run` 现在也接受 `--tgv-amp`。它保留原论文的物体/探针网络、强度域
Loss1 + 探针 Loss2，再对物体实际振幅 `abs(O)` 加同一套 TGV2：

    loss_run = paper_loss + tgv_amp * TGV2(abs(O)/mean_Omega(abs(O)))

`run --tgv-amp 0` 是原基线；非零权重的结果应称为 `paper + TGV`。
`run` 的原损失是 L2 范数，`ad/net` 是振幅 MSE，因此相同的权重数值
不表示相同的相对正则强度。此处 `0.1` 先用于测量开启 TGV 的时间，
不代表已调优的图像质量权重。`run` 仍不支持相位 TGV 或权重调度。

在 Colab 同一 GPU 会话依次运行（输出目录分别新建）：

```python
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py run --preset paper --obj-size 624 --grid 8 --step-px 12 --eval-size 96 --eval-every 25 --iters 2000 --seed 0 --device cuda --tgv-amp 0 --outdir ov80_paper_timing_base
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py run --preset paper --obj-size 624 --grid 8 --step-px 12 --eval-size 96 --eval-every 25 --iters 2000 --seed 0 --device cuda --tgv-amp 0.1 --outdir ov80_paper_timing_tgv01
```

终端显示 `[paper] 用时 ... (ms/it)`；最后一条 NPZ history 还保存
`train_elapsed_s`、`mean_iteration_s`。计时覆盖训练与定期评价，
不包括场景生成、初始化、最终保存/画图。TGV 运行另存 `tgv_aux.npz`，
history 记录 `paper_loss`、`tgv_amp` 和 `tgv_weighted`。

## 新增：AD + 振幅 TGV，以及四组对照

AD对复数物体的精确幅值abs(O)使用现有ObjectAmplitudeTGV，不另造TV或二阶差分近似。
与net共用名义照明域、可微均值归一化、辅助场内迭代、alpha/eps以及mean(I)尺度：

    loss_AD = amplitude_MSE + tgv_amp * mean(I_measured) * TGV2(abs(O)/mean_Omega(abs(O)))

不直接正则化相位或probe；二者仍可能通过数据项耦合间接受影响。辅助场独立优化。
默认关闭时不创建TGV辅助场。AD的lr参数是lr-obj和lr-prb，不是net的lr-net和lr-probe。
AD不支持tgv-phase>0，会明确报错。保存ad_result.npz与tgv_aux.npz，hist字段与net一致。

沿用独立40%场景，Colab直接运行：

```python
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py ad --preset paper --obj-size 624 --iters 2000 --eval-size 96 --step-px 35 --grid 4 --probe-mode pixel --probe-init disk --seed 0 --lr-obj 0.03 --lr-prb 0.03 --tgv-amp 0.1 --tgv-phase 0 --outdir ov40_ad_tgv01
```

四组统一场景的完整单元格（不跳步；如果输出目录已存在则拒绝覆盖）：

```python
import subprocess, sys
from pathlib import Path

script = "/content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py"
root = Path("/content/drive/MyDrive/ProPtyNet/tgv_four_way_ov40")  # 先挂载Drive
for mode, weight, name in [("ad", 0, "ad"), ("ad", .1, "ad_tgv"),
                           ("net", 0, "net"), ("net", .1, "net_tgv")]:
    out = root / name
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite {out}")
    lr_args = (["--lr-obj", "0.03", "--lr-prb", "0.03"] if mode == "ad"
               else ["--lr-net", "0.005", "--lr-probe", "0.02"])
    subprocess.run([
        sys.executable, script, mode, "--preset", "paper",
        "--obj-size", "624", "--grid", "4", "--step-px", "35",
        "--iters", "2000", "--eval-size", "96", "--eval-every", "25",
        "--probe-mode", "pixel", "--probe-init", "disk", "--seed", "0",
        "--tgv-amp", str(weight), "--tgv-phase", "0", "--outdir", str(out),
        *lr_args,
    ], check=True)
```

net学习率取自用户提供的成功40%结果文件（0.005 / 0.02）；AD取当前默认0.03 / 0.03，
不是声称AD已调到最优。先筛查，再给两种表示相近的学习率和正则搜索预算。
四组scene指纹应完全一致。同一表示的有/无TGV组先保持学习率一致。
历史AD评价物体取更新后值、net取当前更新前预测的差异本次未修改；不能据单步对齐
推导严格时间优劣。主比较先看充分迭代下的恢复质量，并保留原始历史数据。

旧批量汇总脚本已清理。各运行目录的 `ad_result.npz` / `net_result.npz`
保留完整配置与历史指标；直接运行 AD 时仍用 `--tgv-amp`。

新增AD测试：

```text
python -m unittest discover -s tests -p test_paper_ad_tgv.py -v
```

小型CPU测试覆盖truth/pixel/support接线、幅值先验一致性、梯度、默认关闭及
AD相位参数拒绝。没有在本地运行完整实验。

## 新增：在已有振幅 TGV 上加相位 TGV

沿用你的40%实验：obj-size624、grid4、step35、eval-size96、amp权重0.1。
同步更新后的代码到 Colab 后直接运行（输出目录另命名，不覆盖原结果）：

```python
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net --preset paper --obj-size 624 --iters 2000 --eval-size 96 --step-px 35 --grid 4 --tgv-amp 1e-1 --tgv-phase 1e-2 --outdir ov40_net_tgv_amp01_phase001
```

`--tgv-phase 0` 恢复仅振幅 TGV；`--tgv-amp 0 --tgv-phase 1e-2` 只开相位项。
建议先保持振幅0.1不变，相位试0（已有对照）、1e-3、1e-2、1e-1。
这些相位权重尚未经完整重建验证，不保证进一步提升。
其余参数（尤其probe模式/初始化、seed、ROI）保持与已有对照一致。

```python
import subprocess, sys
for phase_weight in [1e-3, 1e-2, 1e-1]:
    subprocess.run([
        sys.executable, "/content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py", "net",
        "--preset", "paper", "--obj-size", "624", "--iters", "2000",
        "--eval-size", "96", "--step-px", "35", "--grid", "4",
        "--tgv-amp", "0.1", "--tgv-phase", str(phase_weight),
        "--outdir", f"ov40_net_tgv_amp01_phase{phase_weight:g}",
    ], check=True)
```

### 相位项的定义和边界

从网络相位头(c,s)取phi=atan2(s,c)，不经过振幅或probe；零向量附近赋零相位，
避免atan2(0,0)产生无效梯度。相位使用弧度，**不除以相位均值**。
对每个相邻差分定义 D_c phi = atan2(sin(D phi), cos(D phi))，然后计算

    R_phase(phi,w) = alpha1 * mean rho(D_c phi - w)
                   + alpha0 * mean rho(E(w))
    loss = data_MSE + mean(I_measured) * [lambda_amp * R_amp + lambda_phase * R_phase]

这是 wrapped-gradient TGV2（圆周相位的TGV式扩展），不是对包装后的phi直接做普通
实数TGV，也不是全局phase unwrapping。在局部相位差小于pi且有一致展开时与实数
差分相符；相邻真实变化超过pi存在歧义，恰好pi处仍有分支不光滑性。
不采用cos/sin分别TGV，因为那会处罚本来应允许的线性相位坡度的曲率。

相位项对全局相位平移和逐像素2pi表示变化不敏感。使用与振幅相同的名义照明域，
不依赖GT和eval ROI。两个辅助向量场与Adam状态相互独立；alpha0/alpha1/eps/
inner-steps/lr目前共享配置。每个开启的项默认各做5步内迭代，因此双开有额外开销。
低振幅区域的相位物理上较弱约束，这里仍均匀正则化；不通过减小振幅逃避相位项。

相位辐条本身有真实跳变，过大权重可能抹掉细节；应同时检查相位SSIM、振幅PSNR、
数据误差和恢复图。不能因为振幅改善就认定相位也改善。

新增hist字段：tgv_phase、tgv_phase_first、tgv_phase_second、tgv_phase_weighted。
cfg.tgv_phase是权重，hist.tgv_phase是未加权正则值。原振幅hist字段不变。
相位辅助场保存为tgv_phase_aux.npz，原振幅仍为tgv_aux.npz。
collect会用phase-TGV标签区分相位权重，CSV包含相位配置及最终损失分量。

## 数学定义及实现

令 A = softplus(object_amplitude_head)，在扫描几何给出的名义照明区域
Omega 内令 u = A / mean_Omega(A)。均值参与反向传播，避免在现有
VarPro 数据损失下，通过缩小整个 object amplitude 人为减小正则。

平滑离散二阶 TGV 使用辅助向量场 v：

    R(u,v) = alpha1 * mean_Omega rho(grad(u)-v)
           + alpha0 * mean_Omega rho(E(v))
    E(v)   = (grad(v) + grad(v)^T) / 2
    rho(z) = sqrt(||z||^2 + eps^2) - eps
    loss   = amplitude_MSE + tgv_amp * mean(I_measured) * R(u,v)

最后的固定数据尺度使权重对应“相对振幅数据误差 + lambda * R”，但保留原始
MSE 的优化尺度。该 lambda 与工程里旧的 TV+二阶差分近似的权重不能直接比较。

采用非周期前向差分；对称梯度包含交叉项及 Frobenius 范数里的系数2。
两个范数都在有效差分模板上取均值。区域外不补零，不人为处罚区域边界。
Omega 是所有扫描位置上的名义圆形光斑并集，由位置和 probe_diam_px 确定；
不使用物体真值、真实探针纹理、恢复后的探针或 eval ROI。
因此这是采用已知名义光斑几何的正则区域选择。

v 从0开始，每次网络更新前先固定当前 u，使用 warm-start Adam 更新 v；
默认5步。这是有限内迭代的 TGV2 数值近似，不声称每一步都精确求出了
inf_v R(u,v)。需要检查内迭代精度时，可在固定 lambda 下对比5步和20步。

TGV 直接从幅度头计算，不直接依赖相位输出或 probe。
网络共享特征层、数据项中的 object–probe 耦合仍可能使相位/探针间接受影响。

定义参考：[Bredies, Kunisch & Pock, Total Generalized Variation (2010)](https://doi.org/10.1137/090769521)。

## Colab：先在一个 overlap 比较四个权重

先同步修改：simulations/ProPtyNet_paper.py、
functions/paperrepro/solvers_addip.py，以及 functions/paperrepro/tgv.py。
以下假定代码在 /content/ptychography_AD_DIP，Drive 已挂载。

下面沿用最近的 known-probe 实验。做 blind 时，所有组统一改成相同的
probe-mode（pixel 或 support），不要同时改变 probe 设置和正则权重。

    import subprocess
    import sys

    script = "/content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py"
    root = "/content/drive/MyDrive/ProPtyNet/tgv_amp_grid4_eval48"
    for weight in [0, 1e-4, 1e-3, 1e-2]:
        subprocess.run([
            sys.executable, script, "net",
            "--preset", "paper",
            "--obj-size", "624", "--grid", "4", "--step-px", "24",
            "--eval-size", "48", "--iters", "2000", "--seed", "0",
            "--probe-mode", "truth",
            "--tgv-amp", str(weight),
            "--outdir", f"{root}/lambda_{weight:g}/ov60_net_truth",
        ], check=True)

这些权重是初始探索范围，尚未经完整重建验证。USAF 的细条纹可能被过强正则
抹掉，需同时检查振幅图、相位图、数据误差和固定ROI的指标；TGV不保证优于
基线。常量初始化时 TGV=0 是正常的，等非平坦结构出现后才产生惩罚。
主实验应统一选择权重，不按每个 overlap 的 GT 分数各选一次最好权重。

批量脚本已清理；需要固定权重的其它 overlap 时，沿用上面的主脚本命令，
逐组显式设置 `--step-px`、`--grid`、`--obj-size`、`--eval-size` 和输出目录。
不同正则配置请分开存放，便于从各自 NPZ 的 cfg 中核查。

默认其余参数：alpha0=2，alpha1=1，eps=1e-3，inner_steps=5，lr=1e-2。
都可以使用 --tgv-alpha0、--tgv-alpha1、--tgv-eps、
--tgv-inner-steps、--tgv-lr 显式修改。

日志中 data 为原始数据项，TGV 为未加权 R，weighted TGV 为实际加入的数值。
loss 是两者之和，不能把启用与关闭TGV的总loss当成同一个拟合指标比较。
net_result.npz 的 cfg 保存设置，hist 保存各损失分量；
tgv_aux.npz 保存最终辅助向量场、正则域mask和坐标（并非训练恢复检查点）。

## 验证范围

    python -m unittest discover -s tests -p test_paper_tgv.py -v

测试覆盖跨±pi相位坡度、2pi不变性、梯度检查、零相位头、振幅单开/
相位单开/双开接线。仅小型CPU单元和16x16模拟网络接线检查；
没有运行完整物理仿真或2000步训练。
