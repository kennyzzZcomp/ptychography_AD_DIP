# Object-amplitude TGV2 for ProPtyNet_paper net

这项扩展对应命令里的 net（DIP 对照分支）。只对物体振幅加正则。
默认 --tgv-amp 0，沿用原来的损失和优化器更新。

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

先同步本次修改：simulations/ProPtyNet_paper.py、simulations/paper_overlap_colab.py、
functions/paperrepro/solvers_addip.py，以及新文件 functions/paperrepro/tgv.py。
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

汇总：

    !python /content/ptychography_AD_DIP/simulations/paper_overlap_colab.py collect \
      --root /content/drive/MyDrive/ProPtyNet/tgv_amp_grid4_eval48

不同 lambda 的 DIP 在曲线和表格中有不同标签。
不应在同一汇总中混合相同lambda但不同alpha/eps/内迭代设置的重复实验；
这些调参实验请另建根目录。summary_long.csv保存完整TGV配置及最终损失分量。

## 固定权重跑五个 overlap

    !python /content/ptychography_AD_DIP/simulations/paper_overlap_colab.py run \
      --root /content/drive/MyDrive/ProPtyNet/tgv_amp_1e-3_grid4_eval48 \
      --only dip_known --overlaps 80,70,60,50,40 \
      --obj-size 624 --fixed-grid 4 --eval-size 48 \
      --iters 2000 --seeds 0 --device cuda --tgv-amp 1e-3

默认其余参数：alpha0=2，alpha1=1，eps=1e-3，inner_steps=5，lr=1e-2。
都可以使用 --tgv-alpha0、--tgv-alpha1、--tgv-eps、
--tgv-inner-steps、--tgv-lr 显式修改。

日志中 data 为原始数据项，TGV 为未加权 R，weighted TGV 为实际加入的数值。
loss 是两者之和，不能把启用与关闭TGV的总loss当成同一个拟合指标比较。
net_result.npz 的 cfg 保存设置，hist 保存各损失分量；
tgv_aux.npz 保存最终辅助向量场、正则域mask和坐标（并非训练恢复检查点）。

## 验证范围

    python -m unittest discover -s tests -p test_paper_tgv.py -v

仅小型CPU单元和16x16模拟网络接线检查；没有运行完整物理仿真或2000步训练。
