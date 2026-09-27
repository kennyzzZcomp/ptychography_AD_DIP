# 共享 U-Net 探针头：内部结构对照

新增 `net --probe-mode shared`。这是以当前 `net` 为底座的内部对照，
不是原论文 `run` 的复现版本；`run` 不修改，默认 `probe-mode pixel` 不修改。
原先 `ProPtyUNet(n_fields=2)` 只能提供多场的振幅/相位头，并非本对照所需的线性实虚探针头。

## 两组的实际差别

| 项目 | shared | pixel |
|---|---|---|
| 主干、物体输出 | 实数 U-Net，softplus 振幅和归一化 cos/sin 相位 | 相同 |
| 探针表示 | 最后一层共享特征 -> 3x3 线性双通道头 -> 中心裁剪 -> Pr+iPi | 直接优化 N x N 的 Pr、Pi |
| 实际初值 | O=1+0i，P=1+0i（浮点精度） | alpha=0、probe-init ones 时相同 |
| 传播、数据损失、全局幅度标定 | 原 net 的 Fresnel + amplitude MSE + VarPro | 相同 |
| lr_net | 共享主干与物体头 | 物体网络 |
| lr_probe | 仅探针输出头 | 探针像素 |
| 主干梯度 | 同时经过物体和探针两条路径 | 经过物体路径 |

共享探针没有 softplus/tanh、硬支撑或额外探针损失。先从网络输出中去掉
net_size 到 obj_size 的 padding，再中心裁到 N x N，和现有坐标约定一致。
网络前向只执行一次，共享最后一层特征，不是双 U-Net。

头的权重置零、实部偏置为 1、虚部偏置为 0。物体头也采用原 net 的中性初始化。
因此第一个反传中零权重输出头会阻断对应路径到主干的梯度，但头本身可以更新；
后续主干可接收两条路径的梯度。零头初始化带来的这个差异必须如实理解，
不能声称网络头参数的更新等价于像素更新。

同一 seed 下，共享分支在创建原物体网络之后才添加探针头：公共参数的初始值逐位一致。
但总参数量、更新的有效尺度和优化轨迹并不相同；相同学习率数值只是配对实验的起点，
不是各方法均已最优的保证。优化器为两个不重叠参数集合上的 Adam，主干只更新一次。
共享主干不能同时拥有一个独立的“物体学习率”和“探针学习率”。

## Colab 配对指令

先将本地改动同步到 Colab。每次使用新的输出目录，现有 CLI 不阻止同名结果覆盖。
两组都使用全量数据、不加 TGV/DWT，余弦学习率衰减从初始值降到零。
若已有验证过的另一套学习率，可同时替换两条命令中的值。

```python
# S：共享 U-Net 线性探针头
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net \
  --preset paper --obj-size 624 --grid 10 --step-px 12 \
  --iters 2000 --eval-size 96 --eval-every 25 --seed 0 --device cuda \
  --network-type real --base-ch 32 --skip-mode concat \
  --probe-mode shared --probe-init ones --obj-init-alpha 0 \
  --lr-net 0.001 --lr-probe 0.01 --lr-cosine \
  --tgv-amp 0 --tgv-phase 0 --outdir /content/ov80_shared_ones_cosine_seed0

# H：U-Net 物体 + 自由复数像素探针
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net \
  --preset paper --obj-size 624 --grid 10 --step-px 12 \
  --iters 2000 --eval-size 96 --eval-every 25 --seed 0 --device cuda \
  --network-type real --base-ch 32 --skip-mode concat \
  --probe-mode pixel --probe-init ones --obj-init-alpha 0 \
  --lr-net 0.001 --lr-probe 0.01 --lr-cosine \
  --tgv-amp 0 --tgv-phase 0 --outdir /content/ov80_pixel_ones_cosine_seed0
```

shared 目前要求 real/concat/full input、ones 探针初始化、obj_init_alpha=0；
不支持 progressive measurement schedule，错误组合会直接报错。
原 net 的 TGV 和分段学习率路径仍可使用，但首轮对照不启用 TGV。
`run`、`ad`、`check` 不接受 `probe-mode shared`。

## 保存与验证

- 首次训练前向打印实际初始物体/探针振幅范围、相位 RMS、与 1 的最大偏差；
  shared 会检查 O=P=1，失败则停止。没有额外初始网络前向。
- `network_metadata.json` 记录参数量、探针模式、学习率归属和实际初值；
  `net_result.npz` 的 cfg 中也保存 probe_mode。
- 输出沿用 `net_result.npz`、重建图和 `net_convergence.png`。
- 总时间包含训练循环、循环内评价/打印及一次初始化审计，不含数据生成、存盘、画图。
- 两组网络参数数不同；pixel 总参数应计入独立探针，shared 的探针头已经包含在网络参数中。
- 历史评价沿用当前 net 的更新前前向输出约定，并非新增一次更新后前向。

测试覆盖：公共权重及旧 forward 一致性、实际 ones 初值和初始传播一致性、
中心裁剪（包括奇数 padding）、两路径主干梯度、优化器参数互斥与完整覆盖、
状态字典读写、配置限制，以及两种模式的 CPU 训练/调度/保存接口。

```text
python -m unittest discover -s tests -p test_shared_probe.py
```

CPU 通路测试不构成 A100 速度或重建质量验证。该对照只能研究探针表示与参数共享的
联合改变；不能仅凭结果自动断言“共享必然导致串扰”。
