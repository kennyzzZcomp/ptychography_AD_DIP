# DWT / TGV 对照实验

本次只实现可切换的实验模块，尚未证明其优于基线。物体仍为两个波长共享的实数振幅、相位固定零；两个 probe 仍是独立复数像素参数，固定每模式功率。不新增光谱权重，不改变传播模型或探测器尺寸。

## DWT 实现

`--unet-skip dwt_concat` 在三个 encoder skip 上进行：

```text
encoder feature (B,C,H,W)
  -> Haar DWT: LL / LH / HL / HH
  -> concatenate (B,4C,H/2,W/2)
  -> residual 1x1 convolution: bands + Conv(bands)
  -> split four bands -> inverse Haar (B,C,H,W)
  -> concatenate with the existing decoder feature
```

保留全部子带，无阈值、无显式高频抑制；融合卷积零初始化，因此初始 skip 是数值意义上的恒等映射。原始 backbone 权重和物体初值与同 seed 基线一致。保留原 max pooling 和转置卷积；这是 skip 特征融合实验，不是完整 MWCNN 或 wavelet 下采样替代。

相比 `concat` 增加参数 `336*b*b + 28*b`（b 为 base_channels），base=16 时为 86,464。需同时看参数量和耗时；若有效，后续应增加容量匹配的普通卷积对照。不是此前 `functions/common/wavelet_skip.py` 的高频软阈值模块。

## TGV 实现

`--tgv-weight beta` 为共享振幅加入平滑 TGV2：

```text
loss = normalized amplitude data loss + beta * R(A,v)
R(A,v) = alpha1 * mean(sqrt(|grad A - v|^2 + eps^2) - eps)
       + alpha0 * mean(sqrt(|E(v)|_F^2 + eps^2) - eps)
```

复用 `functions/paperrepro/tgv.py:tgv2_terms`，包含对称梯度非对角项的 Frobenius 权重。不是简单 Laplacian penalty。

- 只取一次共享 A，不随波长数量重复惩罚；不正则化 phase 或 probe。
- 范围为**训练扫描 patch 的并集**，只用全部节点均在域内的有效差分模板。不使用评价 ROI、真实物体、真实 probe 或 holdout 数据。域内弱照明位置同样受到平滑先验影响。
- 差分以像素为单位，振幅不做均值归一化；更换物理采样或强度约定时应重新检查 beta。
- 默认 alpha0=2、alpha1=1、eps=0.001。每次物体更新之前，用固定当前振幅对辅助向量场 v 做 5 次 warm-start Adam 更新，辅助学习率 0.01；随后固定 v，向物体反传一次 TGV 梯度。
- 这是平滑 TGV 的**近似交替优化**，并非精确求得 inf_v TGV；AD 与 U-Net 使用同样辅助策略。辅助求解计算量包含在耗时中。
- `--tgv-weight 0` 默认完全关闭，不创建辅助优化器。第一轮不同时开启 TV/TGV；CLI 对两者同时非零报错。
- 可调参数：`--tgv-weight`、`--tgv-alpha0`、`--tgv-alpha1`、`--tgv-eps`、`--tgv-lr`、`--tgv-inner-steps`。

## Colab 指令

先同步修改到 Colab；本地修改不等于 GitHub 已 push。先确认 `--help` 含 `--unet-skip` 和 `--tgv-weight`。不要使用 `!cd` 切换下一单元的目录。

```python
%cd /content/ptychography_AD_DIP
!python -m multiwave.run_simulation --help
```

下面三条命令生成五组结果。每条命令单独运行，自动写入新的时间戳目录，不覆盖旧数据。固定 1000 次迭代、无噪声、base=16、固定网络学习率 0.002，关闭之前的学习率衰减。所有随机种子明确写出；跨 GPU/软件环境仍不保证逐位一致。

### 1. AD 和原始 U-Net 基线（两组）

```python
!python -m multiwave.run_simulation \
  --preset resolved --device cuda \
  --methods pixel_shared_amp unet_shared_amp \
  --probe-mode pixel --pixel-parameterization softplus --unet-activation softplus \
  --unet-skip concat --tgv-weight 0 --tv-weight 0 \
  --base-channels 16 --iterations 1000 --eval-every 25 --photons-per-scan 0 \
  --lr-net 0.002 --lr-pixel 0.03 --lr-probe 0.01 --lr-net-decay-after 0 \
  --scene-seed 17 --noise-seed 23 --network-seed 31
```

### 2. DWT-U-Net（首选先跑这一组）

```python
!python -m multiwave.run_simulation \
  --preset resolved --device cuda --methods unet_shared_amp \
  --probe-mode pixel --pixel-parameterization softplus --unet-activation softplus \
  --unet-skip dwt_concat --tgv-weight 0 --tv-weight 0 \
  --base-channels 16 --iterations 1000 --eval-every 25 --photons-per-scan 0 \
  --lr-net 0.002 --lr-pixel 0.03 --lr-probe 0.01 --lr-net-decay-after 0 \
  --scene-seed 17 --noise-seed 23 --network-seed 31
```

### 3. AD+TGV 和 U-Net+TGV（两组）

```python
!python -m multiwave.run_simulation \
  --preset resolved --device cuda \
  --methods pixel_shared_amp unet_shared_amp \
  --probe-mode pixel --pixel-parameterization softplus --unet-activation softplus \
  --unet-skip concat --tgv-weight 0.001 --tv-weight 0 \
  --tgv-alpha0 2 --tgv-alpha1 1 --tgv-eps 0.001 --tgv-lr 0.01 --tgv-inner-steps 5 \
  --base-channels 16 --iterations 1000 --eval-every 25 --photons-per-scan 0 \
  --lr-net 0.002 --lr-pixel 0.03 --lr-probe 0.01 --lr-net-decay-after 0 \
  --scene-seed 17 --noise-seed 23 --network-seed 31
```

`0.001` 只是待验证起点，不称为最优权重。需要扫参时，对 AD/U-Net 都试相同候选集（例如 0、0.0001、0.001、0.01），完整报告；不要只挑有利于网络的一组。若反复用 holdout 调参，holdout 就是验证集，最终还需独立测试。不能用真值挑最优迭代。

以后测试 DWT+TGV 可以组合两个开关；首轮先保持分开。新增 Poisson 噪声时，五组全部使用相同正数 photons-per-scan、相同 noise_seed，不与无噪声基线混比。

## 判断与输出

使用预定的第 1000 次结果，比较：

- `shared_amplitude_relative_error`：ROI 物体误差。
- `center_amplitude_rmse`、`high_frequency_relative_error`：细节指标。
- 固定中心条纹放大图和剖面；整体区域对比度不等于 USAF 分辨率。
- probe 误差、train/holdout 测量误差；不要把 holdout 低等同于 ROI 高频更准。
- 参数量、辅助场参数量、含评价的耗时；同迭代不是同时间预算。

`config.json` 保存全部开关。指标 JSON 保存实际 skip 类型、TGV 权重、域像素数、辅助参数量以及每次评价时的 TGV 两项和加权 penalty。图标题标注 skip/TGV。

TGV 启用时，`*_model.pth` 新增 `tgv_state_dict`，保存辅助场、域 mask 和辅助 Adam 状态。主物体/probe 优化器仍未保存，所以**不支持完整断点续训**。

## 本地验证

运行 `python -m unittest discover -s multiwave/tests -v`。

本次 39 项通过，覆盖 DWT 恒等初始化、同 seed backbone/物体初始化、可学习梯度、TGV affine 零空间与数值梯度、训练域不依赖真值/评价 ROI、分块梯度等价、AD/U-Net/DWT+TGV 联合小规模训练、零相位及探针功率约束。此处是实现验证，不是高分辨率性能结论。

另外完成 CPU smoke、4 次更新的端到端保存检查：`results/dwt_tgv_integration_20260929_190728/` 包含 AD+TGV、U-Net+TGV、DWT-U-Net 三组输出。逐数组核对两次运行的 data.npz（测量、物体、探针、扫描及划分）相同，并验证指标 JSON 与含辅助场的 checkpoint 能正常读取。尚未跑 CUDA 或 384×384 的长预算性能实验。
