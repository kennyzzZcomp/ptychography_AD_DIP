# 全分辨率残差细节分支

目的：在原 U-Net 的主干之外，增加浅层特征直达振幅输出的路径，检验是否改善低光子下的细条纹恢复。尚未证明有效，不将模糊预先归因于下采样。

## 实现

`--unet-detail none` 是默认值，沿用原网络。`--unet-detail residual` 启用分支：

```text
x1 = 原 encoder 第一层输出（384×384）
z  = 原 encoder/decoder/head_amp 输出
r  = Conv3x3(两个全分辨率残差块(x1))
A  = softplus(z + r)
```

每个新增残差块为 x + Conv3x3(LeakyReLU(Conv3x3(x)))。宽度沿用 base_channels；分支内部不做 BatchNorm、pooling 或上采样。共享的 e1 保留原始 BatchNorm，且每次前向只计算一次，因此不是全网络去掉 BN。

末层权重和偏置零初始化，使新增 r=0，初始输出与原 U-Net 一致；使用同一 network_seed 时主干权重也一致。第一个更新先学习分支末层，随后梯度传入内部卷积。无额外幅值缩放、阈值或真值监督，r 可正可负，最终统一 softplus 保障振幅非负。

分支输入是测量驱动的学习特征，不是物体真值边缘。没有显式频带限制，所以不能保证它只学习高频；可能也拟合噪声。主干和分支共同接受物理 Poisson 损失梯度。共享零相位物体、像素 probe 和归一化功率保持不变。

base=16 时新增参数 4*(9*b*b+b)+(9*b+1)=9425。结果 JSON 的 detail_parameter_count 和原总参数数目均记录；分支状态一并保存在 model state_dict。配置、报告和物体图标题标注开关。尚不支持完整优化器断点续训。

## Colab 指令

先将修改同步到 Colab；本次没有自动 commit/push。更新涉及 multiwave/detail.py、models.py、config.py、run_simulation.py、reconstruct.py、report.py。可通过 `python -m multiwave.run_simulation --help` 确认出现 `--unet-detail`。

```python
%cd /content/ptychography_AD_DIP

!python -m multiwave.run_simulation \
  --preset resolved --device cuda \
  --methods unet_shared_amp \
  --unet-skip concat --unet-detail residual \
  --probe-mode pixel --spectral-mode equal_power \
  --unet-activation softplus --pixel-parameterization softplus \
  --base-channels 16 --iterations 1000 --eval-every 25 \
  --loss poisson --photons-per-scan 80000 \
  --noise-seed 24 --scene-seed 17 --network-seed 31 \
  --lr-net 0.002 --lr-probe 0.01 --lr-net-decay-after 0 \
  --tgv-weight 0 --tv-weight 0
```

基线使用完全相同指令，仅将 `--unet-detail residual` 改为 `--unet-detail none`。默认新目录不覆盖旧数据。此前用户反馈的原 U-Net、80000 光子、seed24 终点物体误差=0.1519、probe=0.2382，作为历史参照；同环境重跑更适合严格对照。

固定第 1000 次比较物体误差、中心 RMSE、高频相对误差与条纹峰谷，并记录额外耗时。不能用图像锐度或训练目标单项判断成功，也不按真值挑最好迭代。第一轮不同时加 DWT/TGV，不增加宽度。

## 验证

47 项 multiwave 测试通过。新增测试核对同 seed 主干完全一致、零残差对非恒定主干输出仍保持数值一致、BN 只前向一次、Poisson 分块梯度与整批梯度一致、分支内部实际更新、状态重载和零相位/共享物体约束。测试下降只验证小规模路径可训练，不是高分辨率收益结论。

实际 resolved 384×384、base16、pixel probe、80000 光子、Poisson 配置在 CPU 上完成两步端到端运行，成功保存指标、状态和图像：`results/20260929_225110_755074`。这只是集成检查，尚未进行 1000 次效果对照。
