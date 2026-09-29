# 独立复数探针平滑约束

开关 `--probe-smooth-weight`，默认 0；只允许 `--probe-mode pixel`，原始 U-Net 仍直接输出共享物体振幅。本版本不耦合两个波长、不指定真实探针形状、不使用真实相位或评价 ROI，不改为 feedback 方法。

定义 R=mean_lambda[(sum |Dx P_lambda|² + sum |Dy P_lambda|²) / sum |P_lambda|²]，总目标为原数据目标 + probe_smooth_weight * R（若启用其他正则也分别相加）。Dx/Dy 是像素单位相邻差分，不跨边界，不补零；每个波长独立归一化后取平均。复数差分同时约束实部和虚部，等价于约束振幅变化及有振幅区域的相位变化，不直接差分包裹相位。对各波长整体相位和非零整体幅值不变。分辨率改变后系数不可直接按同物理强度解释。

这只是平滑先验：强权重会抹去真实探针结构，平滑外观不等于正确。0.01 是尚未经过效果调优的起始值，不宣称普遍最优。功率依旧每波长固定为 1/L。平滑损失在训练每轮只计入一次，不随扫描 chunk 数重复；报告记录原始 penalty、加权 penalty 和 total objective，终端 loss[poisson] 仍只显示数据目标。

## Colab 指令

先同步代码。本次未自动 commit/push；涉及 config.py、probes.py、reconstruct.py、run_simulation.py、report.py。

```python
%cd /content/ptychography_AD_DIP
!python -m multiwave.run_simulation \
  --preset resolved --device cuda \
  --methods unet_shared_amp \
  --probe-mode pixel --probe-smooth-weight 0.01 \
  --unet-skip concat --unet-detail none \
  --unet-activation softplus --pixel-parameterization softplus \
  --base-channels 16 --iterations 1000 --eval-every 25 \
  --loss poisson --photons-per-scan 20000 \
  --scene-seed 17 --noise-seed 24 --network-seed 31 \
  --lr-net 0.002 --lr-probe 0.01 --lr-net-decay-after 0 \
  --tv-weight 0 --tgv-weight 0
```

对照只把 probe-smooth-weight 改为 0；AD 同样支持此约束，改为 --methods pixel_shared_amp。先不混合物体 TGV/DWT 等改动。若抹掉真实探针结构，可降低到 0.001；不能仅凭数据损失或图像变平滑判断有效，应同时比较物体细节、物体误差及 probe 振幅/相位误差。固定最终迭代，不按真值挑最佳迭代。

## 功能验证

54 项 multiwave 测试通过。新增检查包括复数有限差分梯度、整体相位/幅值不变性、单独下降平滑项有效、全量与分块数据加正则梯度一致、记录的总目标一致、探针功率保持及错误配置拒绝。没有进行 1000 次效果对比，不能声称重建精度已提高。
