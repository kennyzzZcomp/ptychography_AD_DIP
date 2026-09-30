# 物体振幅各向异性 TV（ATV）

这里 ATV 指 anisotropic total variation，不是 adaptive TV。主要针对横竖边缘、分段常数物体；不能据此保证半导体图像或 USAF 的精细线条一定改善。
参考：https://arxiv.org/abs/2207.04757 。该文不是当前双波长盲重建的性能验证。

R(A) = mean_valid_x |A[y,x+1]-A[y,x]| + mean_valid_y |A[y+1,x]-A[y,x]|。
两个方向分别按有效边数归一化；差分为像素单位，权重不能直接沿用采用求和形式的文献。
只在训练扫描窗口并集内、两端都有效的边上计算；不跨边界，不用真值或评价 ROI。
两个波长共享物体，只计算一次；支持 pixel_shared_amp 和 unet_shared_amp。
采用 torch.abs 的次梯度，零点次梯度取零，没有辅助内循环。保持现有 Adam 优化，不是精确 TV proximal 求解。

新增 --atv-weight，默认 0。与 --tv-weight、--tgv-weight 互斥，可以与探针粗网格和平滑共同使用。
旧 amplitude_tv 本身已是平滑绝对值的横纵分离形式，近似 ATV；新选项不是完全不同的正则化家族。新选项使用精确 L1，并将作用域明确为训练覆盖范围，保留旧选项以复现实验。

JSON 记录 atv_penalty、atv_horizontal、atv_vertical、atv_weighted_penalty；总目标包含 ATV 和探针平滑。
终端 loss[poisson] 仍然是数据项。配置和图标题记录 ATV，最终迭代固定，不按真值选迭代。

## Colab

先把本地实现同步到 Colab 的同一代码版本。本次没有自动提交或推送。
0.001 是未调优的试验起点；与 atv_weight=0 对照。过强会削弱细线/对比度，可在固定预算下比较 0.001、0.01，不按真值挑选最优配置后当独立测试结论。

```python
%cd /content/ptychography_AD_DIP

!python -m multiwave.run_simulation \
  --preset resolved --device cuda \
  --methods unet_shared_amp \
  --probe-mode pixel --probe-grid-size 48 --probe-smooth-weight 0.2 \
  --unet-skip concat --unet-detail none \
  --unet-activation softplus --pixel-parameterization softplus \
  --base-channels 16 --iterations 1000 --eval-every 25 \
  --loss poisson --photons-per-scan 20000 \
  --scene-seed 17 --noise-seed 24 --network-seed 31 \
  --lr-net 0.002 --lr-probe 0.01 --lr-net-decay-after 0 \
  --tv-weight 0 --tgv-weight 0 --atv-weight 0.001
```

## 实现检查

14 项单元/回归测试通过：ATV 已知差分值、常量零惩罚、域外不影响、梯度检查、分块 Poisson 梯度与全量目标一致、AD/U-Net 目标记账及既有 TGV/探测器测试。
一轮 CPU CLI smoke（含 Poisson、粗探针、平滑、ATV）生成结果：multiwave/results/20260930_114525_707472。
这些检查只验证实现，不构成 1000 次实验的精度提升证据。
