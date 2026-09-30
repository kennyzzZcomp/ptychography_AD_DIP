# 低维振幅/相位探针试验

使用 --probe-mode basis --probe-amp-order 6 --probe-phase-order 6。
不要同时使用非零 --probe-grid-size。pixel/known 的默认行为保留。

振幅 = softplus(原平滑初值的 inverse-softplus + 振幅余弦展开)，然后按每波长固定功率归一化。
相位 = 独立的低频二维可分离余弦展开，排除全局常数相位。
每轴频率为 0,...,order-1，默认振幅36项、相位35项，每波长71个参数，两个波长共142个参数。
余弦基的非零频率按 RMS 归一化。不使用真实探针、真实波前、物体真值或评价 ROI。
这不是 Zernike 模型，不是精确物理像差模型，也没有添加 support；是可调阶数的低频表示先验。
原48网格有9216个探针参数。basis 起始场与完整像素探针初值一致，与48网格插值后的初值略有不同。

第一轮保持 probe-smooth-weight=0.2、lr-probe=0.01 及其余参数不变，关闭物体 ATV/TV/TGV，单独比较探针表示：

```python
%cd /content/ptychography_AD_DIP
!python -m multiwave.run_simulation \
  --preset resolved --device cuda --methods unet_shared_amp \
  --probe-mode basis --probe-amp-order 6 --probe-phase-order 6 \
  --probe-smooth-weight 0.2 \
  --unet-skip concat --unet-detail none \
  --unet-activation softplus --pixel-parameterization softplus \
  --base-channels 16 --iterations 1000 --eval-every 25 \
  --loss poisson --photons-per-scan 20000 \
  --scene-seed 17 --noise-seed 24 --network-seed 31 \
  --lr-net 0.002 --lr-probe 0.01 --lr-net-decay-after 0 \
  --tv-weight 0 --tgv-weight 0 --atv-weight 0
```

相同 lr 数值不代表相同物理场更新幅度，基系数与像素的参数化不同。
低频相位可能表达不了真实复杂波前；平滑好看不等于正确。比较最终物体、探针、干净训练和留出误差，不按真值挑最佳迭代。
支持原 U-Net 和直接像素 AD，不支持 feedback 试验分支。保存系数、基函数和初始化 buffer，可由配置重建后载入 state_dict。
实现检查：10项测试通过，覆盖初始化、功率、梯度、波长独立性、checkpoint恢复、两种方法短程训练及粗网格/ATV回归；CPU CLI 短程可生成报告。这不是1000次精度验证。
本地修改尚未自动提交/push；Colab需同步代码。
