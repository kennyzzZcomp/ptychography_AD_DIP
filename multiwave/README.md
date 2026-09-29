# 零相位 USAF：共享振幅 + 独立复数像素探针

当前默认执行盲探针联合优化：U-Net（或像素对照）生成一张共同物体振幅，物体相位固定为零；两个探针各自优化实部和虚部，从平滑圆斑、零相位开始。

## Colab

同步更新后的 multiwave 文件夹，并保留项目根目录 USAF.jpg：

```python
# 首轮无噪声：只运行 U-Net + 双像素探针
!python /content/ptychography_AD_DIP/multiwave/run_simulation.py --preset standard --device cuda --photons-per-scan 0 --methods unet_shared_amp

# 无噪声像素物体 / U-Net 物体对照，二者都优化探针
!python /content/ptychography_AD_DIP/multiwave/run_simulation.py --preset standard --device cuda --photons-per-scan 0

# 相同测量设置的已知探针对照
!python /content/ptychography_AD_DIP/multiwave/run_simulation.py --preset standard --device cuda --photons-per-scan 0 --probe-mode known
```

默认 --probe-mode pixel 是盲探针；--probe-mode known 才使用真值探针。
standard 默认有 Poisson 噪声，首轮显式 --photons-per-scan 0。smoke 默认无噪声、较小尺寸。

## 不引入光谱权重参数

默认 equal_power 模式直接计算：

```text
O_1 = O_2 = A + 0i
I_j = |D_1(P_1 * S_j(A))|^2 + |D_2(P_2 * S_j(A))|^2
```

没有单独的光谱权重变量或权重更新。首轮假设两波长有效入射功率相等，探针幅值包含功率：L 个波长时各探针 sum(|P_l|^2)=1/L，总功率为 1。该能量约束用于固定标度，仍是一项已标定功率假设，并非完全无标定。

以前的 0.65/0.35 权重不用于当前默认仿真。旧加权模式只为历史兼容保留，需要显式 --spectral-mode weighted --weights ...；当前无需使用。

## 物体、探针和优化

- USAF.jpg 按灰度振幅载入，保留长宽比，BOX 缩小后放入透光背景；各波长完全相同且零相位。
- U-Net 只保留一个 sigmoid 振幅输出头；没有相位头。
- 两个探针是独立实部／虚部参数；初值仅由几何尺寸生成，不读取真值振幅或波前；无圆形硬支撑约束，估计范围为整个探针窗口。
- 物体与探针同时更新，Adam 参数组分别使用 --lr-net（像素物体用 --lr-pixel）和 --lr-probe；探针默认学习率 0.01。
- 探针每次前向按固定功率归一化。真实探针只用于合成测量、数值审计和事后评价。
- 默认 ROI 根据初始化探针的名义照明覆盖生成，已知／盲探针对照共用；不利用真实探针形状选择训练正则区域。

## 几何和运行设置

默认 515/633 nm；像素 4 um；传播距离 1.5 mm；同网格、带限零填充角谱传播。standard：96×96 物体、48×48 探测器、500 次更新、每扫描总入射光子数 200000。smoke：64×64／32×32、80 次更新、无噪声。

25 个扫描位置，20 个训练、5 个留出；留出图不进入 U-Net 输入。初始化与波长数量独立，单波长也支持：

```powershell
python -m multiwave.run_simulation --wavelengths-nm 515 1030
python -m multiwave.run_simulation --wavelengths-nm 515
python -m multiwave.run_simulation --preset smoke
python -m unittest discover -s multiwave/tests -v
```

## 输出

默认新建 results/时间戳/，拒绝覆盖非空目录。

- *_reconstruction.png：共同振幅真值、重建、误差与剖面。
- *_probes.png：各波长探针的真值／初始／重建振幅，以及真值／去全局相位后的重建相位。
- *_fields.npz：最终物体、initial_probes、probes、预测强度等。
- *_metrics.json：物体和各探针误差、功率、收敛历史。
- *_model.pth：物体 state_dict 与 probe_state_dict；不含优化器状态。
- config.json、data.npz、summary.json、comparison.png、run_report.md。

探针评价只去每通道常数相位，不额外拟合幅值。物体零相位是硬约束，不是相位恢复结果。当前还没有证明双探针能够唯一且稳定分开；数据误差低不能代替探针评价。

[技术说明](TECHNICAL_NOTES.md) · [盲探针验证](BLIND_VALIDATION.md) · [交互 Notebook](multiwave_demo.ipynb) · [旧 OPD 方向备份](backups/README.md)。

