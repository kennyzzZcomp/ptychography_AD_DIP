# 零相位共享 USAF 验证记录

本记录对应当前主线；原共同 OPD／独立吸收验证已保存在 backups/opd_spectral_20260929_150116.zip。

## 验证范围

- 本地 CPU，PyTorch 2.12.0+cpu，2 个线程。
- 20 项测试通过，包含原传播回归与新增共享振幅检查。
- 新检查包括：USAF 真值跨波长一致且零虚部；U-Net 无相位／OPD 参数；像素／网络初值一致；梯度有效；训练与保存恢复后仍严格共享、零相位；单波长使用相同真值、扫描和探针归一化；共同振幅评价及 TV。

## 运行

```powershell
python -m multiwave.run_simulation --preset standard --iterations 300 --eval-every 100 --outdir multiwave/results/usaf_shared_amplitude_300
```

该目录已存在，重跑请换新目录。真值来源为项目 USAF.jpg；默认 515/633 nm、0.65/0.35 固定权重，每扫描 20 万入射光子、已知探针。

| 方法 | 振幅相对误差 | 振幅 PSNR dB | 留出干净数据振幅 NRMSE | 亮暗区域振幅对比度 |
|---|---:|---:|---:|---:|
| pixel_shared_amp | 0.04174 | 29.179 | 0.03704 | 0.8107 |
| unet_shared_amp | 0.04813 | 27.941 | 0.02830 | 0.9133 |

两组最终物体虚部最大值均为 0，波长间物体差异最大值均为 0。这里的零相位是硬约束，不是待恢复相位的准确性指标。

像素法振幅误差较低，U-Net 留出预测误差较低、亮暗区域对比度较高；不能据此宣称某方法全面更优。对比度是区域均值指标，不是条纹分辨能力。图像只展示训练照明确定的 ROI，未照明区域不参与评价。

padding 从 2 倍扩大到 4 倍时预测强度相对差约 0.00137；这验证当前窗口稳定性，不能替代物面采样加密。USAF 原图的细小线条经过缩小后可能已经消失；本次不报告原始图案的 group/element 或实际 lp/mm。

结果：[运行报告](results/usaf_shared_amplitude_300/run_report.md)、[比较图](results/usaf_shared_amplitude_300/comparison.png)、[U-Net 重建](results/usaf_shared_amplitude_300/unet_shared_amp_reconstruction.png)。

当前只做一个物体与一组随机种子的实现验证，未完成参数调优、同时间预算对比、单／双波长性能结论或盲探针验证。

