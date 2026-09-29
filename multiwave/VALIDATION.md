# 首版验证记录（2026-09-29）

## 环境和检查

- 本地 Python：`D:/miniconda/python.exe`。
- PyTorch 2.12.0+cpu，NumPy 2.4.6，Matplotlib 3.10.9；CPU、2 个 PyTorch 线程。
- `python -m unittest discover -s multiwave/tests -v`：15 项检查全部通过。
- 检查覆盖：零距离恒等传播、解析平面波、共同 OPD 的相位关系、非相干强度叠加、独立相位常数不变性、OPD autograd 与中心差分、探针能量、padding 稳定性、四种表示初值一致、数据隔离、可重复噪声、三波长路径、标记评价和更新后模型保存一致性。
- 已运行并检查数值和重建图。下面全部使用最后一次更新结果，未依据真值选择最佳迭代。

## 三次运行

```powershell
python -m multiwave.run_simulation --preset smoke --iterations 40 --eval-every 10 --outdir multiwave/results/initial_smoke

python -m multiwave.run_simulation --preset standard --iterations 300 --eval-every 50 --outdir multiwave/results/standard_noisy_seed31

python -m multiwave.run_simulation --preset smoke --scene dispersive --iterations 120 --eval-every 30 --methods unet_independent unet_coupled --outdir multiwave/results/dispersion_mismatch_seed31
```

这些输出目录已存在；重跑请换新目录，程序不会覆盖已有结果。

`smoke` 四组均能从共同初值降低数据误差；扩大 FFT 窗口后的混合强度相对差为约 5.3e-5。标准场景的对应差为约 7.6e-6。这只验证当前几何的窗口稳定性。

## 带噪声标准场景，300 次更新

两波长 515/633 nm，固定权重 0.65/0.35，每位置 200000 入射光子。扫描种子 17、噪声种子 23、网络种子 31。

| 方法 | 训练观测振幅 NRMSE | 留出干净振幅 NRMSE | 平均复场相对误差 | 标记对角均值（理想 1） | 标记非对角绝对均值（理想 0） |
|---|---:|---:|---:|---:|---:|
| pixel_independent | 0.05874 | 0.12776 | 0.10718 | 0.4741 | 0.3705 |
| pixel_coupled | 0.06023 | 0.10290 | 0.07327 | 0.4782 | 0.3810 |
| unet_independent | 0.06591 | 0.01910 | 0.07252 | 0.4188 | 0.4718 |
| unet_coupled | 0.06571 | 0.01910 | 0.05326 | 0.4443 | 0.4656 |

共同 OPD U-Net 的平均复场误差在这一次运行中较低；但留出强度误差与独立 U-Net 几乎相同，而且标记矩阵远未接近单位矩阵。重建图中可见弱通道出现另一波长的吸收标记。因此不能把结果解释为“已解决光谱串扰”。

像素方法的训练误差更低、留出误差更高，提示这一协议下可能发生噪声／欠约束拟合。但未进行充分的像素正则与学习率调优，不能据此宣称 U-Net 普遍优于像素方法。

完整结果：[运行报告](results/standard_noisy_seed31/run_report.md)、[机器可读指标](results/standard_noisy_seed31/summary.json)、[对照曲线](results/standard_noisy_seed31/comparison.png)、[共同 OPD U-Net 重建](results/standard_noisy_seed31/unet_coupled_reconstruction.png)。

## 色散失配场景，120 次更新

| 方法 | 留出干净振幅 NRMSE | 平均复场相对误差 | 标记距单位矩阵 RMSE |
|---|---:|---:|---:|
| unet_independent | 0.02491 | 0.07337 | 0.52467 |
| unet_coupled | 0.02998 | 0.06364 | 0.52747 |

独立模型的留出数据预测更好，但共同 OPD 模型的整体复场指标仍较好。两个模型都未很好恢复独有吸收标记。这说明不同指标可能给出不同结论；不能以总体误差低为由宣称共同 OPD 对色散仍物理正确，也不能用这一温和失配场景确定适用边界。

完整结果：[色散运行报告](results/dispersion_mismatch_seed31/run_report.md)。

## 结论与下一步

已实现可运行的物理关联仿真、四组表示对照、留出验证和串扰诊断，状态为 `pilot-ready`。当前证据支持继续研究，但不支持“光谱已可靠分离”的结论。

下一步首先应区分：串扰来自测量可辨识性不足，还是优化／网络表示限制。可逐项考察更大波长间隔、不同探针结构、相同总预算下的额外已知照明编码，并调优同物理约束的像素正则基线。继续采用标记保留、跨通道泄漏和独立数据预测共同判断，不能仅追求混合强度拟合。

本轮未验证多随机种子统计、未见物体、真实材料色散、真实实验、盲探针或未知权重；CPU 耗时也不能作为 GPU 性能依据。
