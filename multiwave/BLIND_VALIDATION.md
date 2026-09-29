# 独立像素探针首轮验证

默认直接相加各波长强度，没有独立光谱权重变量；探针每模式功率为 1/L。

25 项检查通过。新增验证覆盖零相位非真值初值、实/虚梯度、每步归一化、真值缓存隔离、训练后更新及探针状态恢复。

两次无噪声 smoke 配置均运行 200 步；探针模式分别为 pixel 和 known。相同真值、扫描、光子约定和名义 ROI。

| 探针模式 | 物体表示 | 最终物体相对误差 | 留出振幅 NRMSE | 初始平均探针误差 | 最终平均探针误差 |
|---|---|---:|---:|---:|---:|
| pixel | pixel_shared_amp | 0.16196 | 0.10152 | 0.56564 | 0.44813 |
| pixel | unet_shared_amp | 0.18517 | 0.09180 | 0.56564 | 0.41442 |
| known | pixel_shared_amp | 0.08308 | 0.03111 | 0.00000 | 0.00000 |
| known | unet_shared_amp | 0.16545 | 0.06266 | 0.00000 | 0.00000 |

盲探针组的数据误差和探针误差下降，但最终探针误差仍较大；这仅证明联合优化路径运行和初步改善，未证明两个探针已正确分离，也未证明 U-Net 优于像素方法。后续应研究收敛、模式歧义和约束；不要只根据混合强度误差评判。

本轮无圆形硬支撑或真值波前初值。功率归一化仍假设每波长有效功率已知且相等，不能称为完全无标定。

[盲探针运行报告](results/blind_equal_power_noiseless/run_report.md) · [已知探针对照](results/known_equal_power_noiseless/run_report.md)

命令：

```powershell
python -m multiwave.run_simulation --preset smoke --iterations 200 --probe-mode pixel
python -m multiwave.run_simulation --preset smoke --iterations 200 --probe-mode known
```
