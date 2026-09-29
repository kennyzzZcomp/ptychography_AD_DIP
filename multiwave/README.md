# Multiwave：零相位 USAF、双波长共享物体

当前主线是你的样品假设：**两波长共享同一张实数振幅图 A，物体相位严格固定为零**。各波长使用自己的已知探针和传播算子，预测强度非相干相加。网络没有可学习的相位头、OPD 头或独立波长物体头。

旧“共同 OPD＋独立吸收”方向已冻结备份，详见 [备份说明](backups/README.md)。

## Colab 运行

先把更新后的 multiwave 文件夹同步到项目中，并确认项目根目录有 USAF.jpg。

```python
# 默认两组：共享振幅像素优化 + 共享振幅 U-Net
!python /content/ptychography_AD_DIP/multiwave/run_simulation.py --preset standard --device cuda

# 只运行你的 U-Net 方案
!python /content/ptychography_AD_DIP/multiwave/run_simulation.py --preset standard --device cuda --methods unet_shared_amp
```

不要再用 unet_coupled 来运行当前方案：它是旧 OPD 方向的方法名。

本地在工程根目录可运行：

```powershell
python -m multiwave.run_simulation --preset smoke
python -m unittest discover -s multiwave/tests -v
```

交互入口：[multiwave_demo.ipynb](multiwave_demo.ipynb)。

## 默认仿真设置

- 真值：项目根目录 USAF.jpg，灰度直接作为振幅，黑色接近不透光、白色接近全透光。保持纵横比，用 BOX 缩小后居中放入透光背景；最长边占物体尺寸 55%。边缘灰度是离散化近似，不是相位。
- 所有波长真值完全相同，虚部和 OPD 均为零。相位为零只约束物体，不限制复探针或传播场的相位。
- 默认波长 515/633 nm，固定有效光子比例 0.65/0.35；可配置为 515/1030 nm。
- 物面与探测器像素均为 4 um，传播距离 1.5 mm，角谱零填充倍率 2。
- standard：物体 96×96、探测器 48×48、500 次更新、每扫描 20 万入射光子；smoke：64×64／32×32、80 次更新、无噪声。
- 25 个扫描位置，20 个用于优化、5 个留出。留出测量不进入网络输入。
- 探针、权重和位置已知；本轮没有新增盲探针估计。

这些是合成实验参数，不是对实际 USAF 装置的标定。96×96 的目标已经下采样，不能据此报告原 USAF 图上细小线组的真实分辨率。

## 对照与输出

默认只比较 pixel_shared_amp 和 unet_shared_amp，两者都只有一个共同振幅未知量，初值相同。可选 --tv-weight 在两组中都作用于共同振幅。

重建图现在显示：共同真值、共同重建、振幅误差、中心行剖面；不再把同一物体重复画成两个光谱通道。指标包括振幅 RMSE、相对误差、PSNR、亮暗区域振幅对比度及留出强度误差。区域对比度不是线组分辨率判据。

输出保存在 multiwave/results/时间戳/；也可指定 --outdir，程序拒绝覆盖非空目录。包含配置、USAF 文件 SHA-256、数据、最终重建、模型权重、指标和图片。历史 results 中的 OPD 实验仍保留，不能与新模型混用。

## 更改波长或作单波长对照

```powershell
python -m multiwave.run_simulation --preset standard --wavelengths-nm 515 1030 --weights 0.2 0.8
python -m multiwave.run_simulation --preset standard --wavelengths-nm 515 --weights 1
python -m multiwave.run_simulation --preset standard --wavelengths-nm 633 --weights 1
```

这三类配置保持每扫描总入射光子数不变；不同波长的传播可能造成不同的探测光子数。单／双波长收益需要实际对照，不能预设双波长更好。自定义图片使用 --usaf-path；调整占比使用 --usaf-fill。

技术细节见 [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md)，本地验证见 [VALIDATION.md](VALIDATION.md)。

