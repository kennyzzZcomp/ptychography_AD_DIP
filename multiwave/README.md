# Multiwave：物理关联的多波长叠层成像仿真

使用现有 INNM U-Net 主干，研究共同 OPD、波长特异吸收与固定光谱权重下的混合强度反演。首版采用**已知探针、已知扫描位置、统一物理网格上的角谱传播**，提供像素／U-Net 与独立／共同 OPD 的四组对照。

完整技术细节、物理假设、文献依据和局限见 [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md)。本版不含盲探针或未知权重估计，创新性尚未确立。

## 快速运行

在 INNM_Code 根目录执行；依赖 Python、PyTorch、NumPy、Matplotlib：

```powershell
# 小尺寸、无噪声、80 次更新、四组对照
python -m multiwave.run_simulation --preset smoke

# 96×96 物体、Poisson 噪声、500 次更新
python -m multiwave.run_simulation --preset standard

# 只运行共同 OPD U-Net；auto 自动选择可用设备
python -m multiwave.run_simulation --preset standard --methods unet_coupled --device auto

# 已安装 CUDA 版 PyTorch 的环境
python -m multiwave.run_simulation --preset standard --device cuda
```

也支持 `python multiwave/run_simulation.py`；交互入口为 [multiwave_demo.ipynb](multiwave_demo.ipynb)。

## 默认配置

| 参数 | smoke | standard |
|---|---:|---:|
| 波长 | 515、633 nm | 515、633 nm |
| 固定光谱权重 | 0.65、0.35 | 0.65、0.35 |
| 物面／探测器像素 | 4 um | 4 um |
| 样品到探测器距离 | 1.5 mm | 1.5 mm |
| 物体／探测器边长 | 64／32 | 96／48 |
| 扫描 | 5×5、步长 6 px、抖动 ±1 px | 5×5、步长 8 px、抖动 ±1 px |
| 训练／留出扫描数 | 20／5 | 20／5 |
| 每扫描入射光子数 | 0（无噪声） | 200000 |
| 迭代数／U-Net 基础宽度 | 80／4 | 500／8 |

这些是合成验证参数，不代表已标定你的 MATLAB 实验装置。

## 可选实验

```powershell
# 接近原 MATLAB 波长设置，保持固定权重和一致网格
python -m multiwave.run_simulation --wavelengths-nm 515 1030 --weights 0.2 0.8

# 三波长：权重个数匹配，正值且总和为 1
python -m multiwave.run_simulation --wavelengths-nm 450 532 633 --weights 0.2 0.5 0.3

# 色散真值，检查共同 OPD 假设失配
python -m multiwave.run_simulation --scene dispersive --methods unet_independent unet_coupled

# 同一个复物体的消融控制
python -m multiwave.run_simulation --methods pixel_common unet_common unet_coupled

# 弱通道、噪声和另一网络初始化
python -m multiwave.run_simulation --weights 0.9 0.1 --photons-per-scan 50000 --network-seed 32

# 两种表示使用同样的共同 OPD 与 TV
python -m multiwave.run_simulation --methods pixel_coupled unet_coupled --tv-weight 0.001
```

场景支持 `shared_complex`、`shared_opd`、`spectral_absorption`（默认）、`dispersive`。更多设置见 `python -m multiwave.run_simulation --help`。改变扫描种子不改变解析物体本身。

## 输出与评价

默认写入 `multiwave/results/<时间戳>/`；可用 `--outdir` 指定新目录，拒绝覆盖非空目录。

- `config.json`：配置、软件环境、随机种子和传播窗口检查。
- `data.npz`：真值、探针、坐标、测量、训练／留出索引及 ROI。
- `*_fields.npz`：最终更新后的复物体、吸收、有效 OPD、预测强度。
- `*_model.pth`：模型权重，不含优化器状态，不能直接续跑 Adam 进度。
- `*_metrics.json`、`channel_metrics.csv`、`summary.json`：指标和收敛历史。
- `scene.png`、`*_reconstruction.png`、`comparison.png`：场景、重建和收敛图。
- `run_report.md`：本次运行摘要和局限。

评价只移除全局相位／OPD 常数，不重缩放振幅。留出图不进入网络输入或损失。不凭混合强度误差判断分离效果：同时检查复场误差、标记传递矩阵的对角保留和非对角泄漏。

## 文件入口

| 文件 | 用途 |
|---|---|
| config.py | 配置、单位与校验 |
| physics.py | 带限零填充角谱传播、强度混合 |
| scene.py | 四类样品、已知探针、扫描和 Poisson 噪声 |
| models.py | 原 U-Net 主干的新输出头及像素模型 |
| reconstruct.py | 训练、更新后评价、串扰诊断 |
| report.py | 保存指标与绘图 |
| run_simulation.py | CLI／Python 入口 |
| tests/test_multiwave.py | 物理模型、梯度、数据划分和训练检查 |

```powershell
python -m unittest discover -s multiwave/tests -v
```

## 参考与后续

外部参考：`C:/Users/kennyzz/Desktop/pim/double_yunnan.m`、`C:/Users/kennyzz/Desktop/pim/multiwave_double_wave.m`。未复制其动态权重或未经核对的采样设置；原有单波长文件未修改。

后续重点为多样品、多初始化、弱通道、同时间预算调参、独立传播／采样验证，以及材料色散和盲探针扩展。当前四组对照不含传统 PIM/ePIE 基线。已完成的本地验证见 [VALIDATION.md](VALIDATION.md)。

