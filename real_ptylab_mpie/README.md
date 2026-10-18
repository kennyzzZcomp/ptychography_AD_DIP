# 独立真实数据基线：PtyLab mPIE

本目录只调用官方 **PtyLab 0.3.1 mPIE**，不导入或修改 `real_ptycho`、仿真脚本，也不使用神经网络、TGV 或 NLM。原始 H5 只读，格式转换自动完成。

## Colab：安装与运行

**需要安装。电脑上的 PtyLab 不会自动出现在 Colab 中。必须使用独立虚拟环境，不要再用 `%pip install -r ...` 装进 Colab 主环境。** 实测用户安装日志表明主环境预装 CuPy/CUDA 13，直接安装本文件会混装两套 CuPy、改变 IPython 和 CUDA 依赖，导致 `libcufft.so.11` 加载失败。重启无法解决包冲突。

先选择 GPU 运行时，在已克隆本项目的 Colab 中执行这个安装单元格；任何一步失败都会停止：

```bash
%%bash
set -e
python -m venv --without-pip /content/ptylab_mpie_env
python -m pip --isolated --python /content/ptylab_mpie_env/bin/python install -r /content/ptychography_AD_DIP/real_ptylab_mpie/requirements-colab.txt
python -m pip --isolated --python /content/ptylab_mpie_env/bin/python check
```

这里使用官方 `ptylab[gpu]==0.3.1`，将 CUDA 12 的 CuPy 和 CUDA 组件装到隔离环境，不继承主环境的 site-packages，也不卸载或升级主环境的软件包。`--without-pip` 避免 Colab 缺少 ensurepip 的问题；由主环境 pip 的 `--python` 选项管理目标环境，要求 pip >=22.3。新分配的 Colab 虚拟机需要重新安装；无需手动 clone PtyLab。不要添加 `--system-site-packages`，不要把两种 CuPy 安装到同一环境。

这不会修复此前已经改动的主环境，只让 mPIE 使用另一套独立 Python 依赖。如果当前会话能执行命令，可以直接使用；如果 Colab 因 IPython 冲突反复崩溃，先把 `/content` 中要保留的文件保存到 Drive，再断开并删除运行时、重新克隆项目和挂载 Drive，然后按本节安装。

挂载 Drive（已经挂载就跳过）：

```python
from google.colab import drive
drive.mount('/content/drive')
```

数据可保持在 `/content/drive/MyDrive/ptyco_data/ptydata_33.h5`，直接运行：

```python
!/content/ptylab_mpie_env/bin/python -I -u /content/ptychography_AD_DIP/real_ptylab_mpie/run.py --data "/content/drive/MyDrive/ptyco_data/ptydata_33.h5"
```

默认就是 **225 点、中心裁剪 1024、毫米坐标、远场、GPU、50 次完整扫描**。需要改变迭代数时加 `--iters 100`。不需要手动生成另一份 H5。

默认结果保存在 `real_ptylab_mpie/results/日期时间/`，Colab 断开后可能丢失。要直接保存到 Drive，可追加一个尚不存在的目录，例如：

```python
!/content/ptylab_mpie_env/bin/python -I -u /content/ptychography_AD_DIP/real_ptylab_mpie/run.py --data "/content/drive/MyDrive/ptyco_data/ptydata_33.h5" --out "/content/drive/MyDrive/ptyco_results/mpie_1024_run01"
```

输出目录禁止覆盖；再次运行换一个目录名。若 GPU/CuPy 不可用，会明确报错，不悄悄改为 CPU。

**后续必须使用 `/content/ptylab_mpie_env/bin/python`，不要换回 `!python`。** `-I` 额外忽略用户 site-packages 与 PYTHONPATH，避免重新引入主环境的包。脚本在读取实验数据之前会执行小尺寸 GPU FFT 检查；通过后才开始处理数据。不需要在 Notebook 内核里 import PtyLab。

依据：[pip 管理独立解释器（包括未安装 pip 的 venv）](https://pip.pypa.io/en/stable/topics/python-option/)、[CuPy 安装说明与多包冲突警告](https://docs.cupy.dev/en/stable/install.html)。隔离安装流程尚未在用户 Colab GPU 上实测，安装完成不等于重建效果已验证。

## 固定设置与可比性

| 项目 | 当前入口 |
|---|---|
| 数据 | `ptychogramx` 全部 225 帧，原始帧顺序不变 |
| 坐标 | `positionx` 为 `(2,225)`；按 **mm × 0.001 → m**，减去共同中心 |
| 轴映射 | 默认第一行 y、第二行 x；可用 `--position-order xy` 交换，`--flip-y/--flip-x` 翻转；实验方向仍需核实 |
| 探测器 | 原 2000² 的 `[488:1512, 488:1512]`，得到 1024²；不是 resize 或 binning |
| 波长 / 距离 / 像元 | 632.8 nm / 6 mm / 3.76 μm |
| 前向 | CPM + `Fraunhofer`；远场是指定假设，不代表已验证真实光路 |
| 物面采样 | λz/(NΔdet)，约 0.986120 μm/pixel |
| 物体 / 探针 | 单模式、自由复数；探针数组 1024²，物体画布由 PtyLab 根据位置自动计算 |
| 初始化 | 官方 `ones` 物体和 `circ` 探针；默认圆直径为探针视场的 1/3，约 336.596 μm，只是初猜 |
| 更新参数 | betaObject=0.25，betaProbe=0.25；其余 mPIE 分母稳定项与动量使用库默认值并记录配置 |
| 正则与辅助修正 | 无 TGV/NLM；关闭 TV、平滑、纯振幅约束、硬边界、位置修正、自动对焦、背景估计等 |
| 数据约束 | 官方 `standard` 强度约束；不是 Adam 优化振幅 loss 的两阶段网络 |
| iteration | 一次完整遍历 225 张图；50 iterations = 11250 次逐帧更新，不能等同于网络 50 次更新 |

**探针的 1024² 是可优化数组尺寸，不是光斑必须占满视场。** 初始化的圆不是每轮强制支持域；没有套用之前 512² 的硬边界。若有可靠的实测光斑直径，可用 `--probe-diameter-um 数值` 指定初始圆直径。

所有帧除以同一个全局平均强度，不改变帧间相对曝光。此入口没有自行扣暗场、做饱和掩膜、重新寻找衍射中心或 binning。假定原始裁剪围绕衍射中心；这些假设不成立时，需要进一步校准，而不能只换算法。

PtyLab 0.3.1 的基础 mPIE 将坐标除以物面像元后取整；它与原网络的亚像素插值并不完全相同。因此这是独立算法诊断基线，不是保证仅改变更新公式的严格消融。探针初始化、支持域和迭代定义也需要在论文比较中说明。

## 输出与检查

- `config.json`：完整参数、实际画布、坐标单位、版本、输入摘要、入口代码摘要。
- `ptylab_input.h5`：转换后的输入，约 0.94 GB；原始 H5 不变。
- `reconstruction.npz`：未裁剪复数物体、探针、内部误差、实际使用的位置与像元尺寸。
- `reconstruction.png/.pdf`：物体与探针的振幅、相位。振幅显示上限为各图 99.5 百分位，保存的数组不截断。
- `error_curve.png/.pdf`：横轴为完整扫描 iteration。
- `summary.json`：误差、扫描次数、同步计时；计时包含 `reconstruct()` 内部准备及传输，不包含文件转换、初值建立、保存和作图。
- 出错时 `failure.json`：错误摘要；不将失败输出当成成功重建。

没有真实物体真值，不报告 PSNR。PtyLab 内部误差也不能直接与两阶段脚本的 amplitude rRMSE 比较。

可追加 `--prepare-only` 只检查数据转换，或 `--check-only` 只转换并建立 PtyLab 对象；后者默认仍要求 GPU。诊断 CPU 模式必须显式加 `--cpu`。这两种检查都不重建。

## 开发验证

2026-10-17：本机 PtyLab 0.3.1 CPU 环境下 **6 项测试全部通过**，并检查了导出图的排版。只读核实原始 H5 为 `(225,2000,2000)`、坐标为 `(2,225)`；没有运行真实数据完整重建。**Colab GPU 尚未实测。**

`test_run.py` 用 225 张 **40² 合成图**验证裁剪、毫米单位、帧配对、全局缩放和输出保护，并在 CPU 上执行 2 次 **32² 合成数据**的 mPIE 扫描，检查保存和作图链路。它不验证这份实验数据的成像质量，也不等同于 Colab GPU 验证。

```powershell
& 'C:/Users/kennyzz/Desktop/ptylab/PtyLab.py/.venv/Scripts/python.exe' -m unittest discover -s real_ptylab_mpie -v
```

官方来源：[PtyLab.py 项目与安装说明](https://github.com/PtyLab/PtyLab.py)、[ExperimentalData 接口](https://ptylab.github.io/PtyLab.py/api/experimental-data/)。
