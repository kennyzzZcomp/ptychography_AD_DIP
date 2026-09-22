# Colab：先定位低重叠训练的真实瓶颈

`net` 新增 `--timing-warmup`，默认 -1 完全关闭计时同步，不改变损失。
传10表示跳过前10个训练iteration，再记录每个阶段的同步墙钟时间。
这是诊断工具，不是性能优化，也不是精确的kernel profiler。

沿用已取得改善的40%设置，手动在Colab运行30步短诊断：

```python
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net --preset paper --obj-size 624 --iters 30 --eval-size 96 --step-px 35 --grid 4 --tgv-amp 0.1 --tgv-phase 0 --timing-warmup 10 --outdir ov40_timing_amp
```

这是短时间性能诊断，30步结果不能评价最终恢复能力。不覆盖2000步实验目录。
将相位权重改成0.01，另设outdir，可估计相位TGV的额外成本。
如设置cosine调度，30步会改变调度进程，不应用来比较2000步收敛。

输出 `net_timing.json` 包含逐步耗时、分项均值/中位数、GPU型号、PyTorch版本、
预热后峰值显存。实际scene与全部超参数在同目录 `net_result.npz` 的cfg中。
读取示例（Colab工作目录与运行时一致）：

```python
import json
from pathlib import Path
p = json.loads(Path("ov40_timing_amp/net_timing.json").read_text())
print("device:", p.get("gpu_name", p["device"]))
print("measured iterations:", p["measured_iterations"])
for name, value in p["stages"].items():
    print(f"{name:26s} mean={value['mean_s']*1000:8.3f} ms  calls={value['calls']}")
```

解释：

- network_decode：全图U-Net、复场与probe解码。
- physics_and_data_loss：传播、幅度与全局标度拟合、MSE，不含反向。
- amplitude_tgv / phase_tgv：各自辅助场内迭代，包括内层反向和Adam。
- outer_backward：整个外层loss反向，网络/物理/TGV梯度没有进一步拆开。
- optimizer：外层网络与probe优化器。
- evaluation_and_logging：评价、CPU转换和日志，仅在评价步出现。
- scheduler / zero_grad：其余调度与清梯度工作。

每个mark同步CUDA，测量含CPU调度和同步开销，禁用了跨阶段异步重叠。因此这些数值
用于找候选瓶颈；不要据此直接承诺“删去一半FFT就快一倍”。网络前向成本也不能代表
全部网络成本，因为其反向计入outer_backward。未测到的阶段不会人为记成零。
如果warmup >= iters，报告0个有效步骤和null平均值，应增加iters而非解读为零耗时。
不包含scene生成、网络初始化、最终文件保存及绘图。

论文速度比较应使用关闭该开关的完整运行，保持同一GPU、精度、评价频率和停止条件，
重复多次并报告达到同一质量阈值的时间及失败率。评分/选图开销必须包括在方法时间中。
仅减少U-Net输入通道，或仅减少loss中的扫描点，都不等同于减少实际采集量。

本地验证限于计时单元测试和16×16模拟网络接线；没有执行上述物理仿真命令。
