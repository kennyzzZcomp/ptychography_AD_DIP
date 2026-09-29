# 粗网格复数探针

`--probe-mode pixel --probe-grid-size 48`：两个波长各自优化 48×48 实部和虚部，使用双线性插值（align_corners=False）形成 192×192 的物理探针，然后分别做固定功率归一化。原 U-Net 仍直接输出物体，不是 feedback 方法。传播网格、探测器和测量数据分辨率不变。

默认 probe-grid-size=0 使用完整 patch_size 网格，保持旧行为；也可显式设为 192。粗网格初始化仅由默认平滑零相位初值做 area 下采样，不使用真值。插值后初值与原完整网格存在小差异，报告实际 initial_probes，不能宣称二者像素级初值完全相同。两个波长的训练参数相互独立，不假设共享 OPD 或振幅。

两个 192×192 复数 probe 共 147456 个实参数，48×48 时为 9216，降低到 1/16。网络仍可以学到网格尺度变化，这不是理想带限滤波器；双线性网格可能限制真实探针的细结构。没有增加额外拟合步数，也没有自动减小探针学习率。相同学习率在不同参数化中不代表相同物理更新幅度。

## Colab

先同步本地代码到 Colab。本次未 commit/push。涉及 probes.py、config.py、run_simulation.py、reconstruct.py、report.py。

```python
%cd /content/ptychography_AD_DIP
!python -m multiwave.run_simulation \
  --preset resolved --device cuda \
  --methods unet_shared_amp \
  --probe-mode pixel --probe-grid-size 48 \
  --probe-smooth-weight 0 \
  --unet-skip concat --unet-detail none \
  --unet-activation softplus --pixel-parameterization softplus \
  --base-channels 16 --iterations 1000 --eval-every 25 \
  --loss poisson --photons-per-scan 20000 \
  --scene-seed 17 --noise-seed 24 --network-seed 31 \
  --lr-net 0.002 --lr-probe 0.01 --lr-net-decay-after 0 \
  --tv-weight 0 --tgv-weight 0
```

完整像素对照只改 `--probe-grid-size 0`；AD 也支持此参数，方法改为 pixel_shared_amp。先关闭平滑/TGV/DWT，单独评价参数化改变。结果 JSON 保存 probe_grid_size、probe_parameter_count、probe_interpolation、probe_state_dict；checkpoint 中配置记录粗网格大小，加载时必须按该大小构建 PixelProbes。

固定最终迭代比较物体误差、细节图、高频误差、probe 振幅/相位误差及留出测量误差；图像变平滑不是成功的充分条件。若有明显网格或欠拟合现象，96×96 是更弱的约束，但当前尚未验证哪种网格最合适。

57 项 multiwave 测试通过：旧默认输出不变、48×48 参数计数/输出形状、功率归一化、两个波长参数独立、复数传播分块梯度与整批梯度一致、粗网格实际参与训练、checkpoint 重载及配置校验。测试不证明完整实验性能。

实际 resolved 384×384、probe_grid_size=48、20000 光子、seed24 两步 CPU 运行完成，成功输出报告及图片：`results/20260930_014209_991224`。未运行 1000 次效果实验。
