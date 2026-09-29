# Poisson 损失使用说明

新增 `--loss amplitude|poisson`，默认 amplitude，旧指令行为保持不变。支持像素 AD、原 U-Net、DWT-U-Net。只改变数据损失，不改变网络输入、物体/探针参数化、前向传播或生成的测量。

当前 Poisson 模式要求 `--photons-per-scan > 0`。0 表示无噪声，不存在可用于本模式的光子计数尺度，因此会给出明确错误。

## 目标与尺度

代码原本生成 y ~ Poisson(F*I)，并保存 measured=y/F。用 round(F*measured) 恢复整数计数，以消除 float32 保存误差。预测计数 mu=F*I_pred+1e-8，微小量单位为计数，不是强度。

实际使用的目标为：

```text
sum(mu - y + y*log(y/mu)) / max(sum(y_train), 1)
```

y=0 时对数项为零。这是 Poisson NLL 减去只依赖观测的饱和模型常数（half deviance），再按训练总计数归一化。其梯度与同尺度归一化 NLL 相同；不是原始含 log(y!) 的数值。所有扫描分块共享整批训练总计数，不做逐帧或逐块归一化。全零观测时分母取 1；不会除零。

原有 TV/TGV 可以组合，但改损失后正则相对尺度改变，不能把旧正则权重视为已调好。第一轮关闭两者。不自动修改学习率或选择 checkpoint。

## 输出

- config.json / metrics.json 记录 loss 类型。
- 控制台 `loss[poisson]` 是新数据目标。
- 控制台 train/heldout 和原有报告表仍为振幅 NRMSE，便于统一比较；heldout clean 只用于仿真评价。
- JSON 的 `train_data_loss` 是实际数据目标；`train_total_objective` 加上已启用的 TV/TGV 项。
- 不直接比较不同损失函数的目标数值大小；比较重建误差、细节及耗时。

## Colab：DWT-U-Net + Poisson

先同步代码。本次未自动 commit/push。运行 `--help` 应可见 `--loss`。

```python
%cd /content/ptychography_AD_DIP

!python -m multiwave.run_simulation \
  --preset resolved --device cuda --methods unet_shared_amp \
  --loss poisson --photons-per-scan 80000 \
  --unet-skip dwt_concat --probe-mode pixel \
  --unet-activation softplus --pixel-parameterization softplus \
  --base-channels 16 --iterations 1000 --eval-every 25 \
  --tgv-weight 0 --tv-weight 0 --lr-net-decay-after 0 \
  --lr-net 0.002 --lr-pixel 0.03 --lr-probe 0.01 \
  --scene-seed 17 --noise-seed 23 --network-seed 31
```

80000 是预定的中等光子试验档位，不与先前 8000 光子结果直接对比。要与之前的 8000 光子运行对照，将该参数改成 8000，其余保持一致。改为 `--loss amplitude` 即同条件旧损失对照。

原 U-Net 用 `--unet-skip concat`。同时运行 AD 与 DWT-U-Net 可用 `--methods pixel_shared_amp unet_shared_amp`，两者共享同一数据。单独运行 AD 时用 `--methods pixel_shared_amp --unet-skip concat`。

本次只实现 Poisson 损失，固定随机张量输入尚未实现，网络仍使用现有测量堆栈输入。

## 验证

43 项 multiwave 单元测试通过：覆盖与计数 NLL 的值/梯度等价、零计数有限梯度、分块与整批梯度一致、AD 和 DWT-U-Net 小规模训练以及原有回归测试。高分辨率 GPU 效果待实验，不能据此宣称优于旧损失或 AD。
