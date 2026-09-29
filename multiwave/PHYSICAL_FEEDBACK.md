# 物理梯度反馈原型

> 2026-09-30 用户 Colab 中间结果：20000 光子、seed24，运行到 600/1000。迭代 200/400/600 的 Poisson 目标为 0.85813/0.84391/0.83809，物体误差为 0.2778/0.3179/0.3528，probe 误差为 0.4019/0.4243/0.4926，heldout clean 为 0.2287/0.2498/0.2624。训练拟合改善而重建退化，当前没有效果优势，不建议按原配置继续投入。此为截图中间读数，不是最终结果或真值挑选 checkpoint。
>
> 机制局限：物体成为自由像素状态，网络只调节步幅，不再具备原 U-Net 直接表示物体的相同结构限制。RMS 梯度归一化在未触及数值下限时消除了梯度整体变小带来的步幅衰减，且 m>=0.5 无法关闭可疑更新。单步训练目标只奖励测量拟合，未提供区分真实结构与噪声的独立依据。这些是设计风险，不是已通过消融确定的唯一原因。未额外启动训练。

方法 `feedback_shared_amp`，不改变原 `pixel_shared_amp` / `unet_shared_amp` 默认行为。目标是检验测量梯度驱动的空间更新权重是否有用，不保证优于 AD 或原 U-Net。不是预训练模型，不是分波长梯度方法。

## 精确定义

状态为共享非负实振幅 A，初值 exp(-0.1)，不训练物体相位。U-Net 使用原 ProPtyUNet 主干，3 通道输入，删除相位头，输出一张更新权重图。无 DWT、TGV、额外残差分支。BN 使用当前 batch 统计，不累加运行统计；评估最终物体直接读取状态，不再前向网络。

每次外层更新：

1. 固定当前估计探针，从全部训练扫描的 Poisson 目标计算共享振幅的总梯度 g。分块累加使用全训练计数归一化，梯度贡献对波长求和。已知探针对照才使用真值探针。
2. 用当前探针和训练位置计算覆盖 C；S={C>0}，s=max(RMS(g[S]),1e-12)，d=g/s。这里实际物体步长采用全局 RMS 归一化的梯度，不是原始梯度单位。S 不是评价 ROI。
3. 输入 concat(A,d,C/max(C))，全部 detach。网络输出 raw，m=1+0.5*tanh(raw)，权重在 [0.5,1.5]，末层零初始化保证初始 m=1。
4. 候选物体 A'=clamp_min(A-feedback_step*m*d,0)。不限制上界为 1，也不使用真值边缘。
5. 对候选物体预测训练测量，计算 Poisson 目标；一次 Adam 更新网络参数及像素探针。不反传到生成 d 的计算过程或历史状态，因此无二阶导数，属于在线单步截断优化，不是完整端到端多步展开。
6. 用更新后的网络、相同的已固定输入和 d 重新计算候选，提交为新物体状态。探针同时更新；下一轮才用新探针重新计算 d。提交后的物体/探针接受统一评估，报告最终轮，不挑最优轮。

有界权重不意味着有限步长必定下降。反馈输入含噪声和探针估计误差。identity 模式不训练网络，固定 m=1；no_gradient 模式仅将网络看到的 d 通道清零，物理更新仍使用真实 d。后者用于判断把梯度作为网络输入是否额外有用。

`feedback_step` 默认 0.01 是物体归一化梯度步长；`lr_net` 默认 0.002 只优化权重网络；`lr_probe` 默认 0.01。像素参数化、unet_activation 不用于本方法的状态投影。当前仅支持 zero-phase + Poisson，禁止 DWT/detail/TV/TGV/网络衰减组合，以保持首轮实验可解释。

## Colab

先同步本地代码到 Colab。本次不自动 commit/push。新增 feedback.py，修改 config.py/reconstruct.py/run_simulation.py/report.py。用 --help 检查是否支持 feedback_shared_amp。

```python
%cd /content/ptychography_AD_DIP
!python -m multiwave.run_simulation \
  --preset resolved --device cuda \
  --methods feedback_shared_amp \
  --feedback-mode learned --feedback-step 0.01 \
  --probe-mode pixel --base-channels 16 \
  --loss poisson --photons-per-scan 20000 \
  --iterations 1000 --eval-every 25 \
  --scene-seed 17 --noise-seed 24 --network-seed 31 \
  --lr-net 0.002 --lr-probe 0.01
```

控制组只将 learned 换成 identity；梯度输入消融换成 no_gradient。每次独立输出新目录。与原 U-Net 对照时使用 `--methods unet_shared_amp --unet-activation softplus --unet-skip concat --unet-detail none`，其余实验场景及预算相同；像素 AD 使用 `--methods pixel_shared_amp --pixel-parameterization softplus`。

每次反馈迭代有两遍完整训练数据梯度计算（产生反馈及训练候选），原方法只有一遍；另有两次权重网络前向。JSON 记录 training_data_gradient_passes 和实际耗时，不能把同迭代数当作同计算预算。identity 为实现一致仍保留零网络梯度路径。梯度次数不含评估，不能替代实际耗时。

保存 state_dict 含当前 amplitude 和网络，probe_state_dict 含探针；可以精确重载最终物体，但不保存 Adam 动量，不宣称完整断点续训。

## 验证与结论边界

51 项 multiwave 测试通过，包括全量/分块物理梯度一致、初始单位权重等价、训练不依赖真值/评价 ROI/留出观测、网络实际更新、状态重载、共享零相位和 identity 控制。测试仅检查实现，不证明细节改善或长期稳定性；未运行 1000 次效果实验。

实际 resolved 384×384、base16、pixel probe、20000 光子、noise_seed=24 的两步 CPU 集成运行完成，输出目录 `results/20260930_003415_739360`，成功保存指标、checkpoint 和图片。Poisson 目标 1.5248 → 1.3032，仅作为功能检查，不作为方法优越性证据。
