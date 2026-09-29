# Colab：AD、U-Net、学习率衰减与 DWT/TGV 实验记录

记录日期：2026-09-29。以下数值来自用户在当前聊天中粘贴的运行日志/报告和截图，未取得这些 1000 次实验的完整 config.json、metrics.json 或原始重建数组。保留原始报告精度，不补造缺失指标，不将不同运行合并成一次实验。

## 共同任务和可比性

共享实数 USAF 振幅，物体相位固定零，515/633 nm，两个独立复数像素 probe，equal_power，每模式功率 0.5。无噪声（photons-per-scan=0），1000 次更新，训练/留出扫描数为 20/5。

图中物体为 384×384、1 μm/px；运行日志的 padding difference 为 8.85e-05。所给运行指令采用 resolved、base_channels=16、softplus 物体参数化、学习率 AD=0.03 / U-Net=0.002 / probe=0.01、种子 17/23/31。由于缺少各组完整配置文件，这些细项尚未逐文件核验，不能声称数据和环境逐位一致。

AD 在本记录中指像素参数化；U-Net 同样采用自动微分。仅使用预定的最终迭代比较，未按真值选择最优迭代。TGV=0 时，不执行辅助优化；报告仍列出的辅助步数 5 只是默认配置。

## 最终结果（数值越低越好，耗时除外）

| 编号 | 实验 | 训练振幅 NRMSE | 留出干净振幅 NRMSE | ROI 物体相对误差 | 平均 probe 相对误差 | 耗时 s（含评价） |
|---|---|---:|---:|---:|---:|---:|
| A1 | AD，固定学习率，无 TGV | 0.0073 | 0.0500 | 0.0223 | 0.0343 | 未提供 |
| U1 | 原 U-Net，固定学习率，首次记录 | 0.0088 | 0.0109 | 0.0269 | 0.0378 | 未提供 |
| U2 | 原 U-Net，第 501 次网络 LR 0.002→0.0004 | 0.0138 | 0.0205 | 0.0474 | 0.0747 | 未提供 |
| W1 | DWT-U-Net，dwt_concat，TGV=0 | 0.00653 | 0.01030 | 0.02427 | 未提供 | 133.45 |
| U3 | 原 U-Net 重跑，concat，TGV=0 | 0.00895 | 0.01330 | 0.03046 | 未提供 | 128.84 |
| AT1 | AD+TGV，beta=0.001 | 0.00717 | 0.04413 | 0.01911 | 0.0289 | 141.51 |
| UT1 | 原 U-Net+TGV，beta=0.001 | 0.00975 | 0.01195 | 0.02981 | 0.0449 | 149.64 |

中心 RMSE、高频相对误差、细条纹分辨率未在本轮用户报告中提供；不能从截图或整体 ROI 误差反推。

## 来源和独立运行标识

- A1：用户粘贴 0–1000 更新日志；Colab 目录 `/content/ptychography_AD_DIP/multiwave/results/20260929_083609_254548`。
- U1：同一消息中的另一份 0–1000 日志；Colab 目录 `/content/ptychography_AD_DIP/multiwave/results/20260929_083701_195913`。
- U2：用户“分学习率直接很差了”消息中的 0–1000 日志；Colab 目录 `/content/ptychography_AD_DIP/multiwave/results/20260929_085221_576438`。
- W1：用户“这个是加上DWT的仿真运行记录”报告及截图；运行目录未提供。
- U3：用户“原unet又运行了一次”报告及截图；运行目录未提供。
- AT1/UT1：用户最新 TGV 汇总报告，以及附件 `f6048e6a-e49f-4b80-84ed-3903e6289f80/已粘贴的文本.txt` 的两组完整日志。共同 Colab 目录 `/content/ptychography_AD_DIP/multiwave/results/20260929_093054_131887`。精度较高的最终训练/留出/物体误差和耗时采用汇总报告；probe 指标采用四位小数日志。原日志归档 `results/colab_ablation_evidence_20260929/tgv_1000_stdout.txt`，解析后 82 条评价记录见同目录 `tgv_1000_history.csv`，来源 SHA256 见 `tgv_log_provenance.json`。
- W1/U3 截图归档在 `results/colab_ablation_evidence_20260929/`，对应 `dwt_unet_detail.png` / `unet_repeat_detail.png`；原文件路径及 SHA256 见该目录 `image_provenance.json`。截图本身不能替代配置或指标 JSON。

早期 CPU、已知 probe、500 次 GPU 实验分别保留在 [AD_UNET_COMPARISON.md](AD_UNET_COMPARISON.md) 和 [COLAB_BLIND_UNET_20260929.md](COLAB_BLIND_UNET_20260929.md)，不与本表的 1000 次盲重建混作同预算比较。已停止的后台套件仍保持停止。

## 基于本轮结果的分析

1. W1 对比最新重跑 U3：ROI 误差从 0.03046 降到 0.02427，相对降低约 **20.3%**；训练误差约降低 27.0%，留出误差约降低 22.6%；报告耗时从 128.84 s 增到 133.45 s，约增加 **3.6%**。这是两次已报告运行的差值，不是重复实验统计结论，也不能排除硬件负载差异。
2. W1 对比早期 U1：ROI 误差相对降低约 9.8%；对比 A1，W1 的 ROI 误差仍高约 8.8%。因此目前支持“DWT 在这些 U-Net 运行之间有改善迹象”，不支持“DWT 已超过 AD 的物体精度”。
3. U1 与 U3 的 ROI 误差分别为 0.0269、0.03046；两次不同结果说明必须保留运行间差异。其来源尚未定位，不擅自归因于 GPU 非确定性、种子或代码变更。
4. U2 的后半程更平稳，但固定预算内误差更高。U2 在第 500 次（尚未衰减）物体误差已为 0.0765，而 U1 是 0.0654；不能把全部最终差距归因于调度。默认继续采用固定学习率。
5. W1/U3 图中大条纹较清楚，但中心细条纹和剖面仍有差异，ROI 外重复纹理仍存在。图像仅供定性参考，不宣称全视场恢复或量化分辨率提升。
6. 目前只有 W1 的一次 DWT 运行，不能做显著性结论。高频收益需完整 metrics.json；性能结论还需要配置核对、成对多种子、容量/耗时对照及相同噪声条件下的验证。

## TGV 运行后的波动分析

在每 25 次记录一次的 40 个相邻区间中，AD 的训练、物体和 probe 误差均未出现上升；这不等于证明每个内部更新都单调。U-Net 训练误差有 8 次上升（终点 475、625、725、775、825、900、950、1000），物体误差有 5 次上升（725、775、825、900、1000），probe 误差没有上升。

例如 U-Net 875→900→925 次：训练误差 0.0096→0.0128→0.0090，物体误差 0.0330→0.0366→0.0311；反弹后恢复。975→1000 次：训练误差 0.0081→0.0097，物体误差 0.0288→0.0298，probe 0.0461→0.0449。总体物体误差 500→1000 次从 0.0759 降到 0.0298，故当前证据更符合下降过程中的间歇震荡，不能称为持续崩溃或证明过拟合。

打印的 train 是测量振幅 NRMSE，不是包含 TGV 的完整目标。真实优化目标是 train_NRMSE² + beta*R(A,v)（TV 关闭）。当前日志未含辅助场或 penalty 历史，不能直接判定每次 train 反弹对应总目标上升。

AT1 的 ROI 误差 0.01911 低于历史 A1 的 0.0223；UT1 的 0.02981 与最新 U3 的 0.03046 接近，但差于历史 U1 的 0.0269。TGV 尚未使 U-Net 稳定胜出，不能从跨运行差异判断统计显著性。

建议先补记录每次更新的完整训练目标与物体实际更新幅度，再测试基于训练目标的候选更新检查/回溯步长。不得用真值或留出误差接受更新。启用 TGV 时应固定辅助场进行候选比较，或独立检查辅助更新是否下降；候选前向必须使用一致的 BN 模式并妥善还原 buffers，拒绝整步时还需处理 Adam 状态。此为待实现方案，不宣称已验证。即便完整训练目标不升，也不能保证真实物体误差或高频误差单调；额外前向成本需计入 AD/U-Net 公平对比。

## 已完成的 AD+TGV 与原 U-Net+TGV 指令

### 后续截图补录：8000 光子对照与无噪声 DWT+TGV

用户随后提供三张截图，并明确确认第三张是 DWT+TGV：运行后修改了 Notebook 单元格中的命令，因此截图里的 concat 不代表该次运行设置。第三张按用户确认归类，前两张单独归入 8000 光子实验。

| 编号 | 截图所示设置 | 入射光子/扫描 | train | holdout clean | object | probe | Colab 结果目录末级 |
|---|---|---:|---:|---:|---:|---:|---|
| N1 | pixel_shared_amp，命令未显式启用 TGV | 8000 | 0.7779 | 0.4378 | 0.4301 | 0.8964 | 20260929_095233_838610 |
| N2 | unet_shared_amp，dwt_concat，命令未显式启用 TGV | 8000 | 0.7798 | 0.4472 | 0.4558 | 0.8981 | 20260929_095452_916583 |
| WT1 | DWT+TGV（用户确认），TGV=0.001 | 0 | 0.0068 | 0.0095 | 0.0249 | 0.0392 | 20260929_094706_786107 |

以上为截图四位小数读数，耗时及完整配置未提供。第三张同目录的 AD 最终日志为 train=0.0072、holdout=0.0441、object=0.0191、probe=0.0289，与 AT1 的四位小数一致；不据此认定是同一运行或复制数据。用户已确认第三张运行时启用了 DWT，截图命令后来被修改；按该确认记录为 WT1，原始 config.json 尚未归档。

N1/N2 属于 8000 光子 Poisson 实验，不与前述无噪声指标混比。目前该噪声条件下 DWT-U-Net 没有超过 AD：物体误差分别 0.4558、0.4301，probe 误差均接近 0.90，重建质量明显不佳。第三张低误差也不能归因于 DWT+TGV 抗噪，因为该组为无噪声。三个截图及路径/SHA256 归档于 results/colab_ablation_evidence_20260929/noisy_and_tgv_screenshot_provenance.json。

无噪声 WT1 的物体误差 0.0249，相比 DWT 单独 W1 的 0.02427 高约 2.6%；相对最新原 U-Net 重跑 U3 的 0.03046 低约 18.3%；仍高于 AD+TGV 的 0.01911。这些单次结果不支持在当前 beta=0.001 下，TGV 对 DWT 存在额外的物体精度收益。完整高频指标和耗时未提供，不作相应结论。

下面一条命令顺序完成两组，并共用一次生成的测量、扫描划分和物体真值，保存到同一个新时间戳目录。两组都使用 softplus，probe 固定学习率 0.01，不启用 DWT 或学习率衰减。beta=0.001 是初始候选值，不是已验证的最优值。

```python
%cd /content/ptychography_AD_DIP

!python -m multiwave.run_simulation \
  --preset resolved --device cuda \
  --methods pixel_shared_amp unet_shared_amp \
  --probe-mode pixel --spectral-mode equal_power \
  --pixel-parameterization softplus --unet-activation softplus \
  --unet-skip concat --tgv-weight 0.001 --tv-weight 0 \
  --tgv-alpha0 2 --tgv-alpha1 1 \
  --tgv-eps 0.001 --tgv-lr 0.01 --tgv-inner-steps 5 \
  --base-channels 16 --iterations 1000 --eval-every 25 \
  --photons-per-scan 0 \
  --lr-net 0.002 --lr-pixel 0.03 --lr-probe 0.01 \
  --lr-net-decay-after 0 \
  --scene-seed 17 --noise-seed 23 --network-seed 31
```

若希望分别运行，用 `--methods pixel_shared_amp` 跑 AD+TGV，或 `--methods unet_shared_amp` 跑 U-Net+TGV，其余参数保持不变。保存两份 metrics.json 和 config.json，后续可补全中心/高频/probe 指标与耗时。TGV 作用范围和算法详见 [DWT_TGV_EXPERIMENTS.md](DWT_TGV_EXPERIMENTS.md)。
