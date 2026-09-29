# 扩大探测器后的 U-Net / AD 对比

更新时间：2026-09-29T18:18:33.925120+10:00；状态：**running**；进程 PID：25416。

物体 384×384、照明探针 192×192、探测器 768×768、1 μm 像素、1.5 mm 传播距离、515/633 nm。FFT 为 1536×1536；无噪声；25 个扫描位置中 20 个训练、5 个留出。

所有主对照使用相同场景／测量／网络种子（17/23/31），共享零相位物体。AD 指直接优化物体像素参数，U-Net 指优化网络权重；两者均通过自动微分优化同一物理损失。

主对照 AD-softplus 与 U-Net-softplus 使用相同非负振幅参数化、相同常数振幅初值；U-Net 宽度为 16/32/64/128、原 ProPtyUNet 主干、无相位头。网络固定输入可降采样到物体尺寸，但前向和损失始终保留 768×768 全部测量像素。

AD 学习率 0.03；U-Net 0.002；盲探针 0.01。主对照各 500 次更新；额外 AD-direct [0,1] 投影基线为 300 次，仅作为几何校验，不用于等预算优劣宣称。盲探针均从相同的平滑圆斑和零相位开始，各波长探针功率固定为 0.5。

| 实验 | 状态 | 更新 | ROI 振幅相对误差 | 中央 RMSE | 高频误差 | 留出振幅误差 | 平均探针误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| known_ad_direct | completed | 300/300 | 0.000130159 | 0.00016665 | 0.000726876 | 0.0146004 | 0 |
| known_ad_softplus | completed | 500/500 | 0.0438112 | 0.0531193 | 0.254928 | 0.0299826 | 0 |
| known_unet_softplus | completed | 500/500 | 0.0394945 | 0.0708341 | 0.330633 | 0.014802 | 0 |
| blind_ad_softplus | running | 425/500 | 0.105876 | 0.152313 | 0.710392 | 0.082843 | 0.118306 |
| blind_unet_softplus | pending | 0/500 | — | — | — | — | — |

评价区域沿用名义照明 ROI。中央与高频指标定义见 RESOLUTION_DIAGNOSIS.md。记录最后迭代，不按真值误差选择最佳模型。留出图不参与训练或网络输入。单种子、无噪声结果只支持本次仿真判断，不代表统计优势或真实实验性能。

## 结果与解释

- known 主对照：AD ROI 误差 0.0438112，U-Net 0.0394945；高频误差分别 0.254928 / 0.330633。只描述本次固定预算结果，不将方法差异全部归因于网络结构。
- blind 主对照尚未全部完成，暂不下结论。

## 结果文件

- known_ad_direct：[C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/known_ad_direct](C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/known_ad_direct)

![known_ad_direct detail](C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/known_ad_direct/pixel_shared_amp_detail.png)

- known_ad_softplus：[C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/known_ad_softplus](C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/known_ad_softplus)

![known_ad_softplus detail](C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/known_ad_softplus/pixel_shared_amp_detail.png)

- known_unet_softplus：[C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/known_unet_softplus](C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/known_unet_softplus)

![known_unet_softplus detail](C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/known_unet_softplus/unet_shared_amp_detail.png)

- blind_ad_softplus：[C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/blind_ad_softplus](C:/Users/kennyzz/Desktop/INNM_Code/multiwave/results/resolved_ad_unet_20260929/blind_ad_softplus)
