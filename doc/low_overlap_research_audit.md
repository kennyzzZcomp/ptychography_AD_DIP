# 低重叠率、稀疏计算与盲重建：研究审查与最小实验

日期：2026-09-22。状态：**pilot-ready**，不是已经实现性能提升或证明新颖性。
范围：ProPtyNet_paper 的固定传播模型、未知探针二维重建；允许后续换成 INR；不做 z 轴优化。
完整训练仅在 Colab 手动运行。本次只做代码审查、文献核对、微型 CPU 代数检查。

## 1. 当前最值得验证的论文命题

“低重叠盲重建的计算预算，不应仅按衍射图数量分配，而应保留对 object–probe
补偿误差最敏感的测量；在相同墙钟预算下，这种更新调度比随机 minibatch 更可靠。”

这是**待验证假设**。不是声称“低重叠 + DIP”“加 TGV”“换 INR”或者
“隔几张输入一张”本身新颖。先证明机制与实际收益，再决定使用 U-Net 还是 INR。

最可能的录用理由：有可复现的失败机制、非 GT 选图规则、跨表示的质量–时间收益。
最可能的拒稿理由：只是现有 minibatch / experimental design 的应用，且不比随机抽样快。

## 2. 代码层面必须先分清的三件事

|集合|含义|只减少它会改变什么|
|---|---|---|
|K_in|固定送入 U-Net 的输入通道|网络条件输入、首层成本；不自动减少物理约束|
|K_loss|每步参与 forward / loss 的扫描点|每步传播成本和梯度；若轮换仍可使用全部已采数据|
|K_acq|实际采集/允许使用的数据|采集时间、剂量和可恢复信息；不能偷偷用未采点选图|

`functions/paperrepro/solvers_addip.py:run_net` 当前把所有衍射图作为输入通道，
每步也计算所有扫描点的数据项。`pos_batch` 虽在配置中，但当前此分支未使用它；
`fwd_chunk` 仅分块计算，不等于少使用数据。

固定输入少几张，不能宣称低剂量；全量测量仍参与 loss 时，也不能称低采集重叠。
固定 grid 增大 step 同时扩大照明 FOV，不能只用 overlap 一个数字解释性能。

### 输入通道减少的成本核算

对应现有 base=32、双相位头、624×624 输入，仅计正向卷积 MAC：

|输入张数变化|卷积 MAC 减少|卷积计数比值 before/after|
|---|---:|---:|
|16 → 4|2.26%|1.023|
|64 → 4|10.36%|1.116|
|100 → 4|15.61%|1.185|

只第一层卷积随输入通道数变化。这不是 GPU 计时或端到端加速上界：没有包含 BN、
反向、优化器、FFT、显存传输和 TGV，实际 kernel 效率也会变化。必须实测。
若全图 U-Net 占主要时间，减少 K_loss 也未必足够快；此时局部坐标 INR 的价值在于
只查询当前照明区域，而不是因为“INR”这个名字新。

## 3. 最近工作及其具体边界

检索覆盖：ProPtyNet、blind/low-overlap DIP 与 INR、raster ambiguity、minibatch、
adaptive scanning、hybrid overlap/probe calibration；核对到 2026-09-22。
这不是系统综述，也不足以证明“无人做过”。摘要级证据不用于细节性能结论。

### A. 像素优化与测量选择

1. **Raster Grid Pathology and the Cure**，Fannjiang，2019，Multiscale Modeling & Simulation。
   [论文](https://epubs.siam.org/doi/10.1137/18M1227354)。
   类型：理论；创新点：刻画规则扫描的周期歧义及扰动扫描的消歧条件。
   数据/指标：不是神经网络恢复基准；以可辨识性结论为主。
   解决：为什么数据一致并不等于盲重建正确；未解决：给定计算预算的神经更新调度。
   evidence status：verified，摘要/理论结论核对；role：ancestor / contrary evidence。
   支持 claim：图连通不充分，网络先验不能凭空创造测量信息。置信度高。

2. **Stochastic minibatch approach to the ptychographic iterative engine**，2024，Optics Express 32, 30199–30225。
   [机构记录](https://escholarship.org/uc/item/4381990k)，[DOI](https://doi.org/10.1364/OE.530136)。
   类型：优化方法；创新点：将随机 minibatch 引入 PIE，改善收敛行为。
   数据/指标：本轮仅核对摘要，具体数据集与计时协议未复核。
   解决：更新顺序/批次对优化的影响；未证明：本项目的弱模式调度优于其实现。
   status：verified，摘要级；role：closest acceleration baseline。
   支持 claim：“随机少算一些图”已有直接先例。置信度高。

3. **Deep reinforcement learning for data-driven adaptive scanning in ptychography**，2023，Scientific Reports。
   [论文](https://www.nature.com/articles/s41598-023-35740-1)。
   类型：采集策略；创新点：学习数据驱动的自适应扫描策略。
   数据/指标：电子 ptychography 场景；详细拆分及数值本轮未完整复核，不引用胜率。
   解决：按信息选择扫描位置；与本项目区别：我们首先只调度已有测量的计算，不训练采集策略。
   status：verified；role：sibling / novelty boundary。
   支持 claim：广义“智能选扫描点”不是空白。置信度中。

### B. 无监督先验与低重叠恢复

4. **Compressive Ptychography using Deep Image and Generative Priors**，2022，arXiv 2205.02397。
   [预印本](https://arxiv.org/abs/2205.02397)。
   类型：方法；创新点：结合深图像/生成先验、判别器与 TV 处理压缩 ptychography。
   数据/指标：低重叠与噪声仿真；具体数据集和数值本轮未复核。
   解决：用先验减少所需扫描；不证明：当前 TGV 或新选图规则有效。
   status：verified preprint，摘要级，未核实最终期刊版；role：ancestor / closest prior。
   支持 claim：DIP + 稀疏扫描 + 正则已经存在。置信度高。

5. **Noise-robust ptychography using unsupervised neural network**，2025，Optics and Lasers in Engineering 186, 108791。
   [论文](https://doi.org/10.1016/j.optlaseng.2024.108791)。
   类型：方法；创新点：将未训练网络与传播物理结合，强调噪声鲁棒性。
   数据/指标：仿真与实验；本轮未取得完整全文，不把本地复现细节视为论文原设定。
   解决：本项目方法锚点；未解决：当前 code 的真实时间瓶颈和扫描子集效应。
   status：verified，出版信息/摘要级；role：task anchor。
   支持 claim：需要在已有无监督物理方法之上建立机制贡献。置信度高。

6. **Learning neural representations for X-ray ptychography reconstruction with unknown probes (PtyINR)**，
   2026，npj Computational Materials；2025 年已有预印本。
   [期刊状态](https://research.cuhk.edu.hk/en/publications/learning-neural-representations-for-x-ray-ptychography-reconstruc/)，
   [技术版本](https://arxiv.org/html/2509.04402v1)，[代码](https://github.com/TISGroup/PtyINR)。
   类型：方法；创新点：(1) 联合神经表示物体和未知探针；(2) 物体与探针采用不同编码；(3) 低信号验证。
   数据/指标：仿真、NSLS-II HXN 实测；PSNR、FRC。
   边界：v1 按 probe FWHM 给出的负 overlap，不等于完整阵列无交叠；作者指出极稀疏实测比仿真更难。
   status：verified，技术细节依据 v1，不假设最终版完全相同；role：closest prior / required baseline。
   支持 claim：仅换 INR 做低重叠 blind reconstruction 不足以建立新颖性。置信度高。

7. **Residual neural-field ptychography for dose-efficient electron, X-ray, and optical nanoscopy**，2026，arXiv 2601.17694。
   [预印本](https://arxiv.org/abs/2601.17694)。
   类型：方法；创新点：围绕物理先验建立复值 residual neural field，覆盖多种成像模态。
   数据/指标：摘要确认跨模态验证；本轮未复核全文指标。
   status：verified preprint，摘要级；role：recent sibling。
   支持 claim：残差神经场/低剂量也已有先例；其 z 等校正不纳入本项目。置信度中。

### C. 最容易撞题的探针稳定方案

8. **Sparse-Scan Ptychography with Hybrid-Overlap Scanning and Reference-Guided Probe Updates**，
   2026-08-14，Sensors 26(16), 5168。
   [论文](https://www.mdpi.com/1424-8220/26/16/5168)，[出版记录](https://pubmed.ncbi.nlm.nih.gov/42655476/)。
   类型：扫描与更新方法；创新点：(1) 局部 dense + 全局 sparse；(2) 高重叠参考引导 probe 更新。
   数据/指标：FZP/KB 探针仿真、软 X 射线实验；恢复图与分辨率分析。
   解决：稀疏测量下探针不稳定；与我们区别：首先使用固定已采数据预算，不额外建立 dense calibration 区。
   status：verified，出版信息和方法/结果文本；role：closest prior / contrary evidence。
   支持 claim：不能把“少量高重叠区校准 probe”直接包装成新贡献。置信度高。

### 趋势与饱和度

2018–2019：规则扫描可辨识性 → 2022：深先验压缩 → 2023–2024：自适应采集/随机批次
→ 2025–2026：联合 INR、跨模态神经场、混合扫描校准。

已拥挤：加先验、换表示、随机少算、dense reference 稳 probe。
待查证缺口 **inferred gap**：对盲重建弱耦合误差有针对性的、开销足够低的计算调度，
是否能跨 U-Net / INR 在同等墙钟预算下稳定优于简单随机批次。不是“文献不存在”的断言。

## 4. 三个候选，及为什么只优先一个

### 弱／安全：TGV + 固定小输入通道

主张：正则能改善部分低 overlap 图像，小输入节省部分资源。
最近工作：压缩 DIP、ProPtyNet。机制：振幅空间先验与输入压缩。
最小实验：同一 scene 下 TGV λ=0/1e-4/1e-3/1e-2，输入 full/4 张/固定随机张量。
基线：原版 net、TV、像素 AD；统一 ROI 和数据项。
风险：磨平细线，输入压缩只影响首层；GT 调 λ 导致过拟合。
审查：工程上合理，**不是单独的强论文点**。若最优固定随机输入也一样，不能声称选图具有信息优势。
停止条件：独立对象无改善或端到端时间节省 <10%；可留作本科论文/技术报告。

### 中等／优先：弱模式感知的计算调度

主张：每步选图不是只追求覆盖或最大残差，而是保留对 object–probe 补偿误差的敏感性。
最近工作：raster theory、stochastic minibatch PIE、PtyINR、reference-guided probe updates。
为什么仍值得试：现有代码的固定 stride 能额外放大可补偿模式；可以直接构造反例和控制实验。
为什么可能不成立：随机轮换可能已经足够；挑图成本可能超过节省；当前16帧规模过小。
证据类型：理论先例 + 本地代数反例；效果证据尚缺。
最小验证：fixed stride / random / rotating residues / proposed，在相同累计图数和墙钟下比较。
合理性判断：**pilot-ready**；非 GT scoring 是待实现项，不能把诊断工具当作方法已完成。
停止条件：多 seed 优势消失、只赢 fixed stride 不赢 random、或只靠额外数据/更多更新取胜。

### 强／高风险：表示无关的可观测性—计算分配框架

主张：同一组局部物体–探针弱模式解释 U-Net、像素 AD、INR 在稀疏扫描下的失败，
并据此统一控制测量批次和 probe 更新频率。
机制候选：局部线性化 + 小型敏感性 Gram 矩阵，排除固有 gauge 后优化弱方向约束。
最近工作：上述理论/优化工作；信息量选测量本身不是新数学。
最小验证：先在像素场小问题检查预测性，再跨两种表示复现，不从大模型直接开始。
风险：局部谱与全局非凸优化行为不相关；网络参数 Jacobian 与像素场 Jacobian 差异大。
审查：有分析型/方法型论文潜力，但目前证据弱；不能承诺会议/期刊水平。
停止条件：所提分数不比 overlap、残差或随机分数更能预测失效；回退到优化工程结果。

## 5. 顶部候选的数学边界和方法草案

设扫描位置 r_j，探针局部坐标 u，出射波 ψ_j(u)=O(r_j+u)P(u)。
对任意非零函数 g，令 O'=Og、P'=P/g（将参考扫描位置平移为0）。
若 g(r_j+u)=g(u)，所有出射波完全不变，任何共同线性传播后强度也不变。

原始 d-raster 上，d-periodic g 是已知歧义。只取每隔 k 行和 k 列的子集，
kd-periodic g 也可不变。加入原始 raster 中不同余类的点，可以打破新增模式，
但不可能凭这些原始位置消除原本 d-periodic 歧义。off-grid 真采集才改变后一个问题。
这里讲二维行列 stride；不能把扁平数组 `[::k]` 不加检查地视为同一种扫描。

这一已知理论的应用不是我们的创新。它提供可控实验：loss 一样的小，不代表恢复正确。
几何图连通、TGV 或神经先验也不是盲解唯一性的证明。

候选评分从小组测试场 h_q 出发：δO=O h_q，δP=−P h_q，则

    δψ_j(u) = ψ_j(u) [h_q(r_j+u) − h_q(u)]
    δU_j = F_fixed(δψ_j)
    δ|U_j| = Re(conj(U_j) δU_j) / max(|U_j|, epsilon)

用当前估计（不能用 GT）形成小维度 B_j，汇总 G(S)=Σ B_j^T W_j B_j。
选择子集时提升弱方向覆盖，例如正则化 logdet 贪心；W 与噪声模型一致。
必须去除对全部已采数据都为零的固有 gauge 方向，否则永远改善不了最小特征值。
不声称这些有限 h_q 覆盖全部病态方向，更不声称 logdet 是新算法。

成本约束：每隔若干步刷新分数；刷新、候选传播、筛选开销全部计入训练时间和测量使用量。
第一版先实现无额外 forward 的 rotating-residue / spatially-stratified baseline。
若它与 random 无差异，没有理由马上投入昂贵的 Gram 评分。

重要：当前 amplitude loss 每步解析消去全局缩放。独立对每个 minibatch 求缩放，
通常不等于完整数据的 VarPro 目标的无偏梯度。必须显式定义新目标、固定公共尺度，
或定期全量估计尺度并标记为近似；不能直接截取 Ua 然后声称 loss 原封不动。
非均匀抽样如要估计同一平均损失，需记录选择概率并校正，且分析 VarPro 的非线性影响。

## 6. Colab 最小实验卡与判定标准

第一阶段（首先要做）：torch profiler / CUDA event 分别测网络、传播、反向、优化器、
TGV、evaluation；预热后计时并 synchronize，固定 GPU 型号、precision 和批次。
不把减少打印或少做 evaluation 的时间算作算法优势。

第二阶段（mechanism pilot）：

1. 从一个 master scene 生成一次数据，再索引子集，不为每个方法重新生成噪声/归一化。
2. 区分 acquired-set 与 iteration-set。固定 acquired-set 时所有方法可用数据完全相同。
3. 原版全量、固定 stride、均匀随机无放回、空间分层、轮换余类：先保持同一 K_in，
   只改 K_loss。随后再做 K_in 消融，防止混在一起。
4. 数据图数、对象 FOV、probe、支持约束、初始场、学习率搜索预算保持一致。
5. 小 pilot 用3个 init seeds；主表至少5个，并另外更换场景。当前 `cfg.seed` 同时
   控制真实 probe 纹理和网络初始化，不能把改变它称为“同一数据的多初始化”。应先拆 seed。
6. 先无噪声定位机制，再 Poisson；固定每点剂量与固定总剂量是两种实验，分开报告。
7. 检查60/50/40%等低 overlap，但阈值依赖样本和 probe；不要把当前失败曲线当普遍定律。

主要终点：同固定ROI、同质量阈值下的 wall-clock；未达到阈值记失败，不能删掉。
同时报告按时间的质量曲线、累计 forward 图数、峰值显存、probe误差、细线分辨率和 held-out loss。
评价只消除明确的全局/仿射 gauge，不允许对每种方法拟合任意周期校正来掩盖伪影。
固定 ROI 必须在共同有效照明域内；同时报告照明域覆盖率，防止中心小 ROI 掩盖外围失败。
held-out 测量不参与输入、选图或调参；若来自同一 lattice，它也可能看不见原生周期歧义。

基线层级：

- 最小：原版 net、等 K_loss random/stratified/rotating、像素 AD、原生 PtyLab-ePIE。
- 强：调参/收敛充分的 rPIE 或其他优化基线、PtyINR；若实际改变采集策略，再加入 hybrid-overlap。
- TGV：作为固定权重消融，不默认开启。应检查 TV 和无正则，以分离 prior 与选图收益。

不要求用户做 known-probe ePIE；若使用 probe 固定为 GT 的内部诊断，必须明确它只是
区分 object 与 probe 困难的 oracle 对照，不放进可部署 blind 方法排名。

预注册的继续条件建议：同质量下 median wall-clock 至少快20%，且固定时间下振幅/相位
指标不明显劣化；至少两个非同类 object、两种 probe、多个 init seed。20%是项目门槛不是文献事实。
若只在 USAF 成立：降级为样本依赖结果。若只赢短迭代 ePIE：不能继续写 superiority。
若在 full U-Net 上瓶颈不可消除：保持算法问题，迁移可局部查询的 INR，而非反复改首层。

## 7. Claim–Evidence 与审稿前检查

|主张|现有证据|缺少什么|
|---|---|---|
|减少输入通道主要动首层|本地网络结构 + MAC 算式|Colab 实测时延|
|固定 stride 可新增周期歧义|既有 raster 理论 + 微型构造检查|实际学习轨迹是否沿这些模式漂移|
|弱模式评分优于 random|目前没有；inferred gap|等预算、多样本、跨表示实验|
|适合论文而非拼模块|仅研究命题与可证伪设计|新机制、实测数据和强基线|

审稿人可能问：

- “这不就是 minibatch？”——必须展示选择机制和 random 的差异，否则接受这个否定。
- “dense anchors 已有人做。”——主设定不增加采集；若增加，必须对照 reference-guided 方法。
- “省图但不省时间？”——计入评分开销，质量–墙钟曲线是主指标。
- “模型只偏好靶图？”——增加非周期对象、弱振幅对象以及真实测量，不能只测 USAF。
- “网络恢复的是先验幻觉？”——独立测量一致性与细节验证，但承认 lattice 固有不可辨识性。

回退路线：若新调度不成立，保留一份揭示输入压缩/计算稀疏/采集稀疏混淆的复现报告；
不要为了保住题目堆叠 TGV、attention、INR 来制造贡献感。

## 8. 本次实际交付与未完成项

新增 `simulations/low_overlap_audit.py`：仅 NumPy 的微型 gauge 构造和解析 MAC 估算，
输出 JSON，不训练、不重建、不改现有 solver。5个单元测试通过。
toy 强度相对误差：stride2 的新增周期场约3.1e-16；加入原生偏移点后约0.109；
原始周期场仍约4.4e-16。该比较点数不同，**仅验证代数，不是公平恢复实验**。
加入两个 off-grid 点只打破这里构造的两个场，不证明盲解全局唯一。

Colab 可直接运行：

```python
!python /content/ptychography_AD_DIP/simulations/low_overlap_audit.py
!python -m unittest discover -s /content/ptychography_AD_DIP/tests -p test_low_overlap_audit.py -v
```

运行测试前 cwd 应为 `/content/ptychography_AD_DIP`，以便导入 simulations 包。

现有 object-amplitude TGV 的实现/命令见 `simulations/README_paper_tgv.md`。
本次没有重新修改 TGV，也没有运行完整 simulation。上述选图调度和 profiler 接入尚未实现，
没有声称达到“低重叠率好效果”。下一步按第一阶段先测真实瓶颈，再决定实现哪条路线。

### 后续进展：相位正则与计时已接入

用户提供的截图报告：40%设置、振幅TGV权重0.1、固定96×96 ROI，2000步最终
振幅PSNR约22.79dB；用户报告无TGV约7dB。前者有截图，后者尚未取得同配置原始
结果文件核对。这是值得追踪的单次改善证据，不等于多场景、多初始化的研究结论。

`--tgv-phase` 已独立接入；另新增 `--timing-warmup` 同步分阶段计时，默认关闭。
3项计时测试与11项TGV/微型接线测试通过，接线测试确认开启计时不改变CPU微型结果。
精确Colab短诊断命令见 `simulations/README_paper_timing.md`。
仍未在本地运行完整重建；GPU瓶颈、相位TGV效果与实际加速收益待Colab验证。
选图调度仍未实施，后续先根据实际时间分解决定减少输入、传播批次或局部INR路线。

### 后续进展：渐进物理测量约束 pilot 已接入

已在net实现measurement-schedule及fixed/random/rotate三种政策；显式0:1作为
同记录开销的全量对照，默认空值保留旧行为。所有政策最后进入完整测量阶段。
此版本固定全量网络输入，**尚未实现用户设想的输入通道渐进加入**，不以此缩小原目标；
它先隔离loss测量调度这一个变量。每步索引、传播帧数、全量评价误差和运行时间均保存。
早期子集VarPro的目标变化已明确记录，不声称是全量目标的无偏梯度。
5项采样/物理算子小测试与11项TGV/接线回归通过，没有跑完整重建。
Colab四组命令与质量–时间绘图见simulations/README_paper_curriculum.md。
效果与加速仍未经GPU实验验证，论文主张未完成。

### 后续进展：输入端渐进已补上

新增input-policy=follow_measurements，通过首层输入与权重的同索引gather，真正减少
子集阶段首层卷积通道数，而非仅置零。网络全部Parameter与优化器状态持续保留。
默认full仍作为loss-only控制；显式全量日程与原数值路径一致。4项输入/梯度/参数身份
测试和11项接线回归通过；第5组Colab对照见README_paper_curriculum.md。
新通道加入可能造成输出冲击与BatchNorm状态变化，尚未证明带来净加速，也未测试GPU。
因此当前具备测试用户完整想法的代码，不等于已获得可发表结论。
