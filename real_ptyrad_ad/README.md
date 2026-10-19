# 官方 PtyRAD AD 实验对照与诊断

状态（2026-10-19，Australia/Sydney）：用户已安装官方 PtyRAD 1.0.0。本地已完成小尺寸与全尺寸检查、两轮100-sweep盲AD、一轮500次ABCD组更新诊断，以及两轮100-sweep固定参考探针诊断（后一轮单独保护振幅的正数定义域）。随后按新授权完成原网络固定参考P的500+200更新（rRMSE0.318980、P逐元素不变），以及3档各50-sweep的盲AD学习率检查（0.583094/0.461177/0.452373，仍有结构串扰）。结果与边界见本目录 [RESULTS.md](RESULTS.md)；早期汇总在`results/report_20261019/`，后续合并图表在`results/fixed_probe_and_lr_report_20261019/`。这些是原因排查，不能把未调优的AD失败结果用作方法性能优势的证据，固定参考P也不能当成原方法盲重建成功。

已完成官方论文/源码核查、版本与发布来源锁定，以及本地 GPU 环境检查。上轮的联网审批故障由用户手动安装解除；未绕过审批。现有仿真、`real_ptycho` 和 `real_ptylab_mpie` 训练代码均未为本任务修改。仅调用原适配器的坐标/初值工具，并记录其源码哈希。

## 基线来源与引用

- 官方项目：https://github.com/chiahao3/ptyrad
- 正式论文：Lee, C.-H., Zeltmann, S. E., Yoon, D., Ma, D., & Muller, D. A. (2025). *PtyRAD: A High-Performance and Flexible Ptychographic Reconstruction Framework with Automatic Differentiation*. Microscopy and Microanalysis, 31(4), ozaf070. https://doi.org/10.1093/mam/ozaf070
- 锁定官方 PyPI `ptyrad==1.0.0`，发布日期 2026-06-07，发布提交 `1ee8e51ca87cbcbe84fd5ace021315fbd92e05f5`。来源：https://pypi.org/project/ptyrad/ 。实际安装后还必须记录安装版本、文件路径及所调用源文件 SHA256。
- 官方 wheel SHA256：`2cf29aa8b1f98aa6e58a460d1cd749fd3a1f60ae74ab574451232508c35c745e`。

```bibtex
@article{lee2025ptyrad,
  title={PtyRAD: A High-Performance and Flexible Ptychographic Reconstruction Framework with Automatic Differentiation},
  author={Lee, Chia-Hao and Zeltmann, Steven E. and Yoon, Dasol and Ma, Desheng and Muller, David A.},
  journal={Microscopy and Microanalysis},
  volume={31}, number={4}, pages={ozaf070}, year={2025},
  doi={10.1093/mam/ozaf070}
}
```

不得把自行实现的像素 AD 标为官方 PtyRAD。本光学入口调用真实官方模型/损失/优化链路，并披露光学数据导入与参数映射；匹配 MSE 的包装单独标注为诊断配置。

## 独立环境与安装记录

已创建本目录 `.venv`，Python 3.10.20。它使用 `--system-site-packages --without-pip` 只读复用 `flatnet` 已安装的 CUDA torch 2.5.1+cu121 / torchvision 0.20.1+cu121；新依赖安装到 venv，不应修改 base/flatnet。它是分层环境，不是所有依赖从零安装的纯隔离环境。普通 ensurepip 在工具沙箱临时目录没有写权限，故不使用 ensurepip。

在你自己的 PowerShell 中执行（不是 Colab）：

```powershell
& "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\.venv\Scripts\python.exe" -m pip install --index-url https://pypi.org/simple "ptyrad==1.0.0" -c "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\constraints-local.txt"
```

此命令只安装依赖，不读取或上传实验 H5，不运行重建。若依赖约束冲突，不应自动升级/覆盖原环境，应保留错误待核查。

安装成功后检查：

```powershell
& "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\.venv\Scripts\python.exe" "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\check_runtime.py"
```

`check_runtime.py` 只验证官方安装、32² 单层远场强度/梯度与官方损失公式，保存软件来源报告；**不是实验重建入口**。本次检查已通过：强度相对误差1.3769e-7，振幅/相位/探针梯度相对误差2.29e-7、4.71e-7、3.22e-7；官方归一化RMSE公式差0。

### Python 3.10兼容加载（必须披露）

官方1.0.0声明Python>=3.10，但主solver入口间接导入`plotting/basic.py`，其中f-string含Python3.12的PEP701语法；本机Python3.10因此SyntaxError。**没有修改安装包或数值实现**。`official_runtime.py`从实际安装的`solver/reconstruction.py`读取完整且未修改的函数AST（create_optimizer、make_batches、recon_step等），在排除绘图导入的命名空间中执行，保留原源文件位置与SHA256。仅使用新建Adam、random分组、单GPU、无scheduler路径；不可声称完整原生CLI已在Python3.10运行。Python>=3.12走正常官方导入路径，此路径本机尚未测试。

实际优化调用官方`PtychoModel`、`CombinedLoss`、`create_optimizer`、`recon_step`、`CombinedConstraint`；我们只写光学输入导入、外层运行预算/保存/公共评估包装。`matched-mse`选项明确替换了损失，因此只称为匹配损失诊断，不称官方默认损失。

## 实验入口及复现命令

以下命令在本机PowerShell运行；首次不传`--prepared`时，会从原H5创建独立的225帧1024²公共归一化缓存。之后可传`--prepared`复用，不能将不同裁剪/种子缓存混用。所有输出新建目录，不覆盖已有结果。

```powershell
& "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\.venv\Scripts\python.exe" "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\run.py" --sweeps 100 --loss official --lr-object 0.0005 --lr-probe 0.005
```

- A：上面命令，官方振幅损失、与当前对照相同学习率。
- B：同上，仅`--lr-probe 0.0001`改为包的默认探针学习率；位置优化等仍关闭，因此不是整个电子显微镜默认配置。
- C：`--sweeps 125 --loss matched-mse --grouping abcd`，500次更新及28125帧访问，匹配S1 ABCD顺序与组MSE；微批末尾按张数加权，梯度测试通过。**PtyRAD只有一个共享物体，网络则有四组条件输出，因此不能称所有因素只差“网络有无”。**
- D：`--sweeps 100 --fixed-probe --reference-probe "C:\Users\kennyzz\Downloads\reconstruction_mpie_100ITER.npz"`，以参考P固定恢复O；只作消除盲探针估计的诊断，不是同圆形初值的盲基线。
- D+：同D，增加`--positive-amplitude-floor 0.000001`。使用官方`obja_thresh`在每次Adam后保护正数振幅定义域；上界1e10未触及。该投影仅用于独立诊断，不是A/B/C的无额外约束基线，不加TGV/NLM。

每个run目录保存`config.json`、`preflight.json`、`initialization.npz`、`batches.json`、公共`history.json`、逐次快照、完整`checkpoint.pt`及`summary.json`。时间只描述本机诊断，不与Colab mPIE时间直接比较。官方random分组`225//4=56`批，实际批大小4或5；每个sweep为56次Adam更新，不是4次ABCD组更新。

小尺寸测试：

```powershell
& "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\.venv\Scripts\python.exe" -m unittest discover -s "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad" -p test_adapter.py -v
```

6项小尺寸测试通过（2.478s测试体）：官方step与同官方模型/损失的直接调用更新逐元素相同；只激活O振幅/O相位/P且额外约束关闭；匹配MSE公式正确；非等长微批累计梯度等于完整组MSE；显式正幅度保护使用官方阈值且不改探针；另有CPU紧支撑随机场反射共轭变换的傅里叶振幅验证。第一次全尺寸接入遗漏了官方`compute_loss_fn`回调，未发生优化，失败目录保留为`preflight_fullsize_20261019`；补齐原官方调用后2-sweep预检和完整运行均正常。

`analyze.py`只复核保存场的公共残差及可选`--dual-case LABEL`反射共轭诊断，不运行优化器。成功mPIE参考变换前后rRMSE为0.303137与0.304431，预测振幅相对差4.9932%；不能称严格同解。对应图和指标在`results/reference_dual_check_20261019_v2/`，含义及边界见`RESULTS.md`§4.1。

## 已核对的接口差异（已安装源码验证）

1. 官方单层、单物体模式、单探针模式的前向退化为出射波的远场 FFT 强度；不需要将可见光波长伪装成电子加速电压。多层内部传播在单层时跳过。光学采样用实际 lambda/z/探测器像元计算后显式注入，不能直接套电子显微镜默认参数。
2. 官方 `PtychoModel` 优化物体振幅和相位、探针复数实虚部，并非神经网络权重。默认位置链路可带亚像素探针平移；本对照需显式固定零平移并注入与 mPIE 对齐的整数切片，不能让默认预处理重排/重新标定位置。
3. 官方前向强度加 `eps=1e-10`；FFT 无输入 ifftshift 的幅度结果与居中 FFT 相同，但必须实际检查强度与梯度，不能只看名称。
4. 官方 `CombinedLoss.get_loss_single` 在 dp_pow=0.5 时使用 `sqrt(mean((sqrt(Ipred)-sqrt(Imeas))^2))/mean(sqrt(Imeas))`，不是当前方法的幅度 MSE，也不是自适应 g。官方配置基线与匹配 MSE 的诊断不可混名；RMSE 在分块/累计反传时不能直接逐块相加冒充完整批次 RMSE。
5. 默认约束、归一化、探针功率校正、位置优化和相位正性均需在真正的安装源码中逐项审计后决定，不能仅关闭 TGV/NLM 就声称“无额外约束”。

## 初始对照计划与停止条件（保留原计划，执行结果另述）

已知观察：此前无特征归一化和用户报告的仅加特征归一化完整运行均出现 O 局部亮斑、P 含 USAF 条纹；mPIE 同数据/同圆形初值能够恢复条纹。候选原因包括网络参数化/分组条件输出、盲 O/P 联合优化的路径、初始化相互作用，以及尚未消除的实现/损失差异。不把候选直接写成结论。

固定输入：原 H5 全部 225 帧，中心裁剪 `[488:1512]` 为 1024²，位置毫米，lambda=632.8 nm，z=6 mm，探测器像元3.76 um；一个全局强度均值，不逐帧曝光归一化。初始 O/P 必须与当前 mPIE 模式逐元素核对；完整1024²探针，不添加512支撑、不使用 mPIE 最终结果作为盲重建初值。

第一轮先做小尺寸官方前向/梯度检查、真实全尺寸少量帧检查，再启动单 seed=0、最多100个全数据 sweep、单进程、最多20分钟的官方 AD 对照；batch 根据6GB显存做小尺寸预检决定并记录，不悄悄改成小于1024的衍射图。每次 sweep 的实际优化器更新次数单独记录，不能等同当前网络一次组更新。任何 OOM、NaN、运行预算到期都保存状态，不自动无穷重试。用户已在本聊天明确授权本地安装和运行；本计划不是自动运行主仿真脚本的许可。

后续诊断候选：同一数据与起点的匹配 MSE 配置，及固定参考探针的物体恢复。后者只用于定位盲分解问题，不能充当原方法从头盲重建成功的结果。官方 AD 成功仅能说明数据在该配置下可解，不能单独区分网络参数化、不同优化器/损失和分组规则的全部影响。

评价：共用振幅 rRMSE、初始/过程中 O/P 幅相图、USAF 条纹及相同物理 ROI 的强度剖面。没有实验真值，不报 PSNR；好看的预览或更低 loss 均不单独证明正确 O/P 分解。所有试验（含失败）保存版本、配置、耗时、更新计数和过程结果。

## 后续固定参考P网络诊断与LR敏感性

`network_fixed_probe.py`在独立进程中调用未修改的`real_ptycho.run.train`，只注入已标定的成功mPIE探针并显式禁止P的Adam步。网络构造、输入、ABCD融合、网络Adam和U32残差均原样执行；P逐次逐元素核对。它提供参考信息，不是盲重建基线，且与旧NPZ还存在整数/双线性算子差异，详见`FIXED_PROBE_PLAN.md`。

本地命令（`--out`必须为新目录；用户已授权本次完整运行）：

```powershell
& "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\.venv\Scripts\python.exe" -u "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\network_fixed_probe.py" --out "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\results\nn_fixed_probe_repeat" --stage1-steps 500 --stage2-steps 200
```

GPU预检保留完整1024²及225帧，只做4+2更新，已通过；不使用会减小衍射图的`--smoke`。完整运行保存来源快照、逐次评估场、两阶段原checkpoint和P不变验证。`test_fixed_probe.py`提供2项额外CPU冻结/原Adam检查。

完整500+200运行已完成，S1/S2公共rRMSE分别0.398634/0.318980，训练320.47/478.63秒，墙钟826.02秒；700次P的更新全部禁用，末态P逐元素不变。USAF主体恢复但外围仍有伪影，不能称整幅物体已高质量恢复。原入口保留的probe Adam文字日志不是实际更新证据，以独立逐次不变检查与`diagnostic_summary.json`为准。全部8项小尺寸测试通过（2.524秒测试体）。

学习率敏感性只跑三档（O/P均共同缩放1、4、10倍），各50 sweep、同一圆形初值、官方损失及1e-6正幅度保护；不自动选择物理解，也不直接当作充分调优的论文排名。已在网络退出后串行执行完毕，三档公共rRMSE为0.583094、0.461177、0.452373，完整O/P仍有明显伪影。复现命令：

```powershell
& "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\.venv\Scripts\python.exe" -u "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\lr_sensitivity.py" --out "C:\Users\kennyzz\Desktop\INNM_Code\real_ptyrad_ad\results\lr_sensitivity_repeat"
```

另需区分“官方模型/更新函数”与“整套官方推荐设置”：此前为对齐初值，未采用官方初始化的探针功率归一化，且关闭了`fix_probe_int`。官方标准模板启用后者，不能把本对照称作整套默认流程。说明见[初始化参数](https://ptyrad.readthedocs.io/en/latest/_autosummary/ptyrad.params.init_params.html)与[标准模板](https://ptyrad.readthedocs.io/en/latest/templates/standard.html)（2026-10-19核对）。这里未自动打开这些设置；电子样品模板的近1振幅约束也不能直接套用强吸收USAF。

## 研究流程辅助说明（不是 AD 基线引用）

本对照设计使用了 Scientific Critical Thinking 与 Hypothesis Generation 程序性技能来区分候选机制、可检验预测与实际证据。其出处仅作分析流程说明，不是 PtyRAD 算法来源，也不要求替代论文的 AD 引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 。作者及当前记录已于2026-10-19通过 arXiv 核对。
