# Colab：共享前程与表示/TGV分支

## 后续单支 E：将后半程 TGV 降为0.01

同步新的 `simulations/ProPtyNet_paper.py`、`functions/paperrepro/branching.py`，
并保留此前已修复的 `functions/paperrepro/report.py`。
E与A/B一样重置优化器，继承原共享检查点的网络、探针、TGV辅助变量与学习率，
仅将tgv_amp由0.1降为0.01。不重新跑前1000步，不从A/B最终状态开始。
已对本轮旧runner哈希建立显式兼容，其他模型/物理/TGV源码哈希仍严格检查，不改写旧.pt。

```python
!python -u /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net \
    --resume /content/branch40_pixel/prefix/shared_step1000.pt \
    --branch E --iters 1000 --device cuda \
    --outdir /content/branch40_pixel/E
```

不要另外传 --tgv-amp。启动日志必须显示E、net、start=1000 + 1000、TGV=0.01、probe_mode=pixel。
E目录必须新建/为空。完成后发送E/net_result.npz和E/branch_metadata.json即可。
原来的汇总脚本仍只收集continue/A/B/C/D，不会自动包含E；这次不用重跑旧汇总命令。

已实现；完整仿真只由用户手动在 Colab 启动。旧命令不带 checkpoint/resume 时不改变原来的求解路径。
先把以下文件同步到 Colab 对应仓库（只上传主脚本是不够的）：

- simulations/ProPtyNet_paper.py
- functions/paperrepro/solvers_addip.py
- functions/paperrepro/branching.py（新增）
- simulations/collect_paper_branches.py（新增）

没有自动提交或推送代码；若用git更新Colab，应先自行提交并推送这些修改。

### 旧版本在最后画图时报 KeyError: 'real'

这是报表接口缺失字段，不是训练失败。修复只涉及
`functions/paperrepro/report.py`：branch历史绘制实际的data_loss，不冒充旧的real误差。
请同步这个文件到Colab。求解器源码哈希未改变，旧检查点仍可直接续跑。
报错前已经依次写入.pt、branch_metadata.json和结果.npz；先用下面代码核验.pt，
通过后直接执行第二格，不重跑第一格，也不要覆盖/修改检查点：

```python
from pathlib import Path
import torch
p = Path('/content/branch40_pixel/prefix/shared_step1000.pt')
assert p.is_file(), '找不到检查点：先确认路径和Colab运行时是否还在'
s = torch.load(p, map_location='cpu', weights_only=True)
assert s['format'] == 'paper-branch-v1' and s['step'] == 1000
assert s['network'] is not None and s['optimizer_object'] is not None
print('可续跑：step =', s['step'], 'probe_mode =', s['cfg']['probe_mode'])
del s
```

如果某个分支也已经跑完、只在画图时报相同错误，不要重复运行该分支。
它的NPZ和元数据通常已存在，可供第三格汇总；新的分支使用修复后的report.py。

## 第一格：共享前1000步

不加硬support。disk只是初始形状，训练中探针全画布可变化。
注意：disk初始化和现有TGV作用域仍使用名义probe直径，不应称为完全没有几何先验。
这是固定40% pilot，不保证无support下达到以前有support的质量。

```python
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net \
    --preset paper --obj-size 624 --grid 4 --step-px 35 \
    --eval-size 96 --eval-every 25 --seed 0 --device cuda \
    --probe-mode pixel --probe-init disk \
    --lr-net 0.005 --lr-probe 0.02 --lr-obj 0.03 \
    --tgv-amp 0.1 --tgv-phase 0 --iters 1000 \
    --outdir /content/branch40_pixel/prefix \
    --checkpoint-out /content/branch40_pixel/prefix/shared_step1000.pt
```

完成后必须看到 saved ...shared_step1000.pt。若前程恢复较差，仍可研究“从该中途状态继续的行为”，
但不能将其解释成已经成功恢复后的维持能力。不要按GT挑一个最好时刻再分支。

## 第二格：独立续跑五支

下面是Colab **Python代码格**，不是shell。前缀成功后执行。
五支都从同一个检查点读取，顺序运行以避免抢GPU内存。异常时立即停止。

```python
import subprocess
import sys
from pathlib import Path

script = '/content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py'
root = Path('/content/branch40_pixel')
checkpoint = root / 'prefix/shared_step1000.pt'
assert checkpoint.is_file(), '先完成第一格'

for branch in ['continue', 'A', 'B', 'C', 'D']:
    command = [sys.executable, script, 'net',
               '--resume', str(checkpoint), '--branch', branch,
               '--iters', '1000', '--device', 'cuda',
               '--outdir', str(root / branch)]
    print('Running:', branch, flush=True)
    subprocess.run(command, check=True)
```

continue：保留网络、TGV及三个Adam状态的正常续跑控制。
A：网络＋TGV；B：网络、不加TGV；C：像素＋TGV；D：像素、不加TGV。
A/B/C/D均重置优化器，但不重置物体/探针；A/C继承同一TGV辅助向量，重置其Adam。
C/D物体用 lr_obj=.03；所有探针仍用 lr_probe=.02，不悄悄切成旧AD的lr_prb。
继承probe_mode，不加入support，也不冻结probe。

续跑的 --iters 是**追加步数**；1000→2000，而非重跑0→1000。
不要再传 --preset、--step-px、--probe-mode、--tgv-amp 等：检查点继承这些设置，B/D自动关TGV。
只有C/D可以显式覆盖 --lr-obj；默认继承前程保存的 .03。本pilot不做LR扫描。
数据loss/复场一致性校验失败会报错停下，不能忽略继续。

输出目录必须是新的/空的；原检查点不覆盖。若某支失败，先检查错误；续跑其它支可只循环剩余名称。
不要删除已有结果来绕过拒绝覆盖。重新整轮请换root，并同时改第一格的两个输出路径。
检查点含测量、网络输入和优化器，可能有数百MB；保留共享.pt，并建议及时复制到Drive。
只从自己生成的检查点续跑；使用PyTorch weights_only加载，不支持历史NPZ恢复网络。
同一轮不要改求解器源码，否则源码哈希检查会拒绝混用；跨GPU/库版本不保证逐位重现。

## 第三格：汇总

```python
!python /content/ptychography_AD_DIP/simulations/collect_paper_branches.py --root /content/branch40_pixel
```

产生 summary.csv、branch_curves.png。五支缺失或不是同一检查点会报错，避免混合结果。
CSV末200步均值只是已记录时刻的均值，并非每步都评价。
时间区分优化部分、含评价的分支耗时及加共享前程的耗时；不含初始化/文件保存/画图，不用于直接宣称加速。

给我看 summary.csv 和 branch_curves.png；需要重新核对数组时，再发A/B的net_result.npz、C/D的ad_result.npz和元数据。
完整.pt留在Colab/Drive即可，不必全部下载发送。

## 实现边界

- 新路径采用更新后的同步物体/探针计算指标，旧路径保存时刻未改。因此与旧日志可能有一步差异。
- 评价额外前向保留训练模式，但恢复BN内部buffer和RNG，不改变下一步的训练状态。
- 不支持measurement schedule、cosine学习率和phase TGV；这些会显式报错，不静默忽略。
- 新路径固定cudnn deterministic、关闭benchmark；轻微耗时/数值差异不能误读成科学发现。
- 若某支更新幅度异常或发散，先看branch_metadata.json与history中的相对更新量，不把它直接解释成组件必需。
- 当前只支持退出到像素，不支持像素再拟合回U-Net。continue也可以恢复像素分支自身的checkpoint。

验证：16×16合成数据CPU单元测试检查连续4步与2+2步续跑精确一致（物体、探针、网络状态、TGV辅助变量），
以及A/B/C/D起点一致、禁止覆盖、禁止场景覆盖和不支持调度的拒绝。
没有本地运行正式paper仿真，也未验证完整GPU科学结果。
