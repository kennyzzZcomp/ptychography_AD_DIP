"""Package only aligned entrypoints, official SIREN, evaluator and the shared scene."""
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
pty=root.parent/'PtyINR-main'
out=root/'results_paper/ptyinr_alignment'
out.mkdir(parents=True,exist_ok=True)
archive=out/'ptyinr_aligned_bundle.zip'
files={
    'PtyINR-main/main_aligned.py':pty/'main_aligned.py',
    'PtyINR-main/check_aligned_scene.py':pty/'check_aligned_scene.py',
    'PtyINR-main/test_aligned.py':pty/'test_aligned.py',
    'PtyINR-main/PtyINR/siren.py':pty/'PtyINR/siren.py',
    'PtyINR-main/official_README.md':pty/'README.md',
    'baseline/functions/common/metrics.py':root/'functions/common/metrics.py',
    'baseline/functions/paperrepro/evaluate.py':root/'functions/paperrepro/evaluate.py',
    'scene.h5':out/'scene.h5',
}
cells=[]
def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
def code(text):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(True)))
md('''# PtyINR：对齐 ProPtyNet 仿真
先在 Colab 选择 GPU。把 `ptyinr_aligned_bundle.zip` 上传到 Google Drive 的 `MyDrive/PtyINR_aligned/`。
包内是已导出的固定场景，不在 Colab 重新生成测量。包含官方 SIREN 和适配入口，不包含编译后的 tiny-cuda-nn。
已知探针实验无需 tiny-cuda-nn。盲重建请复用你跑通过官方 simulation 的运行环境和作者修改的 float 版本。
不要为了运行这个 notebook 升级或重装已经正常工作的 PyTorch。
''')
code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import zipfile
bundle = Path('/content/drive/MyDrive/PtyINR_aligned/ptyinr_aligned_bundle.zip')
work = Path('/content/ptyinr_aligned')
with zipfile.ZipFile(bundle) as z:
    z.extractall(work)
print(work)
''')
code('''import torch
assert torch.cuda.is_available(), '请选择 GPU 运行时'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
import numpy, scipy, h5py, matplotlib
# 如果缺这些基础包，再单独运行：%pip install numpy scipy h5py matplotlib
''')
code('''%cd /content/ptyinr_aligned/PtyINR-main
!python check_aligned_scene.py --scene ../scene.h5 --device cuda
!python -m unittest test_aligned
''')
md('''## 先跑已知探针诊断
运行 100 次全扫描 Adam 更新，官方物体 SIREN、默认 SmoothL1 和学习率。
这是使用真值探针的非盲诊断，不能当成正式盲重建 baseline。
读取本地 /content 数据；每 10 步将结果和 checkpoint 写到 Drive，防止会话结束丢失。
''')
code('''from datetime import datetime
run_root = Path('/content/drive/MyDrive/PtyINR_aligned/results')
known_out = run_root / ('known_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
import subprocess, sys
subprocess.run([sys.executable, 'main_aligned.py', '--scene', '../scene.h5',
    '--baseline-root', '../baseline', '--probe-mode', 'known', '--steps', '100',
    '--eval-every', '10', '--outdir', str(known_out)], check=True)
''')
code('''from IPython.display import display, Image
display(Image(filename=str(known_out / 'reconstruction.png')))
import json
print(json.loads((known_out / 'history.json').read_text())[-1])
''')
md('''## 盲重建：准备好作者版本 tiny-cuda-nn 后再执行
如果你此前跑通的 runtime 还在，无需重新安装。若环境已重置，先运行你原先成功的作者版本安装单元。
官方仓库的安装路径为 `PtyINR/tiny-cuda-nn/bindings/torch`，不是直接安装上游默认版本。
本适配仍保留官方 HashGrid 配置、[-1,1] 探针坐标、1e4 输出缩放、峰值归一化、重居中及前 50 步探针正则。
探针随机网络初始化与其他方法的常数初始化不相同；manifest 已披露，正式实验需报告。
这一单元先做 100 步接入诊断，不保证已经达到收敛。两个运行相互独立，不继承真值探针结果。
''')
code('''import tinycudann as tcnn
print('tinycudann location:', tcnn.__file__)
blind_out = run_root / ('blind_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
subprocess.run([sys.executable, 'main_aligned.py', '--scene', '../scene.h5',
    '--baseline-root', '../baseline', '--probe-mode', 'blind', '--steps', '100',
    '--eval-every', '10', '--outdir', str(blind_out)], check=True)
''')
md('''## 实现范围与输出
- 此入口复用官方 `Siren` 类；HashGrid 配置与输出变换按本地官方 train.py 保留。
- 更改了测量输入、传播、采样尺寸、单卡调度和评价；不再调用官方 simulation，不额外 fftshift、不除以 600。
- 物体/探针坐标按官方两种顺序保留。每次更新累计全部扫描；坐标和扫描分块只用于降低显存。
- `history.json` 和 `result.npz` 都是更新后的结果；同时保存 `manifest.json`、`checkpoint.pt`、最终 `reconstruction.png`。
- checkpoint 保存模型及优化器，当前入口尚不支持 resume。
- 耗时包含评价与存盘，尤其写 Drive 的时间，不能直接用于论文速度排名。
- 单步本地 CUDA 测试和微型梯度等价测试不代表盲重建已验证或结果已收敛。
- 要换成另一组仿真，先在 INNM_Code 用 export_ptyinr_scene.py 重新导出同一份数据，再重新打包。
''')
for i,cell in enumerate(cells):cell['id']=f'aligned-{i:02d}'
notebook=dict(cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
nb=out/'PtyINR_aligned_colab.ipynb'
nb.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for dest,src in files.items():z.write(src,dest)
    z.write(nb,nb.name)
print(archive,archive.stat().st_size)
print(nb)
