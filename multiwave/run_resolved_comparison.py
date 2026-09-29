"""Background CPU experiment suite with a continuously updated Markdown report.

python -m multiwave.run_resolved_comparison --outdir multiwave/results/resolved_ad_unet_20260929
Rerunning resumes by skipping completed trials; partial trials receive a new
attempt directory, preserving every failed/partial result. No training resume
is claimed: a partial trial restarts from its documented initial state.
"""
import argparse
from dataclasses import replace, asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from .config import Config
from .run_simulation import run_experiment


def atomic_text(path, text):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text,encoding='utf-8')
    temp.replace(path)


def render(state, path):
    lines = ['# 扩大探测器后的 U-Net / AD 对比', '',
             f"更新时间：{datetime.now().astimezone().isoformat()}；状态：**{state['status']}**；进程 PID：{state['pid']}。", '',
             '物体 384×384、照明探针 192×192、探测器 768×768、1 μm 像素、1.5 mm 传播距离、515/633 nm。FFT 为 1536×1536；无噪声；25 个扫描位置中 20 个训练、5 个留出。', '',
             '所有主对照使用相同场景／测量／网络种子（17/23/31），共享零相位物体。AD 指直接优化物体像素参数，U-Net 指优化网络权重；两者均通过自动微分优化同一物理损失。', '',
             '主对照 AD-softplus 与 U-Net-softplus 使用相同非负振幅参数化、相同常数振幅初值；U-Net 宽度为 16/32/64/128、原 ProPtyUNet 主干、无相位头。网络固定输入可降采样到物体尺寸，但前向和损失始终保留 768×768 全部测量像素。', '',
             'AD 学习率 0.03；U-Net 0.002；盲探针 0.01。主对照各 500 次更新；额外 AD-direct [0,1] 投影基线为 300 次，仅作为几何校验，不用于等预算优劣宣称。盲探针均从相同的平滑圆斑和零相位开始，各波长探针功率固定为 0.5。', '',
             '| 实验 | 状态 | 更新 | ROI 振幅相对误差 | 中央 RMSE | 高频误差 | 留出振幅误差 | 平均探针误差 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for trial in state['trials']:
        m=trial.get('metrics',{})
        def fmt(key): return f"{m[key]:.6g}" if key in m else '—'
        probes=m.get('probes',[])
        pe=f"{sum(p['complex_relative_error'] for p in probes)/len(probes):.6g}" if probes else '—'
        lines.append(f"| {trial['name']} | {trial['status']} | {trial.get('iteration',0)}/{trial['steps']} | {fmt('shared_amplitude_relative_error')} | {fmt('center_amplitude_rmse')} | {fmt('high_frequency_relative_error')} | {fmt('holdout_clean_amplitude_nrmse')} | {pe} |")
    lines += ['', '评价区域沿用名义照明 ROI。中央与高频指标定义见 RESOLUTION_DIAGNOSIS.md。记录最后迭代，不按真值误差选择最佳模型。留出图不参与训练或网络输入。单种子、无噪声结果只支持本次仿真判断，不代表统计优势或真实实验性能。', '', '## 结果与解释', '']
    completed = {t['name']:t for t in state['trials'] if t['status']=='completed'}
    for mode in ('known','blind'):
        ad,unet=completed.get(mode+'_ad_softplus'),completed.get(mode+'_unet_softplus')
        if ad and unet:
            a,b=ad['metrics'],unet['metrics']
            lines.append(f"- {mode} 主对照：AD ROI 误差 {a['shared_amplitude_relative_error']:.6g}，U-Net {b['shared_amplitude_relative_error']:.6g}；高频误差分别 {a['high_frequency_relative_error']:.6g} / {b['high_frequency_relative_error']:.6g}。只描述本次固定预算结果，不将方法差异全部归因于网络结构。")
        else:
            lines.append(f'- {mode} 主对照尚未全部完成，暂不下结论。')
    lines += ['', '## 结果文件', '']
    for t in state['trials']:
        if t.get('directory'):
            folder=Path(t['directory']).resolve().as_posix()
            lines.append(f"- {t['name']}：[{folder}]({folder})")
            if t['status']=='completed':
                lines.append(f"\n![{t['name']} detail]({folder}/{t['method']}_detail.png)\n")
        if t.get('error'):
            lines += [f"\n{t['name']} 失败记录：",'```text',t['error'],'```']
    atomic_text(path,'\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    root=args.outdir.resolve();root.mkdir(parents=True,exist_ok=True)
    status_path=root/'status.json'
    report=root/'EXPERIMENT_REPORT.md'
    specs=[('known_ad_direct','known','pixel_shared_amp','direct',300),
           ('known_ad_softplus','known','pixel_shared_amp','softplus',500),
           ('known_unet_softplus','known','unet_shared_amp','softplus',500),
           ('blind_ad_softplus','pixel','pixel_shared_amp','softplus',500),
           ('blind_unet_softplus','pixel','unet_shared_amp','softplus',500)]
    if status_path.exists():
        state=json.loads(status_path.read_text(encoding='utf-8'))
    else:
        state={'started':datetime.now().astimezone().isoformat(),'trials':[
            dict(name=n,probe_mode=p,method=m,parameterization=a,steps=k,status='pending') for n,p,m,a,k in specs]}
    state.update(pid=os.getpid(),status='running')
    sources=[Path(__file__)]+[Path(__file__).parent/n for n in ('config.py','physics.py','scene.py','models.py','reconstruct.py')]
    state['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

    def save():
        state['updated']=datetime.now().astimezone().isoformat()
        atomic_text(status_path,json.dumps(state,indent=2,allow_nan=False))
        render(state,report)
        render(state,Path(__file__).parent/'AD_UNET_COMPARISON.md')
    save()
    for trial in state['trials']:
        if trial['status']=='completed': continue
        folder=root/trial['name']
        attempt=1
        while folder.exists() and any(folder.iterdir()):
            attempt+=1;folder=root/f"{trial['name']}_attempt{attempt}"
        cfg=replace(Config.preset('resolved'),probe_mode=trial['probe_mode'],
                    pixel_parameterization=trial['parameterization'],unet_activation='softplus',
                    iterations=trial['steps'],eval_every=25,device='cpu',threads=2)
        trial.update(status='running',directory=str(folder),iteration=0,config=asdict(cfg))
        trial.pop('error',None);trial.pop('metrics',None)
        save();started=time.perf_counter()
        def progress(method,iteration,metrics):
            trial.update(iteration=iteration,metrics=metrics,elapsed_seconds=time.perf_counter()-started)
            save()
        try:
            run_experiment(cfg,[trial['method']],folder,progress_callback=progress)
            trial['status']='completed'
        except Exception:
            trial.update(status='failed',error=traceback.format_exc())
            print(trial['error'],flush=True)
        save()
    state['status']='completed' if all(t['status']=='completed' for t in state['trials']) else 'needs_attention'
    save()
    print(f"SUITE {state['status']}: {report}",flush=True)


if __name__=='__main__': main()
