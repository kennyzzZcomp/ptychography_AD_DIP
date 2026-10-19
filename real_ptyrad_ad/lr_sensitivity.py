"""Bounded, explicitly exploratory three-rate OFFICIAL PtyRAD sensitivity check.

All three blind runs share input, circular P0, object O0, losses, batches, seed,
budget and the SAME positive-amplitude domain guard. Only the two nominal LRs
are co-scaled; this does not isolate object LR from probe LR individually.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=HERE/'results/prepared_20261019_182658_419540')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-total-seconds',type=float,default=1200.)
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    candidates = [('scale1',.0005,.005),('scale4',.002,.02),('scale10',.005,.05)]
    plan = dict(created=datetime.now().astimezone().isoformat(),exploratory=True,
        candidates=[dict(name=n,lr_object=o,lr_probe=q) for n,o,q in candidates],
        common=dict(sweeps=50,batch=4,seed=0,loss='official',grouping='random',
                    positive_amplitude_floor=1e-6,gain=1,TGV=0,NLM=0),
        caution='Both LRs co-scaled. Not independent identification of which LR matters, not a tuned benchmark.',
        previous_evidence='Prior 100-sweep P LR 0.0001 was worse than 0.005; no high-rate results seen before this plan.')
    (args.out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    rows = []
    started = time.perf_counter()
    for name,obj_lr,probe_lr in candidates:
        if time.perf_counter()-started>args.max_total_seconds:
            rows.append(dict(name=name,status='not_started_total_budget'))
            break
        folder = args.out/name
        cmd = [sys.executable,'-u',str(HERE/'run.py'),'--prepared',str(args.prepared),
               '--out',str(folder),'--sweeps','50','--batch','4','--eval-every','10',
               '--lr-object',str(obj_lr),'--lr-probe',str(probe_lr),
               '--positive-amplitude-floor','0.000001','--max-seconds','600']
        print('[LR CASE]',name,'LR O/P',obj_lr,probe_lr,flush=True)
        result = subprocess.run(cmd,check=False)
        row = dict(name=name,command=cmd,exit_code=result.returncode)
        if (folder/'summary.json').is_file():
            row['summary'] = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if (folder/'failure.json').is_file():
            row['failure'] = json.loads((folder/'failure.json').read_text(encoding='utf-8'))
        rows.append(row)
        (args.out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('LR_OUTPUT',args.out.resolve(),flush=True)


if __name__=='__main__':
    main()
