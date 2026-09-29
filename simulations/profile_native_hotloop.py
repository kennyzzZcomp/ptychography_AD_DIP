"""Bounded native-order profiling: 30 warmup + 10 recorded updates."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    base = Path(sys.argv[1]).resolve()
    directory = base/'round18_native_profile'
    directory.mkdir(exist_ok=True)
    if (directory/'training').exists():
        raise FileExistsError('Preserve existing training output')
    from simulations.run_raster_backend_control import configure_api
    configure_api('v8')
    import torch
    from functions.paperrepro import solvers_addip as solver
    from simulations.run_net_raster_acceleration import run_with_raster
    old = json.loads((base/'round15_native_seeds34_1000/protocol.json').read_text())
    command = old['trials'][0]['command']
    arguments = command[command.index('net'):].copy()
    arguments[arguments.index('--audit-stop-after')+1] = '41'
    arguments[arguments.index('--outdir')+1] = str(directory/'training')
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False,
        with_stack=False)
    class Timer:
        def __init__(self, *args):
            self.scope = None
            self.active = False
        def start(self, iteration):
            if iteration == 30:
                torch.cuda.synchronize()
                profiler.__enter__()
                self.active = True
            if iteration == 40:
                torch.cuda.synchronize()
                profiler.__exit__(None, None, None)
                self.active = False
            if self.active:
                self.scope = torch.profiler.record_function('stage:scheduler')
                self.scope.__enter__()
        def mark(self, stage):
            if not self.active:
                return
            self.scope.__exit__(None, None, None)
            following = {'scheduler':'network_decode', 'network_decode':'physics_loss',
                'physics_and_data_loss':'zero_grad', 'zero_grad':'backward',
                'outer_backward':'optimizer', 'optimizer':'loop_tail'}
            self.scope = torch.profiler.record_function('stage:'+following.get(stage, stage))
            self.scope.__enter__()
        def finish(self):
            if self.active:
                self.scope.__exit__(None, None, None)
                self.scope = None
                profiler.step()
        def save(self, *args):
            pass
    with patch.object(solver, 'StageTimer', Timer):
        run_with_raster(arguments, 'unfold', adjoint_precision='native-order')
    profiler.export_chrome_trace(str(directory/'trace.json'))
    (directory/'operators.txt').write_text(profiler.key_averages().table(
        sort_by='self_cuda_time_total', row_limit=100), encoding='utf-8')
    rows = []
    for e in profiler.key_averages():
        rows.append({'name': e.key, 'count': e.count,
          'cpu_us': e.cpu_time_total, 'self_cpu_us': e.self_cpu_time_total,
          'device_us': e.device_time_total, 'self_device_us': e.self_device_time_total})
    (directory/'operators.json').write_text(json.dumps(rows, indent=2))
    (directory/'protocol.json').write_text(json.dumps({'arguments': arguments,
        'warmup_updates':30,'profiled_updates':[31,40], 'record_shapes':False,
        'profile_memory':False, 'with_stack':False,
        'warning':'Profiler-instrumented diagnostic, not production throughput'}, indent=2))
    print(profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=25))

if __name__ == '__main__':
    main()
