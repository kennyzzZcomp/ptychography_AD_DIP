"""Preview optional target geometry, without running a reconstruction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulations.ProPtyNet_paper import Cfg
from functions.paperrepro.sample import make_truth, _object_map


def main():
    cfg = Cfg(obj_size=624, grid=10, step_px=12, phs_image='siemens-sharp',
              obj_amp_binary_invert=True, probe_amp_image='simple-stripes',
              probe_phase_mode='same-texture')
    obj, probe, _, _ = make_truth(cfg)
    # Plot prescribed phase, not angle(obj): phase is undefined where amplitude=0.
    phase = (_object_map(cfg, cfg.phs_image, cfg.obj_size) / 255 * 2 - 1) * cfg.obj_phase_rad
    radius = int(np.ceil(cfg.probe_diam_px / 2)) + 6
    c = cfg.N // 2
    p = probe[c-radius:c+radius+1, c-radius:c+radius+1]
    fields = [abs(obj), phase, abs(p), np.angle(p)]
    titles = ['Object amplitude (inverted binary USAF)', 'Prescribed phase (no gray center)',
              'Probe amplitude (simple vertical bands)', 'Probe phase (same texture)']
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, field, title in zip(axes, fields, titles):
        ax.imshow(field, cmap='gray', interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    dest = Path(__file__).resolve().parents[1] / 'output' / 'sharp_truth_stripes_preview.png'
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    print(dest)


if __name__ == '__main__':
    main()
