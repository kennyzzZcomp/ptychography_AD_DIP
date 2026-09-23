# Continuous amplitude TGV schedule (Colab)

Update the repository files in Colab before running. New runtime file:
`functions/paperrepro/tgv_schedule.py`; changed entry point and solver:
`simulations/ProPtyNet_paper.py`, `functions/paperrepro/solvers_addip.py`.
Also sync `functions/paperrepro/branching.py` for reviewed old-checkpoint compatibility.

Add this option to the otherwise unchanged normal `net` command:

```text
--tgv-amp 0.1 --tgv-amp-schedule "1000:0.01"
```

Updates 1–1000 use 0.1; updates 1001 onward use 0.01. Network, probe,
Adam states, and TGV auxiliary states stay continuous. Learning rates are
not changed by this option. No schedule means the original constant weight.
This is not branch E (which resets Adam), and cannot be combined with
`--resume` or `--checkpoint-out`. It currently applies only to normal `net`.

For the latest screenshot's roughly 60% configuration (step 24, not 35):

```python
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net \
  --preset paper --obj-size 624 --iters 2000 --eval-size 96 \
  --step-px 24 --grid 4 --probe-mode pixel --probe-init ones \
  --tgv-amp 0.1 --tgv-amp-schedule "1000:0.01" \
  --outdir ov60_net_tgv_decay1000
```

Keep the baseline's learning rates, seed, probe initialization, and geometry
unchanged. If your baseline explicitly specified learning rates, copy those
options too. Choose a fresh output folder. The screenshot used an `ov40`
folder name but its actual step was 24 (~59.4% overlap).

The saved NPZ history contains `tgv_amp_weight` at each evaluation;
`tgv_amp_schedule.json` records the weight of every update. Multiple boundaries
are supported, e.g. `"1000:0.01,1500:0.001"`, but start with one boundary.
Weights may be zero; auxiliary updates still run to preserve continuity, so
zero does not remove TGV computation overhead. Compare reconstruction metrics,
not total loss across the boundary (its definition changes).
