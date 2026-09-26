# Optional Haar-wavelet skip filtering (`net` only)

Default `--skip-mode concat` preserves the original network and state-dict keys.
`run` and `ad` do not support this new option and reject non-default skip modes.

At each of the three encoder skips: one-level orthonormal 2D Haar DWT,
unchanged LL, independent soft-thresholding of LH/HL/HH, IDWT, then the original
concatenation with decoder features. Decoder transposed convolutions are unchanged.
Odd spatial sizes are replicate-padded and cropped after IDWT. There is no new dependency.

Options:

- `--skip-mode wavelet`: 9 learned thresholds total (3 bands x 3 skips), shared over channels.
- `--wavelet-threshold 0.01`: positive initial threshold in feature units, not a photon-noise level.
  Softplus keeps thresholds nonnegative; Adam optimizes them with the network learning rate.
- `--skip-mode wavelet-identity`: DWT/IDWT without shrinkage, for numerical equivalence testing.
- TGV remains independently controlled by `--tgv-amp`; wavelet mode does NOT enable it.

Example (Colab):

```python
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net --preset paper --obj-size 624 --grid 10 --step-px 12 --iters 2000 --eval-size 96 --eval-every 25 --base-ch 32 --seed 0 --probe-mode pixel --probe-init ones --lr-net 0.001 --lr-probe 0.01 --skip-mode wavelet --wavelet-threshold 0.01 --tgv-amp 0 --outdir /content/ov80_wavelet_seed0
```

For a controlled comparison, keep every other setting and measurement fixed:

| Variant | Options |
|---|---|
| Plain U-Net + pixel probe | `--skip-mode concat --tgv-amp 0` |
| U-Net + pixel probe + TGV | `--skip-mode concat --tgv-amp 0.001` |
| Wavelet-filtered skip | `--skip-mode wavelet --wavelet-threshold 0.01 --tgv-amp 0` |

Use a separate output directory for every run. Do not compare different probe
initializations or learning-rate schedules as a module-only ablation.

Existing convergence plots, NPZ config, and training-loop timing remain enabled.
Wavelet work is included in training time. `wavelet_skip.json` records thresholds,
input-zero fractions and post-shrinkage zero fractions from the final training
forward (before the final optimizer update), ordered shallow to deep. Statistics
are captured only on that final forward and add a small diagnostic overhead.
Learned thresholds need not track the noise level and can converge toward zero.

This is an experimental feature prior, not guaranteed denoising or a fix for
decoder-generated checkerboard artifacts. Noise-free experiments measure
reconstruction artifacts, not suppression of measurement noise.

Tests: `python -m unittest discover -s tests -p test_wavelet_skip.py`
