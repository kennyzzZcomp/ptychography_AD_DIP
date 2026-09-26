# Complex backbone experiment (`net` only)

Opt-in complex U-Net backbone with **real amplitude/cos-sin readout** and the
existing independently optimized pixel probe. Default `--network-type real`
still uses the previous implementation. No extra package is required.

## Implemented design

- Input is the existing normalized diffraction-intensity stack plus zero
  imaginary part. Latent complex feature phase is not measured physical phase.
- Complex tensors are pairs of real float tensors. Each convolution uses the
  tied block weights `[Wr, -Wi; Wi, Wr]`, four real convolutions and one complex
  bias. These weights are not restricted to unitary matrices.
- Complex Glorot: independent zero-mean Gaussian real/imaginary weights, each
  with variance `1/(fan_in+fan_out)` including the kernel area.
- Each double block uses complex conv -> covariance complex BN -> activation,
  twice. BN computes a regularized analytic 2x2 inverse square root per channel
  across B,H,W. Epsilon is 1e-5; EMA uses population moments and momentum 0.1.
  The learned symmetric affine matrix has diagonal initialization 1/sqrt(2).
  It is not constrained positive definite and does not enforce phase equivariance.
- Default modReLU has a per-channel bias initially -0.01. A smooth radius
  `sqrt(real**2 + imag**2 + 1e-12)` regularizes the origin. The bias is unconstrained;
  positive bias can still produce large near-origin gradients. `crelu` applies
  separate real ReLUs as an optional activation control.
- Pooling chooses the largest modulus and gathers both components at the same
  location. Complex transposed conv keeps the original kernel=3, stride=2,
  padding=1, output_padding=1. This does not fix checkerboard artifacts by design.
- Skips concatenate real-real and imaginary-imaginary components. The final
  real readout concatenates both feature components and retains the original
  softplus amplitude / normalized cos-sin field construction and neutral O=1
  initialization. The solver initializes the output heads just as for real net.
- Probe, data loss, propagation, optimizer and evaluation definitions remain
  unchanged. This is a whole-backbone comparison, not a conv-only ablation.

## Colab pilot

After syncing the changed files, run to a new directory:

```python
!python /content/ptychography_AD_DIP/simulations/ProPtyNet_paper.py net \
  --preset paper --obj-size 624 --grid 10 --step-px 12 \
  --iters 2000 --eval-size 96 --eval-every 25 --seed 0 --device cuda \
  --network-type complex --complex-base-ch 23 --complex-activation modrelu \
  --skip-mode concat --probe-mode pixel --probe-init ones \
  --lr-net 0.001 --lr-probe 0.01 --tgv-amp 0 --tgv-phase 0 \
  --outdir /content/ov80_complex23_modrelu_seed0
```

For the baseline, replace the network options with `--network-type real --base-ch 32`
and change the output directory; leave measurements and all other options fixed.
The complex branch uses `--complex-base-ch`, not `--base-ch`.

For 100 input patterns (actual real-valued scalar parameter counts):

| Backbone | Network parameters |
|---|---:|
| Real, base 32 | 2,172,227 |
| Complex, base 22, modReLU | 2,070,027 |
| Complex, base 23, modReLU | 2,260,167 |
| Complex, base 32, modReLU | 4,347,267 |

Base 23 is an approximate parameter match (+4.05%), not an exact equal-capacity
or equal-compute comparison. Counts exclude the probe. Four real convolutions
and covariance BN can slow training even when parameter counts are similar.
Do not claim an acceleration without GPU timing or compare only best seeds.

## Limits, output, and checks

- `run`/`ad` reject `--network-type complex`.
- Complex + wavelet (including identity mode), measurement schedules, and
  selected-input progressive are deliberately rejected, not silently ignored.
- TGV remains independently controlled; the first pilot should keep it off.
- Config is saved in `net_result.npz`; convergence plots and training-loop
  timing are unchanged. `network_metadata.json` additionally records exact
  network parameter count, channel type, activation and output parameterization.
- The real-network memory estimate is suppressed for complex mode because it
  does not cover complex activation/whitening buffers. Measure memory on the GPU.
- CPU tests cover native-complex conv/deconv forward and gradients, initialization,
  modReLU near zero, BN whitening/degenerate inputs/running state, magnitude
  pooling, real-head neutral initialization, physics backpropagation, optimizer
  updates, state-dict round trips and invalid combinations.

```text
python -m unittest discover -s tests -p test_complex_network.py
```

Reference: Trabelsi et al., *Deep Complex Networks*, ICLR 2018,
https://arxiv.org/abs/1705.09792. This implementation is a task-specific
experimental adaptation, not the authors' original code or an efficacy claim.
