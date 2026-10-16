# INNM simulations

This folder contains the simulation notebooks. Training results stay in memory and are shown inside the notebook; running a notebook does not create `res_*`, MAT metric files, checkpoints, or reconstructed-image files.

## Files

### 64 / 128 FFT controlled comparison (2026-10-13)

Use `run_sampling128.py` as the single entrypoint. Existing 64 defaults are not
replaced. In this mode `N=64` denotes the probe/patch size and `fft_size=128`
denotes the physical-forward FFT and measured detector-array size. The exit wave
is center-zero-padded BEFORE FFT; measured intensities are never resized.

- Same saved 612×612 object, 64×64 probe (effective diameter 59 px), 10×10 scan,
  step 10 px, quadratic phase, evaluation ROI and constant-one probe initialization.
- Same object-plane pixel pitch, wavelength and propagation distance. Detector
  pitch halves from 15.04 to 7.52 µm, retaining detector extent. Probe unknowns
  remain 64×64, avoiding a simultaneous change in blind-reconstruction capacity.
- Retains source wave/intensity scaling (not a new peak normalization).
  Orthonormal FFT padding preserves each frame's total clean intensity.
- Generates new fixed Poisson realizations at mean expected TOTAL counts/frame
  of 1e5 and 1e4, seed 42. Four times as many detector pixels does NOT mean four
  times as many photons. Array-energy SNR changes; do not reuse the 64-data SNR.
- Ideal point-intensity sampling, not finite detector-pixel integration/binning.
  This is a sampling-sensitivity experiment, not proof of an information ceiling.

From the repository root, generate all three datasets once:

```bash
python simulations/run_sampling128.py prepare
```

Default clean source is the existing
`results_paper/ptyinr_obj612_grid10_d59_s10_center/scene_obj612_grid10_d59_s10_center.h5`.
On another machine, supply its actual uploaded location with `--source "PATH_TO_CLEAN_H5"`.
All new data, audit metadata and results are under `results_paper/sampling128/fft128/`.
Preparation refuses to overwrite an existing destination; already prepared data
can be copied to another machine instead of regenerating them.

```bash
# Frozen U-Net16 -> residual U-Net32: 500 + 500 updates, internal concat kept.
python simulations/run_sampling128.py ours --dose clean
python simulations/run_sampling128.py ours --dose 1e4

# DeePIE on precisely the same measurements, 1000 updates.
python simulations/run_sampling128.py deepie --dose clean
python simulations/run_sampling128.py deepie --dose 1e4
```

`--dose 1e5` is also available. `--dry-run` prints commands without training.
`--root PATH` relocates all outputs. `--fft-size 64` prepares/runs the matching
legacy-size control in a separate `fft64/` subdirectory; pass the same option
at preparation and training. No old `windows_result.npz` is required.

Ours: network LR 0.0005 and probe LR 0.01 in both stages; stage-1 amplitude
loss + TGV 0.01; stage-2 clean amplitude / noisy Poisson loss + TGV 0.005;
no NLM, same energy-scaled regularization. The 128 diffraction inputs occupy a
128×128 area before padding to the object-network input size; the input is not
downsampled back to 64. Thus the end-to-end comparison also changes conditioning.

DeePIE: same model defaults as the supplied `deepie_1e4噪声.npz`; object LR 1e-6,
probe LR 0.001, decay by 0.5 every 500 updates (user-requested change from 800),
scan chunk 100. Previously completed runs retain their original 800-step setting.
Existing weighted
amplitude loss is retained (not silently changed to Poisson). Hardware and timing
must still be matched for speed comparisons.

Validation: 13 H5/residual tests passed, including legacy equality, centered
padding, energy conservation, gradients, metadata/noise roundtrip and overwrite
guard. Actual 128 data passed DeePIE check-only, a 4-update U16 smoke test and a
full-measurement U32 Poisson+TGV backward check; no full reconstruction was run.
The separate old progressive-launcher test expects `inner` while that launcher
currently defaults to `sparse`; that pre-existing assertion was not changed.

### Notebooks

| Notebook | Purpose | Runtime |
|---|---|---|
| `INNM.ipynb` | Original FPM simulation kept as a legacy reference | Original TensorFlow 1 / Keras environment |
| `INNM_FPM.ipynb` | TensorFlow 2 migration of the INNM FPM simulation | TensorFlow/Keras 2 |
| `INNM_Ptycho.ipynb` | AD-based ptychography simulation and regularization ablations | TensorFlow/Keras 2 |

Shared tools are in `../functions/`:

- `innm_common.py`: trainable object/probe layers, safe complex operations, subpixel Fourier shifts, regularizers, alignment, metrics, and plots.
- `diag_kurtosis.py`: gradient and kurtosis-pathology diagnostics.
- `project_paths.py`: paths to the project-level simulation assets.

The experiment-specific forward model, train loop, and configuration remain in each notebook.

## Shared simulation inputs

| File | Use |
|---|---|
| `../cameraman.bmp` | Ground-truth object amplitude |
| `../westconcordorthophoto.bmp` | Ground-truth object phase |
| `../spiral_kxky.txt` | FPM illumination positions |
| `../Zernike_Polyminals.mat` | FPM Zernike basis |

## FPM configuration

The current settings in `INNM_FPM.ipynb` and the legacy notebook are:

| Item | Value |
|---|---|
| Wavelength | 532 nm |
| Objective NA | 0.1 |
| Sensor pixel size used by the model | 1.725 µm |
| Reconstructed-plane pixel size | 0.43125 µm |
| High-resolution grid | 128 × 128 |
| Raw image size | 32 × 32 |
| Illumination array | 15 × 15 = 225 measurements |
| Illumination NA step | 0.05 |
| Defocus | 50 µm |
| Pupil representation | 10 Zernike modes |
| Object amplitude | `cameraman.bmp` |
| Object phase | normalized `westconcordorthophoto.bmp`, multiplied by 0.5π |
| Poisson noise | off |
| Optional Poisson peak | 5000 photons per measurement |
| Noise seed | 42 |
| Evaluation crop | 10 pixels per side |

The active TensorFlow 2 training cell uses 10 alternating stages, 20 object epochs and 20 pupil epochs per stage, object learning rate 200, pupil learning rate 0.001, decay 0.9, amplitude-TV weight 0.01, and phase-TV weight 1. The legacy notebook uses 5 stages with the same per-stage epoch counts and learning rates.

## Ptychography configuration

| Item | Value |
|---|---|
| Wavelength | 632.8 nm |
| Grid / detector size (patch) | 128 × 128 |
| Object canvas | 224 × 224 |
| Object addressing | Crop patch: integer slice plus a Fourier sub-pixel shift inside the patch |
| Sample and detector pixel size | 10 µm |
| Sample-to-detector distance | 8 mm |
| Nominal probe diameter | 48 pixels |
| True aperture-to-sample distance | 3.0 mm |
| Initial aperture-to-sample distance | 3.9 mm |
| Probe parameterization | Free complex matrix inside a binary support |
| Probe initialization | Propagated circular aperture |
| Object amplitude | `cameraman.bmp`, scaled to 0.4–1.0 |
| Object phase | normalized `westconcordorthophoto.bmp`, multiplied by 0.8 rad |
| Scan pattern | Jittered raster |
| Scan positions | 25 |
| Scan step | 20 pixels |
| Jitter | ±20% of the scan step |
| Scan seed | 0 |
| Data loss mode | Direct amplitude comparison, `abs(U)` against `sqrt(I)` (the Gerchberg–Saxton branch was removed after being shown bitwise equivalent) |
| Optimization | Alternating object/probe updates |
| Object / probe learning rate | 0.03 / 0.03 |
| Learning-rate decay | 0.75 per stage |
| Object / probe epochs | 8 / 8 per stage |
| Poisson noise | off |
| Optional Poisson peak | 5000 photons with a shared global scale |
| Noise seed | 42 |
| Evaluation crop | 64 pixels per side of the 224 canvas, giving a 96 × 96 ROI |
| Regularization crop | 48 pixels per side, giving the central 128 × 128 (roughly the illuminated area) |

The active reconstruction uses 8 stages with amplitude TGV weight 0.01, phase weight 0, and TGV coefficients `w1=1`, `w2=2`. The noise ablation uses a 16-pixel regularization crop, peak 1000 photons, seed 2026, and 8 stages.

## Rules for comparable tests

When comparing another implementation, keep the object construction, probe truth and initialization, scan positions, noise seed and normalization, detector size, forward propagation, loss mode, optimizer schedule, regularization region, and evaluation crop unchanged. Change one factor at a time and keep returned records under distinct Python variable names for in-memory comparisons.
