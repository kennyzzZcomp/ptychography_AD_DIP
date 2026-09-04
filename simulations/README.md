# INNM simulations

This folder contains the simulation notebooks. Training results stay in memory and are shown inside the notebook; running a notebook does not create `res_*`, MAT metric files, checkpoints, or reconstructed-image files.

## Files

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
