"""Explicit initial fields; NumPy-only replay of PtyLab 0.3.1 CPM defaults.

No PtyLab/CuPy dependency, learned result, or ground truth is used here.
"""
import numpy as np


def initial_fields(mode, n, side, positions_px, pixel_um, seed):
    if mode == 'ones':
        return (np.ones((side, side), np.complex64),
                np.ones((n, n), np.complex64),
                {'rule': 'Exact object=1+0j, probe=1+0j; no perturbation or rescaling'})
    if mode != 'mpie':
        raise ValueError('initial_fields supports mpie or ones')
    # Reproduce the reference canvas sizing and RNG consumption, not our geometry.
    # PtyLab first evaluates positions on a temporary 4*N canvas.
    centered = np.asarray(positions_px, np.float64)
    centered = centered - centered.mean(axis=0)
    temporary_positions = np.round(centered + 1.5*n).astype(int)
    reference_side = int(np.ptp(temporary_positions, axis=0).max()) + 2*n
    reference_side += reference_side % 2
    rng = np.random.RandomState(seed)  # same stream as np.random.seed/rand
    reference_object = (np.ones((reference_side, reference_side))
                        + .001*rng.rand(reference_side, reference_side)).astype(np.complex64)
    # Keep the smaller network canvas; align origins to nearest integer pixel.
    shift = reference_side/2-n/2-np.asarray(positions_px).mean(axis=0)
    offset = np.round(shift).astype(int)
    y, x = map(int, offset)
    if min(y, x) < 0 or max(y+side, x+side) > reference_side:
        raise ValueError('Network canvas does not fit reference initialization canvas')
    obj = reference_object[y:y+side, x:x+side].copy()
    # PtyLab xp is linspace including BOTH endpoints, not arange(-N/2,N/2).
    axis = np.linspace(-n/2, n/2, n)*pixel_um
    xx, yy = np.meshgrid(axis, axis)
    diameter = n*pixel_um/3
    pupil = xx**2+yy**2 < (diameter/2)**2
    probe = ((np.ones((n, n), np.complex64)+.001*rng.rand(n, n))*pupil).astype(np.complex64)
    return obj, probe, dict(
        rule='PtyLab 0.3.1 ones object then circ probe; 0.001 real perturbation; no power rescaling',
        reference_object_side=reference_side, reference_object_crop_yx=[y,x],
        reference_origin_rounding_residual_px=(shift-offset).tolist(),
        probe_initial_diameter_um=float(diameter),
        object_alignment='Integer crop of reference initial canvas; our fractional coordinates and canvas remain unchanged',
        probe_alignment='Same seed, RNG order, full array, pupil grid, amplitude and zero phase as reference')
