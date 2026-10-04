"""Local amplitude-Jacobian information audit; no network training or edits.

The actual 512x512 Fresnel model is linearized at the simulated true O,P.
18 object directions (relative amplitude/phase, tensor Legendre degree 0..2)
and 6 probe nuisance directions (relative amplitude/phase, 1,x,y) are used.
J.T J is information for unit-variance independent Gaussian amplitude noise,
or an unnormalised Gauss-Newton matrix. Current training is noiseless: this
is a restricted local diagnostic, not its full Hessian or a measured CRLB.
All subsets use the SAME parameterization and evaluation point.
"""
import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
import sys
import time

os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from scipy.fft import fft2, fftshift, ifftshift
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    def threadpool_limits(limits):
        return nullcontext()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.ProPtyNet_paper import Cfg, PRESETS
from functions.paperrepro.sample import make_truth, make_positions
from functions.paperrepro.window_training import measurement_groups


def marginal_object_information(gram, nobj):
    a, b, c = gram[:nobj, :nobj], gram[:nobj, nobj:], gram[nobj:, nobj:]
    eff = a - b @ np.linalg.pinv(c, rcond=1e-12) @ b.T
    return (eff + eff.T)/2


def retention(reference, candidate):
    eig, vec = np.linalg.eigh(reference)
    keep = eig > eig.max()*1e-9
    whitening = vec[:, keep] / np.sqrt(eig[keep])
    value = np.linalg.eigvalsh(whitening.T @ candidate @ whitening)
    return dict(identifiable_reference_modes=int(keep.sum()),
                minimum=float(value.min()), mean=float(value.mean()),
                maximum=float(value.max()), eigenvalues=value.tolist())


def fft(a):
    return fftshift(fft2(ifftshift(a, axes=(-2, -1)), norm="ortho", workers=1), axes=(-2, -1))


def calculate():
    kw = dict(PRESETS["paper"])
    kw.update(grid=10, step_px=12, obj_size=624)
    cfg = Cfg(**kw, preset="paper", eval_size=96, seed=0, device="cpu")
    cfg.quad_sign = -1.0  # same default assigned by build_scene
    obj, probe, _, _ = make_truth(cfg)
    obj, probe = obj.astype(complex), probe.astype(complex)
    positions = make_positions(cfg)
    m, n = cfg.obj_size, cfg.N
    oy, ox = np.mgrid[:m, :m]
    # Fixed coordinates common to all layouts, spanning the illuminated region.
    ox, oy = (ox-m/2)/84, (oy-m/2)/84
    px = [np.ones_like(ox), ox, (3*ox**2-1)/2]
    py = [np.ones_like(oy), oy, (3*oy**2-1)/2]
    modes = np.stack([px[i]*py[j] for i in range(3) for j in range(3)])
    dy, dx = np.mgrid[:n, :n]-n//2
    pmodes = np.stack([np.ones_like(dx), dx/(cfg.probe_diam_px/2), dy/(cfg.probe_diam_px/2)])
    q = np.exp(1j*cfg.quad_sign*np.pi/(cfg.wlength*cfg.z)*cfg.dx1**2*(dx**2+dy**2))
    grams, crosses, energies = [], [], []
    checks = []
    start = time.perf_counter()
    nobj = 2*len(modes)
    with threadpool_limits(limits=4):
        for j, (y, x) in enumerate(positions):
            field = obj[y:y+n, x:x+n]*probe*q
            u = fft(field)
            a = np.abs(u)
            direction = np.divide(np.conj(u), a, out=np.zeros_like(u), where=a>1e-14)
            local_modes = modes[:, y:y+n, x:x+n]
            spatial = np.concatenate([field*local_modes, 1j*field*local_modes,
                                      field*pmodes, 1j*field*pmodes])
            jac = np.empty((len(spatial), n*n))
            for lo in range(0, len(spatial), 4):
                jac[lo:lo+4] = np.real(direction*fft(spatial[lo:lo+4])).reshape(-1, n*n)
            grams.append(jac @ jac.T)
            crosses.append(jac @ a.ravel())
            energies.append(float(np.sum(a*a)))
            if j == 0:
                for index in (2, 12, 19, 23):
                    eps = 1e-5
                    diff = (abs(fft(field+eps*spatial[index]))-abs(fft(field-eps*spatial[index])))/(2*eps)
                    error = np.linalg.norm(diff.ravel()-jac[index])/max(np.linalg.norm(jac[index]), 1e-12)
                    checks.append(dict(direction=index, relative_error=float(error)))
                    assert error < 1e-5, checks
            if (j+1)%20 == 0:
                print(f"Jacobian frames {j+1}/100, {time.perf_counter()-start:.1f}s", flush=True)
    grams, crosses, energies = np.stack(grams), np.stack(crosses), np.array(energies)
    specs = {"corner4": ("sparse", 4), "inner4": ("inner", 4),
             "sparse5": ("sparse", 5), "compact64": ("compact-matched", 4),
             "compact100": ("compact-full", 4)}
    full = grams.sum(0)
    effective_full = marginal_object_information(full, nobj)
    result = dict(model="Gaussian amplitude, unit variance; true-state low-dimensional tangent audit",
                  object_modes=nobj, probe_nuisance_modes=6,
                  finite_difference_checks=checks, layouts={}, elapsed_s=time.perf_counter()-start)
    matrices = dict(frame_grams=grams, frame_scale_cross=crosses, frame_scale_energy=energies,
                    full=full, full_effective=effective_full)
    for name, (layout, side) in specs.items():
        groups = measurement_groups(10, layout, side)
        selected = sorted(set(sum(groups, [])))
        mat = grams[selected].sum(0)
        effective = marginal_object_information(mat, nobj)
        # Per-group amplitude calibration is a separate nuisance parameter.
        profiled = mat.copy()
        for g in groups:
            v = crosses[g].sum(0)
            profiled -= np.outer(v, v)/energies[g].sum()
        total_v = crosses[selected].sum(0)
        global_profiled = mat - np.outer(total_v, total_v)/energies[selected].sum()
        recovered = global_profiled - profiled
        recovered_eig = np.linalg.eigvalsh(recovered)
        recovered_rank = int(np.sum(recovered_eig > np.linalg.norm(mat, 2)*1e-9))
        assert recovered_eig.min() > -np.linalg.norm(mat, 2)*1e-9
        assert recovered_rank <= len(groups)-1
        profiled = marginal_object_information(profiled, nobj)
        result["layouts"][name] = dict(frames=len(selected),
            known_probe=retention(full[:nobj, :nobj], mat[:nobj, :nobj]),
            unknown_probe=retention(effective_full, effective),
            unknown_probe_and_group_scales=retention(effective_full, profiled),
            group_to_global_scale_information_rank=recovered_rank,
            group_to_global_scale_information_eigenvalues=recovered_eig.tolist(),
            extra_data_effective_min_eigenvalue=float(np.linalg.eigvalsh(effective_full-effective).min()))
        matrices[name] = effective
        matrices[name+"_scale_profiled"] = profiled
        assert result["layouts"][name]["unknown_probe"]["maximum"] < 1+1e-7
        assert result["layouts"][name]["unknown_probe"]["minimum"] > -1e-7
    np.testing.assert_allclose(matrices["corner4"], matrices["compact64"], atol=1e-8)
    np.testing.assert_allclose(matrices["sparse5"], matrices["compact100"], atol=1e-8)
    return result, matrices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    out = Path(args.outdir)
    if out.exists():
        raise FileExistsError(out)
    result, matrices = calculate()
    serialized = json.dumps(result, indent=2)
    out.mkdir(parents=True, exist_ok=False)
    (out/"information.json").write_text(serialized, encoding="utf-8")
    np.savez_compressed(out/"information_matrices.npz", **matrices)
    print(serialized)


if __name__ == "__main__":
    main()
