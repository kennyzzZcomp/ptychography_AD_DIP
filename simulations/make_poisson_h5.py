"""Create one fixed Poisson-corrupted copy of a clean shared simulation H5.

No reconstruction is run and no existing file is overwritten. All scan geometry,
physical calibration, truth arrays, initial probe and ROI are copied verbatim.
Only intensity and diffamp are replaced; raw counts and provenance are added.

For ALL frames, use the SAME gain:
    alpha = mean_photons / mean_i(sum_p clean_intensity[i,p])
    counts[i,p] ~ Poisson(alpha * clean_intensity[i,p])
    intensity = counts / alpha; diffamp = sqrt(intensity)

mean_photons is the average EXPECTED total detected count per diffraction frame,
not per pixel or a calibrated incident dose. No per-frame renormalization,
Gaussian/readout noise, background, upper clipping or detector saturation.
The stored noisy array is fixed throughout reconstruction.

Example:
  python simulations/make_poisson_h5.py --source clean_scene.h5 \
      --output poisson_1e5_seed42.h5 --mean-photons 100000 --noise-seed 42

Requires numpy and h5py only. Use existing H5 adapters to reconstruct the output;
do NOT regenerate this noise using the legacy --noise/--snr simulation options.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def poisson_observation(clean, mean_photons=100000.0, seed=42):
    clean = np.asarray(clean, dtype=np.float64)
    if (clean.ndim != 3 or not clean.size or not np.isfinite(clean).all()
            or np.any(clean < 0)):
        raise ValueError("Expected finite, nonnegative (frames,height,width) clean intensities")
    if not np.isfinite(mean_photons) or mean_photons <= 0:
        raise ValueError("mean_photons must be finite and positive")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("noise seed must be a nonnegative integer")
    frame_signal = clean.sum(axis=(1, 2), dtype=np.float64)
    mean_signal = float(frame_signal.mean())
    if not np.isfinite(mean_signal) or mean_signal <= 0:
        raise ValueError("Clean dataset must have nonzero finite total intensity")
    alpha = float(mean_photons) / mean_signal
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid photon-to-intensity gain")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    counts = np.empty(clean.shape, dtype=np.int64)
    for i, frame in enumerate(clean):
        counts[i] = rng.poisson(alpha * frame)
    measured = (counts / alpha).astype(np.float32)
    if not np.isfinite(measured).all():
        raise ValueError("Photon scaling produced nonfinite float32 measurements")
    difference = measured.astype(np.float64) - clean
    signal_energy = float(np.square(clean).sum())
    error_energy = float(np.square(difference).sum())
    expected_error_energy = float(clean.sum() / alpha)
    observed_frame_counts = counts.sum(axis=(1, 2))
    expected_frame_counts = alpha * frame_signal
    snr = float(10 * np.log10(signal_energy / error_energy)) if error_energy else None
    info = dict(
        model="independent Poisson photon counts", seed=int(seed), rng="NumPy PCG64",
        numpy_version=np.__version__,
        mean_expected_photons_per_frame=float(mean_photons),
        photons_per_frame_definition="Average expected TOTAL detected counts per diffraction frame, not per pixel or incident dose",
        global_alpha=alpha, clean_mean_frame_intensity_sum=mean_signal,
        expected_total_photons=float(expected_frame_counts.sum()),
        observed_total_photons=int(observed_frame_counts.sum()),
        observed_mean_photons_per_frame=float(observed_frame_counts.mean()),
        expected_photons_per_frame=expected_frame_counts.tolist(),
        observed_photons_per_frame=observed_frame_counts.tolist(),
        measured_intensity_snr_db=snr,
        snr_definition="10*log10(sum(clean_intensity**2)/sum((measured_intensity-clean_intensity)**2)); full detector stack",
        expected_noise_energy_snr_db=float(10 * np.log10(signal_energy / expected_error_energy)),
        relative_intensity_l2=float(np.sqrt(error_energy / signal_energy)),
        zero_count_fraction=float(np.mean(counts == 0)),
        fraction_above_clean_global_peak=float(np.mean(measured > clean.max())),
        measured_intensity_min=float(measured.min()), measured_intensity_max=float(measured.max()),
        shape=list(clean.shape), per_frame_normalization=False, upper_clipping=False,
        additive_gaussian_noise=False, background_counts=False, fixed_during_reconstruction=True,
        diagnostic_truth_usage="Clean intensity used only to synthesize noise; other truth arrays copied unchanged for evaluation",
    )
    return measured, counts, info


def dataset_names(h5file):
    names = []
    h5file.visititems(lambda name, node: names.append(name) if isinstance(node, h5py.Dataset) else None)
    return names


def scene_fingerprint(h5file):
    """Self-described named-array hash; not the old simulation's SHA1 recipe."""
    digest = hashlib.sha256()
    for name in sorted(dataset_names(h5file)):
        if name.startswith("measurement_noise/"):
            continue
        array = np.ascontiguousarray(h5file[name][()])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(str(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()[:12]


def generate(source, output, mean_photons=100000.0, seed=42):
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    report_path = output.with_suffix(".noise.json")
    if source == output:
        raise ValueError("Source and output must differ")
    if output.exists() or report_path.exists():
        raise FileExistsError("Output H5/report already exists; choose a new filename")
    source_sha = file_sha256(source)
    with h5py.File(source, "r") as original:
        if original.attrs.get("schema") != "proptynet-ptyinr-v1":
            raise ValueError("Expected proptynet-ptyinr-v1 shared H5 scene")
        config = json.loads(original.attrs["scene_config_json"])
        if config.get("noise") != "none":
            raise ValueError("Start from an explicitly clean source (noise=none); never stack noise")
        if "measurement_noise" in original:
            raise ValueError("Source already has measurement-noise metadata")
        clean = original["diagnostic_truth/clean_intensity"][()]
        if not np.array_equal(original["intensity"][()], clean):
            raise ValueError("Source measured intensity is not identical to its clean intensity")
        if not np.allclose(original["diffamp"][()] ** 2, clean, rtol=2e-5, atol=1e-7):
            raise ValueError("Source intensity and diffraction amplitude disagree")
        for key in ("intensity", "diffamp"):
            if original[key].dtype != np.dtype("float32"):
                raise ValueError("This exporter expects the shared format's float32 measurements")
        measured, counts, noise = poisson_observation(clean, mean_photons, seed)
        if noise["measured_intensity_snr_db"] is None:
            raise ValueError("No representable corruption; choose a lower photon level")
        amp = np.sqrt(measured).astype(np.float32)
        old_fingerprint = str(original.attrs["scene_fingerprint"])
        noisy_config = dict(config)
        noisy_config.update(
            noise="poisson", noise_seed=int(seed),
            snr_db=noise["measured_intensity_snr_db"],
            snr_db_role="measured intensity SNR, NOT the legacy add_noise control parameter",
            noise_parameterization="fixed-global-alpha; mean expected detected photons per frame",
            poisson_mean_photons_per_frame=float(mean_photons), poisson_global_alpha=noise["global_alpha"],
            noise_source="stored H5; load verbatim, do not regenerate from legacy snr_db",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "x") as target:
            for key, value in original.attrs.items():
                target.attrs[key] = value
            for key in original:
                original.copy(key, target)
            target["intensity"][...] = measured
            target["diffamp"][...] = amp
            target.attrs["scene_config_json"] = json.dumps(noisy_config)
            target.attrs["source_scene_config_json"] = original.attrs["scene_config_json"]
            target.attrs["source_scene_fingerprint"] = old_fingerprint
            target.attrs["source_scene_sha256"] = source_sha
            target.attrs["noise_metadata_json"] = json.dumps(noise)
            group = target.create_group("measurement_noise")
            group.create_dataset("counts", data=counts, compression="gzip", shuffle=True)
            group.create_dataset("expected_photons_per_frame", data=noise["expected_photons_per_frame"])
            group.create_dataset("observed_photons_per_frame", data=noise["observed_photons_per_frame"])
            group.attrs["global_alpha"] = noise["global_alpha"]
            group.attrs["noise_seed"] = int(seed)
            group.attrs["model"] = "Poisson; intensity=counts/global_alpha; no clipping"
            target.attrs["scene_fingerprint_algorithm"] = "sha256-named-arrays-excluding-measurement_noise-v1-first12"
            target.attrs["scene_fingerprint"] = scene_fingerprint(target)

    # Verify that every original dataset except the two measured arrays is intact.
    preserved = []
    with h5py.File(source, "r") as original, h5py.File(output, "r") as target:
        for name in dataset_names(original):
            if name not in ("intensity", "diffamp"):
                if (original[name].dtype != target[name].dtype
                        or not np.array_equal(original[name][()], target[name][()])):
                    raise AssertionError(f"Unexpected change to {name}")
                preserved.append(name)
        np.testing.assert_array_equal(target["intensity"][()], measured)
        np.testing.assert_array_equal(target["diffamp"][()], amp)
        np.testing.assert_array_equal(target["measurement_noise/counts"][()], counts)
        if not np.allclose(amp ** 2, measured, rtol=2e-5, atol=1e-7):
            raise AssertionError("Saved noisy amplitude and intensity disagree")
        new_fingerprint = str(target.attrs["scene_fingerprint"])
        if new_fingerprint != scene_fingerprint(target):
            raise AssertionError("Scene fingerprint mismatch")
    if file_sha256(source) != source_sha:
        raise AssertionError("Source H5 unexpectedly changed")
    report = dict(source=str(source), output=str(output), source_sha256=source_sha,
        output_sha256=file_sha256(output), source_scene_fingerprint=old_fingerprint,
        scene_fingerprint=new_fingerprint, noise=noise,
        changed_original_datasets=["intensity", "diffamp"], preserved_original_datasets=preserved,
        original_file_unchanged=True, reconstruction_run=False,
        generator=Path(__file__).name, generator_sha256=file_sha256(__file__))
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mean-photons", type=float, default=100000.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    args = parser.parse_args()
    report = generate(args.source, args.output, args.mean_photons, args.noise_seed)
    noise = report["noise"]
    print(f"Created: {report['output']}")
    print(f"Frames: {noise['shape']}; expected mean photons/frame: {noise['mean_expected_photons_per_frame']:g}")
    print(f"Observed mean photons/frame: {noise['observed_mean_photons_per_frame']:.2f}; alpha: {noise['global_alpha']:.8g}")
    print(f"Measured intensity SNR: {noise['measured_intensity_snr_db']:.3f} dB; zero counts: {100*noise['zero_count_fraction']:.2f}%")
    print(f"Scene fingerprint: {report['scene_fingerprint']}; no clipping or per-frame rescaling")
    print("Original source, geometry and truth preserved. No reconstruction was run.")
    print(f"Audit: {Path(report['output']).with_suffix('.noise.json')}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
