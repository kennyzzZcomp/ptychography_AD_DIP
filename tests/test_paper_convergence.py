"""Convergence plot is saved from recorded ROI metrics after training."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from functions.paperrepro.report import _save, _save_convergence
from simulations.ProPtyNet_paper import Cfg


class ConvergencePlotTests(unittest.TestCase):
    def test_writes_separate_plot_for_each_solver_tag(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib is not installed")

        hist = [
            {"it": 25, "psnr_amp": 8.0, "relerr": 0.6},
            {"it": 50, "psnr_amp": 12.0, "relerr": 0.3},
        ]
        with tempfile.TemporaryDirectory() as td:
            cfg = SimpleNamespace(outdir=td)
            for tag in ("paper", "ad", "net"):
                _save_convergence(cfg, hist, tag, plt)
                path = Path(td) / f"{tag}_convergence.png"
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)

    def test_result_saves_curve_and_training_time_separately(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as td:
            cfg = Cfg(N=8, obj_size=16, grid=2, step_px=4, iters=50,
                      outdir=td)
            obj = np.ones((16, 16), dtype=np.complex64)
            probe = np.ones((8, 8), dtype=np.complex64)
            hist = [{"it": 25, "psnr_amp": 8.0, "relerr": 0.6,
                     "ssim_amp": 0.2, "ssim_phs": 0.1, "loss": 1.0},
                    {"it": 50, "psnr_amp": 12.0, "relerr": 0.3,
                     "ssim_amp": 0.4, "ssim_phs": 0.3, "loss": 0.5}]
            positions = np.array([[2, 2], [2, 6], [6, 2], [6, 6]])
            _save(cfg, obj, probe, obj, probe, hist,
                  (slice(4, 12), slice(4, 12)), positions,
                  tag="net", train_elapsed_s=2.0)

            self.assertTrue((Path(td) / "net_convergence.png").is_file())
            self.assertTrue((Path(td) / "net_result.png").is_file())
            with np.load(Path(td) / "net_result.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["train_elapsed_s"].item(), 2.0)
                self.assertEqual(saved["mean_iteration_s"].item(), 0.04)


if __name__ == "__main__":
    unittest.main()
