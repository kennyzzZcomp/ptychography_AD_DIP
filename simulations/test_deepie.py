"""Scientific-contract tests: exact chain rule, shared scene, no GT optimization.

Kept beside the standalone entry because the repository ignores new tests/ files.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import torch

from simulations.DeePIE import parser, configurations
from simulations.ProPtyNet_paper import Cfg, PRESETS
from functions.paperrepro.deepie_model import (DeePIEObject, ModelConfig, coordinate_grid,
                                               fourier_basis, FourierLinear)
from functions.paperrepro.deepie_solver import (TrainingConfig, exact_chunked_gradients,
                                                measurement_loss, scan_weights, initial_probe_scale,
                                                balance_network_gradients, run_deepie)
from functions.paperrepro.scene import build_scene


@contextlib.contextmanager
def workspace_temp():
    # Windows sandbox cannot reopen Python 3.10's mode-0700 mkdtemp directories.
    root = Path(__file__).resolve().parents[1] / "tmp" / "deepie_tests"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield str(path)
    finally:
        assert path.resolve().parent == root.resolve()
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


class DeePIETests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        torch.manual_seed(7)

    def small_model(self, fourier=True):
        return DeePIEObject(ModelConfig(width=6, hidden_layers=2, high_frequencies=3,
                             low_frequencies=2, phases=4, encoding_side=8, omega=3.,
                             head_scale=.2, fourier_weights=fourier)).double()

    def test_literal_basis_and_trainable_coefficients(self):
        b = fourier_basis(7, 3, 2, 4)
        self.assertEqual(b.shape, (20, 7))
        z = torch.linspace(0, 1, 7, dtype=torch.float64)
        torch.testing.assert_close(b[0].double(), (.5*z).cos(), rtol=1e-6, atol=1e-7)
        model = self.small_model()
        layer = model.amplitude.layers[0]
        self.assertIsInstance(layer, FourierLinear)
        self.assertFalse(layer.basis.requires_grad)
        self.assertIn("coefficients", dict(layer.named_parameters()))
        self.assertNotIn("basis", dict(layer.named_parameters()))

    def test_chunked_complex_gradients_equal_direct_autograd(self):
        for fourier in (True, False):
            with self.subTest(fourier=fourier):
                model = self.small_model(fourier)
                reference = copy.deepcopy(model)
                coords = coordinate_grid(7, "cpu", torch.float64)
                probe = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.complex128))
                pref = torch.nn.Parameter(probe.detach().clone())
                pos = torch.tensor([[0,0], [2,1], [3,3]])
                q = torch.exp(1j * torch.randn(4,4,dtype=torch.float64))
                measured = torch.rand(3,4,4,dtype=torch.float64)
                weights = scan_weights(measured, "intensity")
                obj = reference(coords).reshape(7,7)
                from functions.paperrepro.optics import forward_field
                pred = forward_field(obj,pref,pos,q,4).abs()
                loss = ((pred-measured).square().mean((-2,-1))*weights).mean()
                loss.backward()
                tc = TrainingConfig(scan_chunk=2,coordinate_chunk=11,balance="none")
                actual = exact_chunked_gradients(model,probe,coords,7,pos,q,measured,weights,tc)
                torch.testing.assert_close(actual,loss.detach(),rtol=1e-10,atol=1e-12)
                torch.testing.assert_close(probe.grad,pref.grad,rtol=1e-9,atol=1e-11)
                for a,b in zip(model.parameters(),reference.parameters()):
                    torch.testing.assert_close(a.grad,b.grad,rtol=1e-8,atol=1e-10)

    def test_zero_residual_has_zero_gradient(self):
        model=self.small_model()
        coords=coordinate_grid(6,"cpu",torch.float64)
        probe=torch.nn.Parameter(torch.randn(4,4,dtype=torch.complex128))
        q=torch.ones(4,4,dtype=torch.complex128)
        pos=torch.tensor([[0,0],[2,2]])
        from functions.paperrepro.optics import forward_field
        measured=forward_field(model(coords).reshape(6,6),probe,pos,q,4).abs().detach()
        loss=exact_chunked_gradients(model,probe,coords,6,pos,q,measured,torch.ones(2),
                                     TrainingConfig(scan_chunk=1,coordinate_chunk=7))
        self.assertLess(loss.item(),1e-25)
        self.assertLess(probe.grad.abs().max().item(),1e-12)

    def test_measurement_calibration_no_truth_and_chunk_invariant(self):
        obj=torch.ones(6,6,dtype=torch.complex64)
        probe=torch.ones(4,4,dtype=torch.complex64)
        q=torch.ones_like(probe)
        pos=torch.tensor([[0,0],[2,2]])
        from functions.paperrepro.optics import forward_field
        measured=forward_field(obj,probe,pos,q,4).abs()*.13
        for chunk in (1,2):
            self.assertAlmostEqual(initial_probe_scale(obj,probe,pos,q,measured,chunk),.13,places=6)
        weights=scan_weights(measured,"intensity")
        torch.testing.assert_close(weights.mean(),torch.tensor(1.))

    def test_shared_scene_exactly_matches_baseline(self):
        cfg,_,_=configurations(parser().parse_args(["check","--preset","smoke","--noise","mixed",
                                                     "--snr","30","--seed","3","--device","cpu"]))
        ref=Cfg(**PRESETS["smoke"],preset="smoke",noise="mixed",snr_db=30,seed=3,device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            a,b=build_scene(cfg,"cpu"),build_scene(ref,"cpu")
        self.assertEqual(a.fp,b.fp)
        for name in ("Im","Icl","pos","obj","probe","P0"):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))

    def test_import_baseline_result_and_cli_override(self):
        with workspace_temp() as d:
            p=Path(d)/"net_result.npz"
            np.savez(p,cfg=json.dumps({"preset":"smoke","obj_size":180,"step_px":9,
                                      "seed":4,"lr":.9,"quad_sign":1.,"probe_init":"disk"}))
            cfg,mc,tc=configurations(parser().parse_args(["check","--scene-config",str(p),
                                                          "--seed","6"]))
            self.assertEqual((cfg.obj_size,cfg.step_px,cfg.seed,cfg.quad_sign),(180,9,6,1.))
            self.assertEqual(tc.lr,1e-4)
            self.assertEqual(mc.width,256)

    def test_balance_scales_only_network_gradients(self):
        model=self.small_model()
        for p in model.amplitude.parameters(): p.grad=torch.ones_like(p)
        for p in model.phase.parameters(): p.grad=4*torch.ones_like(p)
        stats=balance_network_gradients(model,"equal-norm")
        self.assertAlmostEqual(stats["amp_gradient_factor"],2.)
        self.assertAlmostEqual(stats["phase_gradient_factor"],.5)

    def test_checkpoint_metrics_are_post_update_and_reloadable(self):
        # Two updates on a tiny synthetic scene; no intensive reconstruction.
        cfg=Cfg(N=8,obj_size=16,grid=2,step_px=4,eval_size=8,iters=2,eval_every=1,
                device="cpu",probe_diam_um=2000)
        mc=ModelConfig(width=6,hidden_layers=2,high_frequencies=3,low_frequencies=2,
                       phases=4,encoding_side=8,omega=3.,head_scale=.1)
        tc=TrainingConfig(scan_chunk=3,coordinate_chunk=31,lr=1e-5,probe_lr=1e-5)
        with workspace_temp() as d:
            cfg.outdir=d
            with patch("functions.paperrepro.deepie_solver._save_convergence"), \
                 contextlib.redirect_stdout(io.StringIO()):
                hist=run_deepie(cfg,mc,tc)
                scene=build_scene(cfg,"cpu")
            saved=torch.load(Path(d)/"deepie_checkpoint.pt",weights_only=False)
            model=DeePIEObject(mc)
            model.load_state_dict(saved["model"])
            coords=coordinate_grid(16,"cpu")
            with torch.no_grad():
                obj=model(coords).reshape(16,16)
                loss=measurement_loss(obj,saved["probe"],scene.post,scene.Q,scene.sqrtIm,
                                      scan_weights(scene.sqrtIm,"intensity"),1)
            self.assertAlmostEqual(loss.item(),hist[-1]["loss"],places=6)
            manifest=json.loads((Path(d)/"deepie_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"],"complete")
            self.assertEqual(manifest["scene_fingerprint"],scene.fp)
            self.assertFalse(manifest["initialization_uses_ground_truth"])
            self.assertEqual(len(hist),2)
            with np.load(Path(d)/"deepie_result.npz",allow_pickle=False) as result:
                np.testing.assert_allclose(result["obj_rec"],obj.numpy(),rtol=1e-5,atol=1e-6)
                self.assertEqual(str(result["scene_fingerprint"]),scene.fp)
                self.assertEqual(json.loads(str(result["cfg"]))["quad_sign"],-1.)


if __name__ == "__main__":
    unittest.main()
