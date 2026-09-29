import unittest
from dataclasses import replace
import torch
from multiwave.config import Config
from multiwave.scene import simulate


class HighresTests(unittest.TestCase):
    def test_physical_geometry_and_scan_split_preserved(self):
        torch.set_num_threads(2)
        coarse = replace(Config.preset('standard'), photons_per_scan=0)
        fine = Config.preset('highres')
        coarse.validate()
        fine.validate()
        for key in ('object_size', 'patch_size', 'step', 'jitter'):
            self.assertEqual(getattr(coarse, key)*coarse.pixel_um,
                             getattr(fine, key)*fine.pixel_um)
        a, b = simulate(coarse), simulate(fine)
        torch.testing.assert_close(a.operator.positions*4, b.operator.positions)
        torch.testing.assert_close(a.train, b.train)
        torch.testing.assert_close(a.holdout, b.holdout)
        self.assertEqual(tuple(b.objects.shape), (2, 384, 384))
        self.assertEqual(float(b.objects.imag.abs().max()), 0)
        torch.testing.assert_close(b.objects[0], b.objects[1])

    def test_invalid_scan_quantum_rejected(self):
        with self.assertRaises(ValueError):
            replace(Config.preset('highres'), jitter=3).validate()


if __name__ == '__main__':
    unittest.main()
