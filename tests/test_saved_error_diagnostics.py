import unittest
import numpy as np

from simulations.diagnose_saved_errors import coverage, periodic_cv


class SavedErrorDiagnosticsTests(unittest.TestCase):
    def test_coverage_adds_intensity_not_amplitude(self):
        p = np.full((2, 2), 2+0j)
        actual = coverage((4, 4), p, np.array([[0, 0], [1, 1]]))
        self.assertEqual(actual[1, 1], 8)
        self.assertEqual(actual[0, 0], 4)
        self.assertEqual(actual[3, 3], 0)
        with self.assertRaises(ValueError):
            coverage((4, 4), p, np.array([[3, 3]]))

    def test_periodic_signal_generalizes_to_heldout_tiles(self):
        rng = np.random.default_rng(91)
        template = rng.normal(size=(5, 5)) + 1j*rng.normal(size=(5, 5))
        values = np.tile(template, (4, 4))
        out = periodic_cv(values, np.ones(values.shape), 5)
        self.assertAlmostEqual(out['heldout_skill_vs_train_constant'], 1.)
        self.assertEqual(out['scored_pixels'], 400)
        self.assertEqual(out['min_train_pixels_per_scored_residue'], 8)

    def test_independent_noise_is_not_in_sample_explained_variance(self):
        rng = np.random.default_rng(91)
        values = rng.normal(size=(96, 96)) + 1j*rng.normal(size=(96, 96))
        out = periodic_cv(values, np.ones(values.shape), 35)
        self.assertLess(out['heldout_skill_vs_train_constant'], 0)
        self.assertEqual(out['scored_pixels'], values.size)


if __name__ == '__main__':
    unittest.main()
