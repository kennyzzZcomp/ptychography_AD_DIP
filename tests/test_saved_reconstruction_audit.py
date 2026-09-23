import unittest
import numpy as np

from simulations.audit_saved_reconstructions import center_roi, coverage_stats, json_safe, weighted_overlap


class SavedAuditTests(unittest.TestCase):
    def test_center(self):
        self.assertEqual(center_roi((624, 624), 96), (slice(264, 360), slice(264, 360)))
        with self.assertRaises(ValueError):
            center_roi((20, 20), 96)

    def test_overlap_zero_padding_not_wrap(self):
        p = np.ones((4, 4), complex)
        self.assertEqual(weighted_overlap(p, 0), [1, 1])
        self.assertEqual(weighted_overlap(p, 1), [.75, .75])
        self.assertEqual(weighted_overlap(p, 4), [0, 0])
        with self.assertRaises(ValueError):
            weighted_overlap(p, 1.5)

    def test_coverage(self):
        p = np.ones((3, 3), complex)
        r = coverage_stats((8, 8), p, np.array([[0, 0], [1, 1]]), (slice(1, 3), slice(1, 3)))
        self.assertEqual(r['roi_min_support_count'], 2)
        self.assertEqual(r['roi_unilluminated_fraction'], 0)
        with self.assertRaises(ValueError):
            coverage_stats((8, 8), p, np.array([[7, 7]]), (slice(1, 3), slice(1, 3)))

    def test_nonfinite_history_is_explicit(self):
        self.assertEqual(json_safe({'history': [float('nan'), float('inf')]}), {'history': ['nan', 'inf']})


if __name__ == '__main__':
    unittest.main()
