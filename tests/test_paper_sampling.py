import unittest
import numpy as np
import torch

from functions.paperrepro.sampling import MeasurementSchedule
from functions.paperrepro.optics import forward_field


class SamplingTests(unittest.TestCase):
    def test_nested_fixed_and_full(self):
        plan = MeasurementSchedule("0:4,2:2,4:1", 8, 6)
        a, _ = plan.select(0)
        b, _ = plan.select(2)
        c, _ = plan.select(4)
        self.assertEqual([len(a), len(b), len(c)], [4, 16, 64])
        self.assertTrue(set(a) <= set(b) <= set(c))
        np.testing.assert_array_equal(c, np.arange(64))

    def test_rotating_covers_grid(self):
        plan = MeasurementSchedule("0:2,8:1", 8, 10, "rotate")
        groups = [plan.select(i)[0] for i in range(4)]
        self.assertEqual(len(np.unique(np.concatenate(groups))), 64)
        self.assertTrue(all(len(g) == 16 for g in groups))

    def test_random_reproducible_independent(self):
        a = MeasurementSchedule("0:2,8:1", 8, 10, "random", 7)
        b = MeasurementSchedule("0:2,8:1", 8, 10, "random", 7)
        for i in range(10):
            x, _ = a.select(i)
            y, _ = b.select(i)
            np.testing.assert_array_equal(x, y)
            self.assertEqual(len(x), len(set(x)))

    def test_invalid_schedules(self):
        for spec in ("1:2,3:1", "0:2", "0:9,4:1", "0:1,2:2,4:1",
                     "0:2,0:1", "0:2,10:1", "bad", "0:0,2:1"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                MeasurementSchedule(spec, 8, 10)

    def test_ten_by_ten_fixed_stride_three(self):
        plan = MeasurementSchedule("0:3,1000:1", 10, 4000)
        expected = np.array([10*r+c for r in (0, 3, 6, 9) for c in (0, 3, 6, 9)])
        for it in (0, 999):
            np.testing.assert_array_equal(plan.select(it)[0], expected)
        np.testing.assert_array_equal(plan.select(1000)[0], np.arange(100))
        for policy in ("rotate", "random"):
            with self.assertRaisesRegex(ValueError, "divide grid"):
                MeasurementSchedule("0:3,1000:1", 10, 4000, policy)

    def test_actual_forward_subset_matches_full_and_backward(self):
        torch.manual_seed(1)
        obj = torch.randn(12, 12, dtype=torch.complex64, requires_grad=True)
        probe = torch.randn(8, 8, dtype=torch.complex64, requires_grad=True)
        q = torch.ones_like(probe)
        pos = torch.tensor([[0, 0], [0, 4], [4, 0], [4, 4]])
        full = forward_field(obj, probe, pos, q, 8)
        part = forward_field(obj, probe, pos[[0, 3]], q, 8)
        torch.testing.assert_close(part, full[[0, 3]])
        part.abs().square().mean().backward()
        self.assertTrue(torch.isfinite(obj.grad).all())
        self.assertTrue(torch.isfinite(probe.grad).all())


if __name__ == "__main__":
    unittest.main()
