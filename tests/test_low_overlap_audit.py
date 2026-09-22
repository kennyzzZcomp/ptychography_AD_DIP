import unittest

from simulations.low_overlap_audit import audit, unet_conv_macs


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = audit()["gauge_cases"]

    def test_native_gauge_survives_all_native_subsets(self):
        for name in ("master", "stride2", "stride2_plus_native_offsets"):
            self.assertLess(self.cases[name]["native_period_d"]["exit_relative_error"], 1e-12)

    def test_stride_adds_gauge_not_shared_by_master(self):
        self.assertLess(self.cases["stride2"]["enlarged_period_2d"]["intensity_relative_error"], 1e-12)
        self.assertGreater(self.cases["master"]["enlarged_period_2d"]["intensity_relative_error"], .01)

    def test_native_offsets_break_enlarged_test_gauge(self):
        value = self.cases["stride2_plus_native_offsets"]["enlarged_period_2d"]
        self.assertGreater(value["intensity_relative_error"], .01)

    def test_off_grid_breaks_both_constructed_gauges_not_uniqueness_proof(self):
        for value in self.cases["stride2_plus_new_off_grid_positions"].values():
            if isinstance(value, dict):
                self.assertGreater(value["intensity_relative_error"], .01)

    def test_only_first_conv_depends_on_input_channels(self):
        expected_delta = (64 - 4) * 32 * 9 * 624 ** 2
        self.assertEqual(unet_conv_macs(64) - unet_conv_macs(4), expected_delta)
        with self.assertRaises(ValueError):
            unet_conv_macs(4, size=625)


if __name__ == "__main__":
    unittest.main()
