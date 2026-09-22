import unittest
from unittest.mock import patch

from functions.paperrepro.timing import StageTimer


class TimingTests(unittest.TestCase):
    def test_disabled_has_no_clock_or_cuda_calls(self):
        timer = StageTimer("cuda", -1)
        with patch("torch.cuda.synchronize") as sync, \
             patch("time.perf_counter") as clock:
            timer.start(0)
            timer.mark("stage")
            timer.finish()
            sync.assert_not_called()
            clock.assert_not_called()
        self.assertEqual(timer.report()["measured_iterations"], 0)

    def test_warmup_and_durations(self):
        timer = StageTimer("cpu", 2)
        timer.start(1)
        timer.mark("ignored")
        timer.finish()
        with patch("time.perf_counter", side_effect=[10., 11., 13., 20., 24.]):
            timer.start(2)
            timer.mark("network")
            timer.mark("physics")
            timer.finish()
            timer.start(3)
            timer.mark("network")
            timer.finish()
        report = timer.report()
        self.assertEqual(report["measured_iterations"], 2)
        self.assertEqual(report["stages"]["network"]["mean_s"], 2.5)
        self.assertEqual(report["stages"]["physics"]["calls"], 1)
        self.assertEqual(report["mean_iteration_s"], 3.5)

    def test_cuda_sync_and_peak_reset_after_warmup(self):
        timer = StageTimer("cuda:0", 1)
        with patch("torch.cuda.synchronize") as sync, \
             patch("torch.cuda.reset_peak_memory_stats") as reset:
            timer.start(0)
            timer.mark("ignored")
            timer.finish()
            sync.assert_not_called()
            timer.start(1)
            timer.mark("network")
            timer.finish()
            timer.start(2)
            timer.mark("network")
            timer.finish()
            self.assertEqual(sync.call_count, 4)
            reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
