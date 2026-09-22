"""Opt-in synchronized stage timing. Disabled path never synchronizes CUDA."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch


class StageTimer:
    def __init__(self, device, warmup=-1):
        self.device = torch.device(device)
        self.warmup = warmup
        self.rows = []
        self.current = None
        self.last = None

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self, iteration):
        self.current = None
        if self.warmup < 0 or iteration < self.warmup:
            return
        self._sync()
        if iteration == self.warmup and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.current = {"iteration": iteration + 1}
        self.last = time.perf_counter()

    def mark(self, stage):
        if self.current is None:
            return
        self._sync()
        now = time.perf_counter()
        self.current[stage] = self.current.get(stage, 0.0) + now - self.last
        self.last = now

    def finish(self):
        if self.current is not None:
            self.rows.append(self.current)
            self.current = None

    def report(self):
        stages = sorted({key for row in self.rows for key in row if key != "iteration"})
        summaries = {}
        for stage in stages:
            values = [r[stage] for r in self.rows if stage in r]
            summaries[stage] = {"calls": len(values), "total_s": sum(values),
                                "mean_s": statistics.mean(values),
                                "median_s": statistics.median(values)}
        totals = [sum(v for k, v in r.items() if k != "iteration") for r in self.rows]
        result = {"warmup_iterations": self.warmup, "measured_iterations": len(self.rows),
                  "device": str(self.device), "torch_version": torch.__version__,
                  "stages": summaries, "iterations": self.rows,
                  "mean_iteration_s": statistics.mean(totals) if totals else None,
                  "note": "Synchronized diagnostic timings, not production throughput. "
                          "Setup, file saving and plotting excluded. Evaluation includes logging. "
                          "TGV stages include auxiliary backward/Adam; backward is outer backward."}
        if self.rows and self.device.type == "cuda":
            result.update(gpu_name=torch.cuda.get_device_name(self.device),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        return result

    def save(self, path):
        if self.warmup >= 0:
            Path(path).write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
