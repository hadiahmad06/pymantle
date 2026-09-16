"""CUDA-aware timing helpers.

The rule that matters here: never let process startup (CUDA context init,
cuDNN/cuBLAS handle creation, kernel JIT/autotune, first-batch allocator
growth) leak into a measured window. We do that by running discarded warmup
iterations first, and only calling Timer.start() once the process is already
"hot" -- so the clock starts after startup, not at process launch.
"""

import time

import torch


def warmup(step_fn, n=20):
    for _ in range(n):
        step_fn()
    torch.cuda.synchronize()


class Timer:
    """Wall-clock + GPU-clock timer over a measured window.

    Wall time uses perf_counter bracketed by cuda syncs (catches host-side
    launch-gap overhead). GPU time uses cuda Events (catches device-only
    busy time). The gap between the two is, roughly, launch/dispatch
    overhead not hidden by async execution.
    """

    def __init__(self):
        self._start_evt = torch.cuda.Event(enable_timing=True)
        self._end_evt = torch.cuda.Event(enable_timing=True)
        self._wall_start = None

    def start(self):
        torch.cuda.synchronize()
        self._wall_start = time.perf_counter()
        self._start_evt.record()

    def stop(self):
        self._end_evt.record()
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - self._wall_start
        gpu_s = self._start_evt.elapsed_time(self._end_evt) / 1000.0
        return wall_s, gpu_s


def gpu_name():
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(torch.cuda.current_device())
