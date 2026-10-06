"""AR002 gemm-lab F2a: numba-side benchmark (run with the solutions/.venv python).

Series & N grids (design.md §3.2, per-kernel tiering to bound wall time):
  K0_naive              : 128, 256, 512, 1024
  K1_tiled_T{8,16,32}   : 256, 512, 1024, 2048
  K2a / K2b             : 256, 512, 1024, 2048, 4096

Timing discipline (identical semantics as bench_torch.py), hardened against
display-GPU session noise (WDDM + power-state drift measured up to ~10%
between sessions on this Quadro):
  per config: PASSES=3 consecutive passes, each = sustained warmup ~1.0s
  (clock ramp: idle Quadro downclocks) then 5 CUDA-event timed reps.
  Report the MEDIAN over all 15 samples per config. Device buffers are
  allocated/transferred once per config (PCIe H2D is the wall-time bottleneck).
JSON is the only cross-env artifact.
"""
import json
import os
import platform
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from numba import cuda  # noqa: E402

from kernels_numba import KERNELS  # noqa: E402

GRIDS = {
    "K0_naive": [128, 256, 512, 1024],
    "K1_tiled_T8": [256, 512, 1024, 2048],
    "K1_tiled_T16": [256, 512, 1024, 2048],
    "K1_tiled_T32": [256, 512, 1024, 2048],
    "K2a_regblock_32x32_2x2": [256, 512, 1024, 2048, 4096],
    "K2b_regblock_64x64_4x4": [256, 512, 1024, 2048, 4096],
}

PASSES = 3
REPS_PER_PASS = 5
WARMUP_S = 1.0


def time_config(kernel, grid, block, dA, dB, dC):
    """One pass: sustained warmup then REPS_PER_PASS event-timed runs."""
    t_end = time.perf_counter() + WARMUP_S
    while time.perf_counter() < t_end:
        kernel[grid, block](dA, dB, dC)
    cuda.synchronize()
    times = []
    for _ in range(REPS_PER_PASS):
        e1, e2 = cuda.event(), cuda.event()
        e1.record()
        kernel[grid, block](dA, dB, dC)
        e2.record()
        cuda.synchronize()
        times.append(e1.elapsed_time(e2))
    return times


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "bench_numba.json")
    rng = np.random.default_rng(2026)
    series = []
    for name, Ns in GRIDS.items():
        kernel, grid_fn, block = KERNELS[name]
        for N in Ns:
            # device buffers allocated & transferred ONCE per config; host
            # arrays are PAGEABLE-SLOW over WDDM (~2 GB/s measured), so pin
            # them for DMA transfer.
            A = cuda.pinned_array((N, N), dtype=np.float32)
            B = cuda.pinned_array((N, N), dtype=np.float32)
            A[:] = rng.standard_normal((N, N))
            B[:] = rng.standard_normal((N, N))
            dA, dB = cuda.to_device(A), cuda.to_device(B)
            dC = cuda.device_array((N, N), dtype=np.float32)
            grid = grid_fn(N)
            times = []
            for _ in range(PASSES):
                times.extend(time_config(kernel, grid, block, dA, dB, dC))
            med_ms = float(np.median(times))
            series.append({
                "kernel": name,
                "dtype": "fp32",
                "N": N,
                "time_ms": round(med_ms, 4),
                "tflops": round(2.0 * N**3 / (med_ms * 1e-3) / 1e12, 4),
                "reps": len(times),
            })
            print(f"{name:<26} N={N:<5} {med_ms:>10.3f} ms  {series[-1]['tflops']:>8.3f} TFLOPS  ({len(times)} samples)", flush=True)
            del A, B, dA, dB, dC
    dev = cuda.get_current_device()
    payload = {
        "env": {
            "python": platform.python_version(),
            "lib": "numba 0.68.0 (numba-cuda) + numpy",
            "gpu": str(dev.name),
            "cc": f"{dev.compute_capability.major}.{dev.compute_capability.minor}",
            "timing": (
                f"cuda.event; {PASSES} consecutive passes x {REPS_PER_PASS} reps = {PASSES * REPS_PER_PASS}"
                " samples per config, median; sustained warmup ~1s per pass (clock ramp)"
            ),
        },
        "series": series,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    for r in series:
        print(f"{r['kernel']:<26} N={r['N']:<5} {r['time_ms']:>10.3f} ms  {r['tflops']:>8.3f} TFLOPS")
    print(f"\nwrote {out_path} ({len(series)} records)")


if __name__ == "__main__":
    sys.exit(main())
