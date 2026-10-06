"""AR002 gemm-lab F2b: cuBLAS baseline benchmark (run with the GLOBAL python).

Series & N grids (design.md §3.2):
  cublas_fp32 : 256, 512, 1024, 2048, 4096   (torch.matmul on float32)
  cublas_fp16 : 256, 512, 1024, 2048, 4096   (torch.matmul on float16, tensor cores)

Timing discipline identical to bench_numba.py (hardened against display-GPU
session noise): PASSES=3 full sweeps; per (config, pass) sustained warmup ~1.0s
then 5 torch.cuda.Event timed reps; report MEDIAN over all 15 samples.
Also writes results/env.json (machine/driver snapshot for the report header).
"""
import json
import os
import platform
import sys
import time
from collections import defaultdict

import numpy as np
import torch

GRIDS = {
    "cublas_fp32": ([256, 512, 1024, 2048, 4096], torch.float32),
    "cublas_fp16": ([256, 512, 1024, 2048, 4096], torch.float16),
}

PASSES = 3
REPS_PER_PASS = 5
WARMUP_S = 1.0


def time_config(A, B, C):
    fn = lambda: torch.matmul(A, B, out=C)
    t_end = time.perf_counter() + WARMUP_S
    while time.perf_counter() < t_end:
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS_PER_PASS):
        e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e1.record()
        fn()
        e2.record()
        torch.cuda.synchronize()
        times.append(e1.elapsed_time(e2))
    return times


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "results", "bench_torch.json")
    env_path = os.path.join(here, "results", "env.json")
    samples = defaultdict(list)  # (kernel, N) -> [time_ms]
    for p in range(PASSES):
        for name, (Ns, dtype) in GRIDS.items():
            for N in Ns:
                A = torch.randn(N, N, dtype=dtype, device="cuda")
                B = torch.randn(N, N, dtype=dtype, device="cuda")
                C = torch.empty(N, N, dtype=dtype, device="cuda")
                samples[(name, N)].extend(time_config(A, B, C))
                del A, B, C
                torch.cuda.empty_cache()
        print(f"pass {p + 1}/{PASSES} done")
    series = []
    for name, (Ns, dtype) in GRIDS.items():
        for N in Ns:
            times = samples[(name, N)]
            med_ms = float(np.median(times))
            series.append({
                "kernel": name,
                "dtype": "fp32" if dtype == torch.float32 else "fp16",
                "N": N,
                "time_ms": round(med_ms, 4),
                "tflops": round(2.0 * N**3 / (med_ms * 1e-3) / 1e12, 4),
                "reps": len(times),
            })
    payload = {
        "env": {
            "python": platform.python_version(),
            "lib": f"torch {torch.__version__} (cuBLAS)",
            "gpu": torch.cuda.get_device_name(0),
            "cc": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "timing": (
                f"torch.cuda.Event; {PASSES} full passes x {REPS_PER_PASS} reps = {PASSES * REPS_PER_PASS}"
                " samples per config, median; sustained warmup ~1s per config per pass (clock ramp)"
            ),
        },
        "series": series,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    prop = torch.cuda.get_device_properties(0)
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "sm_count": prop.multi_processor_count,
        "total_mem_gb": round(prop.total_memory / 2**30, 1),
        "torch": torch.__version__,
        "date": "2026-10-05",
    }
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2, default=str)
    for r in series:
        print(f"{r['kernel']:<14} N={r['N']:<5} {r['time_ms']:>10.3f} ms  {r['tflops']:>8.3f} TFLOPS")
    print(f"\nwrote {out_path} ({len(series)} records) + {env_path}")


if __name__ == "__main__":
    sys.exit(main())
