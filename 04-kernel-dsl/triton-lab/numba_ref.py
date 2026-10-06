"""AR004 triton-lab: numba-side runner (executed by the AR001 venv python, no torch).

Provides the numba leg of the three-way comparison (triton / cuBLAS / numba):
- vector add (naive 1D, 128 threads/block)
- SGEMM K1_tiled_T16 / K2b_regblock_64x64_4x4 imported VERBATIM from AR002 gemm-lab
  (sys.path import: zero re-implementation drift, AR002 stays untouched)

Protocol mirrors bench_dsl.py: 2s burn-in GEMM, 3 warmup, adaptive reps median,
CUDA-event timing. Emits JSON to stdout path given by --out so the global-python
parent can merge it into the same session (back-to-back scheduling).

Usage (venv python):
  python numba_ref.py --exp add --sizes 4096 1048576 ... --out results/e1na.json
  python numba_ref.py --exp gemm --sizes 256 512 1024 2048 --out results/e3na.json
  python numba_ref.py --exp probe            # env probe only
"""
import argparse
import json
import sys

import numpy as np
from numba import cuda, float32

sys.path.insert(0, r"D:\Infra\02-handwritten-kernels\gemm-lab")
import kernels_numba as k2  # noqa: E402  (AR002 verbatim kernels)


@cuda.jit
def add_kernel(x, y, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = x[i] + y[i]


def cdiv(a, b):
    return -(-a // b)


def median_time_ms(fn, target_seconds=1.0, min_reps=3, max_reps=10):
    """Adaptive-reps median timing with numba CUDA events (same discipline as AR002/003/004)."""
    fn()
    cuda.synchronize()
    start, end = cuda.event(), cuda.event()
    start.record()
    fn()
    end.record()
    end.synchronize()
    t1 = start.elapsed_time(end) / 1e3
    reps = max(min_reps, min(max_reps, int(round(target_seconds / max(t1, 1e-9)))))
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], reps


def burn(seconds=2.0):
    N = 2048
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    kernel, grid_fn, block = k2.KERNELS["K2b_regblock_64x64_4x4"]
    dA, dB = cuda.to_device(A), cuda.to_device(B)
    dC = cuda.device_array((N, N), dtype=np.float32)
    import time
    t0 = time.time()
    while time.time() - t0 < seconds:
        kernel[grid_fn(N), block](dA, dB, dC)
    cuda.synchronize()


def exp_add(sizes):
    records = []
    for n in sizes:
        x = np.random.randn(n).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        dx, dy = cuda.to_device(x), cuda.to_device(y)
        dout = cuda.device_array(n, dtype=np.float32)
        grid, block = cdiv(n, 128), 128
        fn = lambda: add_kernel[grid, block](dx, dy, dout, n)
        fn()
        cuda.synchronize()
        ref = x + y
        got = dout.copy_to_host()
        assert np.array_equal(got, ref), f"numba add gate FAIL at n={n}"
        ms, reps = median_time_ms(fn)
        gbps = 3 * n * 4 / (ms * 1e-3) / 1e9
        print(f"numba add n={n}: {ms:.4f} ms, {gbps:.1f} GB/s")
        records.append({"variant": "numba", "n": n, "time_ms": round(ms, 6),
                        "gbps": round(gbps, 2), "reps": reps})
    return records


def exp_gemm(sizes):
    records = []
    rng = np.random.default_rng(42)
    for name in ["K1_tiled_T16", "K2b_regblock_64x64_4x4"]:
        kernel, grid_fn, block = k2.KERNELS[name]
        for n in sizes:
            A = rng.standard_normal((n, n)).astype(np.float32)
            B = rng.standard_normal((n, n)).astype(np.float32)
            dA, dB = cuda.to_device(A), cuda.to_device(B)
            dC = cuda.device_array((n, n), dtype=np.float32)
            grid = grid_fn(n)
            fn = lambda: kernel[grid, block](dA, dB, dC)
            fn()
            cuda.synchronize()
            ref = A.astype(np.float64) @ B.astype(np.float64)
            got = dC.copy_to_host()
            ok = np.allclose(got, ref, atol=1e-3, rtol=1e-3)
            assert ok, f"numba gemm gate FAIL {name} n={n}"
            ms, reps = median_time_ms(fn)
            tf = 2 * n ** 3 / (ms * 1e-3) / 1e12
            print(f"numba {name} n={n}: {ms:.4f} ms, {tf:.3f} TFLOPS")
            records.append({"variant": name, "N": n, "time_ms": round(ms, 6),
                            "tflops": round(tf, 3), "reps": reps})
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["probe", "add", "gemm"])
    p.add_argument("--sizes", type=int, nargs="*", default=[])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = cuda.get_current_device()
    env = {"runtime": "numba-venv", "numba": __import__("numba").__version__,
           "device": str(dev.name), "cc": f"{dev.compute_capability.major}.{dev.compute_capability.minor}"}
    if args.exp == "probe":
        print(json.dumps(env))
        return
    print(f"[numba_ref] env: {env}")
    burn(2.0)
    print(f"[numba_ref] burned 2s")
    if args.exp == "add":
        records = exp_add(args.sizes)
        doc = {"exp": "E1numba", "env": env, "records": records}
    else:
        records = exp_gemm(args.sizes)
        doc = {"exp": "E3numba", "env": env, "records": records}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        print(f"[numba_ref] saved -> {args.out}")
    else:
        print(json.dumps(doc))


if __name__ == "__main__":
    main()
