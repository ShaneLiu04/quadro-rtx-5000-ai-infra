"""AR002 gemm-lab F1: FP32 SGEMM optimization progression on numba CUDA (sm_75 real GPU).

Mirrors the LeetCUDA kernels/sgemm progression, reimplemented in numba CUDA
because this machine has no nvcc/MSVC (CUDA C++ toolchain unavailable):

  K0 naive            : one thread per output, k-loop reads global memory directly
  K1 tiled(T)         : T x T shared-memory tiles, cooperative guarded load,
                        double syncthreads around the compute loop
  K2 regblock(BM,BN,TM,TN)
                      : register blocking; each thread accumulates TM x TN outputs
                        in registers; cooperative linear-strided tile loading
                        configs: K2a = (32,32,2,2), K2b = (64,64,4,4)

All kernels handle non-multiple-of-tile N (zero-padding on load, guard on store).
Uniform-path syncthreads discipline: sync is never inside a divergent branch.

Run as script = correctness gate (T002): every kernel x N in {96,128,200,500}
compared against a float64 reference on the real GPU. Non-zero exit on failure.
"""
import sys

import numpy as np
from numba import cuda, float32


# ---------------------------------------------------------------- K0 naive
@cuda.jit
def sgemm_naive(A, B, C):
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    N = C.shape[0]
    if i < N and j < N:
        acc = float32(0.0)
        for k in range(N):
            acc += A[i, k] * B[k, j]
        C[i, j] = acc


# ---------------------------------------------------------------- K1 tiled
def make_tiled(T):
    """K1: T x T shared tiling. T is frozen as a compile-time constant."""

    @cuda.jit
    def sgemm_tiled(A, B, C):
        i0 = cuda.blockIdx.y * T
        j0 = cuda.blockIdx.x * T
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        N = C.shape[0]
        As = cuda.shared.array((T, T), dtype=float32)
        Bs = cuda.shared.array((T, T), dtype=float32)
        acc = float32(0.0)
        for k0 in range(0, N, T):
            # cooperative guarded load; out-of-range elements contribute 0.0
            if i0 + ty < N and k0 + tx < N:
                As[ty, tx] = A[i0 + ty, k0 + tx]
            else:
                As[ty, tx] = float32(0.0)
            if k0 + ty < N and j0 + tx < N:
                Bs[ty, tx] = B[k0 + ty, j0 + tx]
            else:
                Bs[ty, tx] = float32(0.0)
            cuda.syncthreads()
            for t in range(T):
                acc += As[ty, t] * Bs[t, tx]
            cuda.syncthreads()
        if i0 + ty < N and j0 + tx < N:
            C[i0 + ty, j0 + tx] = acc

    return sgemm_tiled


# ---------------------------------------------------------------- K2 regblock
def make_regblock(BM, BN, TM, TN):
    """K2: register blocking. Thread (tx,ty) computes a TM x TN sub-tile of C.

    Block threads: (BN//TN) x (BM//TM). K step = BM = BN (square configs).
    Cooperative loading: each thread strides linearly over the tile elements,
    guard + zero-pad beyond N. Accumulators live in registers (unrolled loops).
    """
    NTX = BN // TN
    NTY = BM // TM
    NTHREADS = NTX * NTY
    K_STEP = BM

    @cuda.jit
    def sgemm_regblock(A, B, C):
        i0 = cuda.blockIdx.y * BM
        j0 = cuda.blockIdx.x * BN
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        rank = ty * NTX + tx
        N = C.shape[0]
        As = cuda.shared.array((BM, K_STEP), dtype=float32)
        Bs = cuda.shared.array((K_STEP, BN), dtype=float32)
        acc = cuda.local.array((TM, TN), dtype=float32)
        for m in range(TM):
            for n in range(TN):
                acc[m, n] = float32(0.0)
        for k0 in range(0, N, K_STEP):
            for idx in range(rank, BM * K_STEP, NTHREADS):
                r = idx // K_STEP
                c = idx % K_STEP
                if i0 + r < N and k0 + c < N:
                    As[r, c] = A[i0 + r, k0 + c]
                else:
                    As[r, c] = float32(0.0)
            for idx in range(rank, K_STEP * BN, NTHREADS):
                r = idx // BN
                c = idx % BN
                if k0 + r < N and j0 + c < N:
                    Bs[r, c] = B[k0 + r, j0 + c]
                else:
                    Bs[r, c] = float32(0.0)
            cuda.syncthreads()
            for kk in range(K_STEP):
                for m in range(TM):
                    for n in range(TN):
                        acc[m, n] += As[ty * TM + m, kk] * Bs[kk, tx * TN + n]
            cuda.syncthreads()
        for m in range(TM):
            for n in range(TN):
                gi = i0 + ty * TM + m
                gj = j0 + tx * TN + n
                if gi < N and gj < N:
                    C[gi, gj] = acc[m, n]

    return sgemm_regblock


# ---------------------------------------------------------------- registry
def cdiv(a, b):
    return -(-a // b)


sgemm_tiled_8 = make_tiled(8)
sgemm_tiled_16 = make_tiled(16)
sgemm_tiled_32 = make_tiled(32)
sgemm_rb_32x32_2x2 = make_regblock(32, 32, 2, 2)
sgemm_rb_64x64_4x4 = make_regblock(64, 64, 4, 4)

# name -> (jit kernel, grid_fn(N), block_2d)
KERNELS = {
    "K0_naive": (sgemm_naive, lambda N: (cdiv(N, 16), cdiv(N, 16)), (16, 16)),
    "K1_tiled_T8": (sgemm_tiled_8, lambda N: (cdiv(N, 8), cdiv(N, 8)), (8, 8)),
    "K1_tiled_T16": (sgemm_tiled_16, lambda N: (cdiv(N, 16), cdiv(N, 16)), (16, 16)),
    "K1_tiled_T32": (sgemm_tiled_32, lambda N: (cdiv(N, 32), cdiv(N, 32)), (32, 32)),
    "K2a_regblock_32x32_2x2": (sgemm_rb_32x32_2x2, lambda N: (cdiv(N, 32), cdiv(N, 32)), (16, 16)),
    "K2b_regblock_64x64_4x4": (sgemm_rb_64x64_4x4, lambda N: (cdiv(N, 64), cdiv(N, 64)), (16, 16)),
}


def run_gpu(kernel, grid, block, A, B):
    """Launch kernel on the real GPU and return the host-side C."""
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.device_array(A.shape, dtype=np.float32)
    kernel[grid, block](dA, dB, dC)
    cuda.synchronize()
    return dC.copy_to_host()


# ---------------------------------------------------------------- T002 gate
def correctness_gate(verbose=True):
    rng = np.random.default_rng(42)
    sizes = [96, 128, 200, 500]  # deliberately non-multiple-of-tile cases
    failures = []
    print(f"{'kernel':<26} " + " ".join(f"N={n:<5}" for n in sizes) + " max_abs_err")
    for name, (kernel, grid_fn, block) in KERNELS.items():
        row = f"{name:<26} "
        worst = 0.0
        for N in sizes:
            A = rng.standard_normal((N, N)).astype(np.float32)
            B = rng.standard_normal((N, N)).astype(np.float32)
            C = run_gpu(kernel, grid_fn(N), block, A, B)
            ref = A.astype(np.float64) @ B.astype(np.float64)
            ok = np.allclose(C, ref, atol=1e-3, rtol=1e-3)
            err = float(np.max(np.abs(C - ref)))
            worst = max(worst, err)
            row += f"{'PASS' if ok else 'FAIL':<7}"
            if not ok:
                failures.append((name, N, err))
        print(row + f" {worst:.2e}")
    if verbose:
        if failures:
            print(f"\nGATE FAILED: {len(failures)} case(s): {failures}")
        else:
            print("\nGATE PASSED: all kernels correct incl. non-multiple-of-tile N")
    return len(failures) == 0


if __name__ == "__main__":
    dev = cuda.get_current_device()
    print(f"device: {dev.name} (cc {dev.compute_capability.major}.{dev.compute_capability.minor})")
    sys.exit(0 if correctness_gate() else 1)
