"""AR004 triton-lab: Triton kernels (tutorial-line reimplementations, sm_75 real GPU).

01-vector-add  -> add_kernel        (BLOCK_SIZE constexpr, masked OOB)
02-fused-softmax -> softmax_kernel  (row-stride loop, num_stages pipelining, -inf fill)
03-matrix-multiplication -> matmul_kernel (GROUP_M swizzle, BM/BN/BK constexpr, fp32 acc)

Deviations from tutorial files (documented, minimal):
- launchers return the compiled-kernel handle so bench can read n_regs/n_spills/PTX
- no triton.testing.do_bench anywhere (self-built CUDA-event protocol, cross-AR comparable)
- matmul kernel keeps the tutorial GROUP_M logic but exposes all tile sizes as constexpr
  so the bench can sweep configs manually (equivalent to @triton.autotune but recordable)
- matmul A/B loads are UNMASKED: bench sizes are powers of two >= 256 and all block sizes
  divide them evenly (documented constraint; store keeps its mask harmlessly)
"""
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x, y, out, BLOCK_SIZE=1024, num_warps=4):
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    return add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = in_ptr + row_idx * in_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        out_row_start_ptr = out_ptr + row_idx * out_row_stride
        tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)


def softmax(x, y, num_warps=8, num_stages=2, num_programs=None):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if num_programs is None:
        num_programs = min(48 * 2, n_rows)
    grid = (num_programs,)
    return softmax_kernel[grid](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                                BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
                                num_warps=num_warps)


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def matmul(a, b, c, BM, BN, BK, GROUP_M=8, num_warps=4, num_stages=2):
    M, K = a.shape
    N = b.shape[1]
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return matmul_kernel[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK, GROUP_M=GROUP_M,
                               num_warps=num_warps, num_stages=num_stages)
