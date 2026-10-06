"""GPU-Puzzles (srush/GPU-Puzzles) 14 道 kernel 的独立解答。

- 每个工厂函数与上游 GPU_puzzlers.py 中的同名函数签名一致：
  ``def xxx_test(cuda) -> callable``，返回 ``call(out, *inputs, *args)``。
- kernel 内遵守 numba CUDA 限制：不使用 numpy 属性/列表推导；
  shared memory 尺寸为字面常量；所有线程统一到达 ``cuda.syncthreads()``。
- shared 数组所有元素在 sync 前显式初始化（GPU 上 shared 内容未定义，
  不能依赖零值）。
- 本文件可在无 numba 环境下被 verify.py 的 CPU 通道执行：``cuda`` 由调用方
  注入（numba.cuda 或 CPU 模拟对象）；kernel 内的 ``numba.float32`` 仅为
  dtype 标记，numba 缺失时由下方 shim 提供同名占位。
"""

try:
    import numba
except ImportError:  # CPU 纯 numpy 通道：float32 仅作占位标记

    class _NumbaShim:
        float32 = "float32-shim"

    numba = _NumbaShim()

TPB = 8
MAX_CONV = 4
TPB_MAX_CONV = TPB + MAX_CONV


# ---------------------------- Puzzle 1: Map ----------------------------
def map_test(cuda):
    def call(out, a) -> None:
        local_i = cuda.threadIdx.x
        out[local_i] = a[local_i] + 10

    return call


# ---------------------------- Puzzle 2: Zip ----------------------------
def zip_test(cuda):
    def call(out, a, b) -> None:
        local_i = cuda.threadIdx.x
        out[local_i] = a[local_i] + b[local_i]

    return call


# --------------------------- Puzzle 3: Guards ---------------------------
def map_guard_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        if local_i < size:
            out[local_i] = a[local_i] + 10

    return call


# -------------------------- Puzzle 4: Map 2D ---------------------------
def map_2D_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        if local_i < size and local_j < size:
            out[local_i, local_j] = a[local_i, local_j] + 10

    return call


# ------------------------- Puzzle 5: Broadcast -------------------------
def broadcast_test(cuda):
    def call(out, a, b, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        if local_i < size and local_j < size:
            out[local_i, local_j] = a[local_i, 0] + b[0, local_j]

    return call


# --------------------------- Puzzle 6: Blocks --------------------------
def map_block_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < size:
            out[i] = a[i] + 10

    return call


# ------------------------- Puzzle 7: Blocks 2D -------------------------
def map_block2D_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        if i < size and j < size:
            out[i, j] = a[i, j] + 10

    return call


# --------------------------- Puzzle 8: Shared --------------------------
TPB8 = 4


def shared_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB8, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        shared[local_i] = a[i] if i < size else 0.0
        cuda.syncthreads()
        if i < size:
            out[i] = shared[local_i] + 10

    return call


# -------------------------- Puzzle 9: Pooling --------------------------
def pool_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        shared[local_i] = a[i] if i < size else 0.0
        cuda.syncthreads()
        if i < size:
            acc = shared[local_i]
            if local_i >= 1:
                acc += shared[local_i - 1]
            if local_i >= 2:
                acc += shared[local_i - 2]
            out[i] = acc

    return call


# ------------------------ Puzzle 10: Dot Product -----------------------
def dot_test(cuda):
    def call(out, a, b, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        shared[local_i] = a[i] * b[i] if i < size else 0.0
        cuda.syncthreads()
        shared[local_i] += shared[local_i + 4] if local_i < 4 else 0.0
        cuda.syncthreads()
        shared[local_i] += shared[local_i + 2] if local_i < 2 else 0.0
        cuda.syncthreads()
        shared[local_i] += shared[local_i + 1] if local_i < 1 else 0.0
        cuda.syncthreads()
        if i == 0:
            out[0] = shared[0]

    return call


# ------------------------ Puzzle 11: 1D Convolution --------------------
def conv_test(cuda):
    def call(out, a, b, a_size, b_size) -> None:
        shared = cuda.shared.array(TPB_MAX_CONV, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        shared[local_i] = a[i] if i < a_size else 0.0
        if local_i < MAX_CONV:
            if i + TPB < a_size:
                shared[TPB + local_i] = a[i + TPB]
            else:
                shared[TPB + local_i] = 0.0
        cuda.syncthreads()

        if i < a_size:
            acc = 0.0
            for j in range(b_size):
                if i + j < a_size:
                    acc += shared[local_i + j] * b[j]
            out[i] = acc

    return call


# ------------------------- Puzzle 12: Prefix Sum -----------------------
def sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        cache[local_i] = a[i] if i < size else 0.0
        cuda.syncthreads()
        skip = TPB // 2
        while skip > 0:
            if local_i < skip:
                cache[local_i] += cache[local_i + skip]
            cuda.syncthreads()
            skip = skip // 2
        if local_i == 0:
            out[cuda.blockIdx.x] = cache[0]

    return call


# -------------------------- Puzzle 13: Axis Sum ------------------------
def axis_sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        batch = cuda.blockIdx.y

        cache[local_i] = a[batch, local_i] if local_i < size else 0.0
        cuda.syncthreads()
        cache[local_i] += cache[local_i + 4] if local_i < 4 else 0.0
        cuda.syncthreads()
        cache[local_i] += cache[local_i + 2] if local_i < 2 else 0.0
        cuda.syncthreads()
        cache[local_i] += cache[local_i + 1] if local_i < 1 else 0.0
        cuda.syncthreads()
        if local_i == 0:
            out[batch, 0] = cache[0]

    return call


# ------------------------ Puzzle 14: Matrix Multiply -------------------
TPB3 = 3


def mm_oneblock_test(cuda):
    def call(out, a, b, size: int) -> None:
        a_shared = cuda.shared.array((TPB3, TPB3), numba.float32)
        b_shared = cuda.shared.array((TPB3, TPB3), numba.float32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y

        acc = 0.0
        for k in range(0, size, TPB3):
            if i < size and k + local_j < size:
                a_shared[local_i, local_j] = a[i, k + local_j]
            else:
                a_shared[local_i, local_j] = 0.0
            if k + local_i < size and j < size:
                b_shared[local_i, local_j] = b[k + local_i, j]
            else:
                b_shared[local_i, local_j] = 0.0
            cuda.syncthreads()
            for l in range(TPB3):
                acc += a_shared[local_i, l] * b_shared[l, local_j]
            cuda.syncthreads()
        if i < size and j < size:
            out[i, j] = acc

    return call


ALL_KERNELS = {
    "map_test": map_test,
    "zip_test": zip_test,
    "map_guard_test": map_guard_test,
    "map_2D_test": map_2D_test,
    "broadcast_test": broadcast_test,
    "map_block_test": map_block_test,
    "map_block2D_test": map_block2D_test,
    "shared_test": shared_test,
    "pool_test": pool_test,
    "dot_test": dot_test,
    "conv_test": conv_test,
    "sum_test": sum_test,
    "axis_sum_test": axis_sum_test,
    "mm_oneblock_test": mm_oneblock_test,
}
