"""GPU-Puzzles 14 题 17 用例的双通道验证脚本。

通道：
- GPU：复刻上游 lib.py ``CudaProblem.run_cuda`` 的调用方式
  （``numba.cuda.jit(fn)[grid, block](out, *inputs, *args)``），
  在本机 NVIDIA GPU 上真实编译执行。不依赖 chalk/colour/IPython。
- CPU：栅栏模拟器——block 内每个 CUDA 线程映射为一个 Python 线程，
  ``cuda.shared.array`` 返回 block 内共享的 numpy 数组，
  ``cuda.syncthreads()`` 为 ``threading.Barrier.wait()``，
  栅栏语义与 CUDA 硬件一致；block 之间顺序执行。

用法：
- python verify.py            自动：GPU 可用则 GPU，否则回退 CPU
- python verify.py --gpu      强制 GPU（不可用则报错退出，不静默降级）
- python verify.py --cpu      强制 CPU（不触碰 numba）
- python verify.py --selftest 负向自检：故意写错的 kernel 必须被判定 FAIL

退出码：0 = 全部通过；1 = 有失败；2 = 环境错误（如强制 GPU 但不可用）。
"""

from __future__ import annotations

import argparse
import sys
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

import solutions

# ---------------------------------------------------------------------------
# 上游数据结构复刻
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coord:
    x: int
    y: int

    def tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


# ---------------------------------------------------------------------------
# 上游 spec 函数逐字复刻（GPU_puzzlers.py）
# ---------------------------------------------------------------------------


def map_spec(a):
    return a + 10


def zip_spec(a, b):
    return a + b


def pool_spec(a):
    out = np.zeros(*a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    return out


def dot_spec(a, b):
    return a @ b


def conv_spec(a, b):
    out = np.zeros(*a.shape)
    len = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
    return out


TPB = 8


def sum_spec(a):
    out = np.zeros((a.shape[0] + TPB - 1) // TPB)
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[j] = a[i : i + TPB].sum()
    return out


def axis_sum_spec(a):
    out = np.zeros((a.shape[0], (a.shape[1] + TPB - 1) // TPB))
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[..., j] = a[..., i : i + TPB].sum(-1)
    return out


def matmul_spec(a, b):
    return a @ b


# ---------------------------------------------------------------------------
# 用例定义（与上游逐题一致，共 17 例）
# ---------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    fn: Callable[[Any], Callable[..., None]]
    inputs: List[np.ndarray]
    out: np.ndarray
    args: Tuple[int, ...]
    blockspergrid: Coord
    threadsperblock: Coord
    spec: Callable[..., np.ndarray]


def build_cases() -> List[Case]:
    cases: List[Case] = []

    # Puzzle 1: Map
    SIZE = 4
    cases.append(
        Case(
            "Map",
            solutions.map_test,
            [np.arange(SIZE)],
            np.zeros((SIZE,)),
            (),
            Coord(1, 1),
            Coord(SIZE, 1),
            map_spec,
        )
    )

    # Puzzle 2: Zip
    SIZE = 4
    cases.append(
        Case(
            "Zip",
            solutions.zip_test,
            [np.arange(SIZE), np.arange(SIZE)],
            np.zeros((SIZE,)),
            (),
            Coord(1, 1),
            Coord(SIZE, 1),
            zip_spec,
        )
    )

    # Puzzle 3: Guards
    SIZE = 4
    cases.append(
        Case(
            "Guard",
            solutions.map_guard_test,
            [np.arange(SIZE)],
            np.zeros((SIZE,)),
            (SIZE,),
            Coord(1, 1),
            Coord(8, 1),
            map_spec,
        )
    )

    # Puzzle 4: Map 2D
    SIZE = 2
    cases.append(
        Case(
            "Map 2D",
            solutions.map_2D_test,
            [np.arange(SIZE * SIZE).reshape((SIZE, SIZE))],
            np.zeros((SIZE, SIZE)),
            (SIZE,),
            Coord(1, 1),
            Coord(3, 3),
            map_spec,
        )
    )

    # Puzzle 5: Broadcast
    SIZE = 2
    cases.append(
        Case(
            "Broadcast",
            solutions.broadcast_test,
            [np.arange(SIZE).reshape(SIZE, 1), np.arange(SIZE).reshape(1, SIZE)],
            np.zeros((SIZE, SIZE)),
            (SIZE,),
            Coord(1, 1),
            Coord(3, 3),
            zip_spec,
        )
    )

    # Puzzle 6: Blocks
    SIZE = 9
    cases.append(
        Case(
            "Blocks",
            solutions.map_block_test,
            [np.arange(SIZE)],
            np.zeros((SIZE,)),
            (SIZE,),
            Coord(3, 1),
            Coord(4, 1),
            map_spec,
        )
    )

    # Puzzle 7: Blocks 2D
    SIZE = 5
    cases.append(
        Case(
            "Blocks 2D",
            solutions.map_block2D_test,
            [np.ones((SIZE, SIZE))],
            np.zeros((SIZE, SIZE)),
            (SIZE,),
            Coord(2, 2),
            Coord(3, 3),
            map_spec,
        )
    )

    # Puzzle 8: Shared
    SIZE = 8
    cases.append(
        Case(
            "Shared",
            solutions.shared_test,
            [np.ones(SIZE)],
            np.zeros(SIZE),
            (SIZE,),
            Coord(2, 1),
            Coord(4, 1),
            map_spec,
        )
    )

    # Puzzle 9: Pooling
    SIZE = 8
    cases.append(
        Case(
            "Pooling",
            solutions.pool_test,
            [np.arange(SIZE)],
            np.zeros(SIZE),
            (SIZE,),
            Coord(1, 1),
            Coord(8, 1),
            pool_spec,
        )
    )

    # Puzzle 10: Dot Product
    SIZE = 8
    cases.append(
        Case(
            "Dot",
            solutions.dot_test,
            [np.arange(SIZE), np.arange(SIZE)],
            np.zeros(1),
            (SIZE,),
            Coord(1, 1),
            Coord(SIZE, 1),
            dot_spec,
        )
    )

    # Puzzle 11: 1D Convolution（Test 1 / Test 2）
    SIZE, CONV = 6, 3
    cases.append(
        Case(
            "1D Conv (Simple)",
            solutions.conv_test,
            [np.arange(SIZE), np.arange(CONV)],
            np.zeros(SIZE),
            (SIZE, CONV),
            Coord(1, 1),
            Coord(8, 1),
            conv_spec,
        )
    )
    cases.append(
        Case(
            "1D Conv (Full)",
            solutions.conv_test,
            [np.arange(15), np.arange(4)],
            np.zeros(15),
            (15, 4),
            Coord(2, 1),
            Coord(8, 1),
            conv_spec,
        )
    )

    # Puzzle 12: Prefix Sum（Test 1 / Test 2）
    SIZE = 8
    cases.append(
        Case(
            "Sum (Simple)",
            solutions.sum_test,
            [np.arange(SIZE)],
            np.zeros(1),
            (SIZE,),
            Coord(1, 1),
            Coord(TPB, 1),
            sum_spec,
        )
    )
    cases.append(
        Case(
            "Sum (Full)",
            solutions.sum_test,
            [np.arange(15)],
            np.zeros(2),
            (15,),
            Coord(2, 1),
            Coord(TPB, 1),
            sum_spec,
        )
    )

    # Puzzle 13: Axis Sum
    BATCH, SIZE = 4, 6
    cases.append(
        Case(
            "Axis Sum",
            solutions.axis_sum_test,
            [np.arange(BATCH * SIZE).reshape((BATCH, SIZE))],
            np.zeros((BATCH, 1)),
            (SIZE,),
            Coord(1, BATCH),
            Coord(TPB, 1),
            axis_sum_spec,
        )
    )

    # Puzzle 14: Matrix Multiply（Test 1 / Test 2）
    SIZE = 2
    cases.append(
        Case(
            "Matmul (Simple)",
            solutions.mm_oneblock_test,
            [
                np.arange(SIZE * SIZE).reshape((SIZE, SIZE)),
                np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T,
            ],
            np.zeros((SIZE, SIZE)),
            (SIZE,),
            Coord(1, 1),
            Coord(3, 3),
            matmul_spec,
        )
    )
    SIZE = 8
    cases.append(
        Case(
            "Matmul (Full)",
            solutions.mm_oneblock_test,
            [
                np.arange(SIZE * SIZE).reshape((SIZE, SIZE)),
                np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T,
            ],
            np.zeros((SIZE, SIZE)),
            (SIZE,),
            Coord(3, 3),
            Coord(3, 3),
            matmul_spec,
        )
    )

    return cases


# ---------------------------------------------------------------------------
# CPU 通道：真线程 + Barrier 栅栏模拟器
# ---------------------------------------------------------------------------


class _SharedNamespace:
    """模拟 ``cuda.shared``：``array(shape, dtype)`` 返回 block 级共享数组。"""

    def __init__(self, block_ctx: "_BlockCtx", thread_ctx: "_ThreadCtx"):
        self._block = block_ctx
        self._thread = thread_ctx

    def array(self, shape, dtype):  # dtype 仅占位，统一 float64 承载
        return self._block._shared_array(shape, self._thread._alloc_count())


class _BlockCtx:
    """一个 block 的执行上下文：共享数组池 + 线程栅栏。"""

    def __init__(self, block_idx: Coord, block_dim: Coord):
        self.blockIdx = block_idx
        self.blockDim = block_dim
        self._pool: List[np.ndarray] = []
        self._lock = threading.Lock()
        self.barrier = threading.Barrier(block_dim.x * block_dim.y)

    def _shared_array(self, shape, alloc_idx: int) -> np.ndarray:
        if isinstance(shape, int):
            shape = (shape,)
        with self._lock:
            while len(self._pool) <= alloc_idx:
                self._pool.append(np.zeros(shape, dtype=np.float64))
            return self._pool[alloc_idx]


class _ThreadCtx:
    """一个 CUDA 线程的模拟 cuda 对象。"""

    def __init__(self, block_ctx: _BlockCtx, thread_idx: Coord):
        self.threadIdx = thread_idx
        self.blockIdx = block_ctx.blockIdx
        self.blockDim = block_ctx.blockDim
        self.shared = _SharedNamespace(block_ctx, self)
        self._allocs = 0
        self._barrier = block_ctx.barrier

    def _alloc_count(self) -> int:
        idx = self._allocs
        self._allocs += 1
        return idx

    def syncthreads(self) -> None:
        self._barrier.wait()


def _cpu_worker(
    call: Callable[..., None],
    out: np.ndarray,
    inputs: Sequence[np.ndarray],
    args: Tuple[int, ...],
    errors: List[BaseException],
) -> None:
    try:
        call(out, *inputs, *args)
    except BaseException as exc:  # 线程内异常必须传出，不得静默
        errors.append(exc)


def cpu_run_case(case: Case, timeout: float = 30.0) -> np.ndarray:
    out = case.out.copy()
    inputs = [a.copy() for a in case.inputs]
    errors: List[BaseException] = []
    bpg, tpb = case.blockspergrid, case.threadsperblock

    for bx in range(bpg.x):
        for by in range(bpg.y):
            block_ctx = _BlockCtx(Coord(bx, by), tpb)
            threads: List[threading.Thread] = []
            for tx in range(tpb.x):
                for ty in range(tpb.y):
                    thread_ctx = _ThreadCtx(block_ctx, Coord(tx, ty))
                    call = case.fn(thread_ctx)
                    t = threading.Thread(
                        target=_cpu_worker, args=(call, out, inputs, case.args, errors)
                    )
                    threads.append(t)
            for t in threads:
                t.start()
            deadline = timeout
            for t in threads:
                t.join(timeout=max(deadline, 0.1))
            if any(t.is_alive() for t in threads):
                # 栅栏未对齐（syncthreads 未被所有线程统一到达）→ 打破栅栏
                block_ctx.barrier.abort()
                for t in threads:
                    t.join(timeout=1.0)
                raise TimeoutError(
                    "CPU emulator barrier deadlock: syncthreads must be called "
                    "uniformly by all threads in the block"
                )
            if errors:
                raise errors[0]
    return out


# ---------------------------------------------------------------------------
# GPU 通道：numba.cuda 真机编译执行（与 lib.py run_cuda 一致）
# ---------------------------------------------------------------------------


def gpu_status() -> Tuple[bool, str]:
    """探测 GPU 通道可用性，返回 (可用, 诊断信息)。"""
    try:
        import numba
        from numba import cuda
    except ImportError as exc:
        return False, f"numba 未安装（{exc}）；请在 solutions/.venv 中安装"
    try:
        if not cuda.is_available():
            return False, "numba.cuda.is_available() == False（无可用 CUDA 设备）"
    except Exception as exc:
        return False, f"cuda.is_available() 探测失败：{exc}"
    return True, "numba.cuda 可用"


def gpu_run_case(case: Case) -> np.ndarray:
    import numba
    from numba import cuda

    out = case.out.copy()
    inputs = [a.copy() for a in case.inputs]
    fn = case.fn(cuda)
    with warnings.catch_warnings():
        # 小规模 puzzle 的 host-array 拷贝与低占用告警属预期，抑制之；
        # 真正的失败以异常形式抛出，不受影响
        warnings.simplefilter("ignore")
        jitfn = numba.cuda.jit(fn)
        jitfn[case.blockspergrid.tuple(), case.threadsperblock.tuple()](
            out, *inputs, *case.args
        )
        cuda.synchronize()
    return out


# ---------------------------------------------------------------------------
# 执行与报告
# ---------------------------------------------------------------------------


def check_case(case: Case, runner: Callable[[Case], np.ndarray]) -> Tuple[bool, str]:
    out = runner(case)
    expected = case.spec(*[a.copy() for a in case.inputs])
    try:
        np.testing.assert_allclose(out, expected)
        return True, ""
    except AssertionError as exc:
        return False, f"Yours: {np.array2string(np.asarray(out), precision=4)}\nSpec : {np.array2string(np.asarray(expected), precision=4)}"


def run_all(cases: Sequence[Case], runner: Callable[[Case], np.ndarray], label: str) -> int:
    failures: List[str] = []
    for case in cases:
        try:
            ok, detail = check_case(case, runner)
        except Exception as exc:
            ok, detail = False, f"execution error: {type(exc).__name__}: {exc}"
        status = "PASS" if ok else "FAIL"
        print(f"[{label}] {case.name:<18} {status}")
        if not ok:
            failures.append(case.name)
            for line in detail.splitlines():
                print(f"        {line}")
    print(f"[{label}] summary: {len(cases) - len(failures)}/{len(cases)} passed")
    if failures:
        print(f"[{label}] failed cases: {', '.join(failures)}")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# 负向自检：故意写错的 kernel 必须被判定 FAIL
# ---------------------------------------------------------------------------


def _bad_map_test(cuda):
    def call(out, a) -> None:
        out[cuda.threadIdx.x] = a[cuda.threadIdx.x] + 20  # 错：+20 而非 +10

    return call


def _bad_guard_test(cuda):
    def call(out, a, size) -> None:
        # 错：去掉 guard，线程数 8 > 数据量 4 → 越界写
        out[cuda.threadIdx.x] = a[cuda.threadIdx.x] + 10

    return call


def selftest() -> int:
    cases = build_cases()
    map_case = next(c for c in cases if c.name == "Map")
    guard_case = next(c for c in cases if c.name == "Guard")
    bad_map = Case(
        map_case.name, _bad_map_test, map_case.inputs, map_case.out,
        map_case.args, map_case.blockspergrid, map_case.threadsperblock, map_case.spec,
    )
    bad_guard = Case(
        guard_case.name, _bad_guard_test, guard_case.inputs, guard_case.out,
        guard_case.args, guard_case.blockspergrid, guard_case.threadsperblock,
        guard_case.spec,
    )
    print("== Negative selftest (CPU channel): broken kernels MUST be detected ==")
    results = []
    for case in (bad_map, bad_guard):
        try:
            ok, detail = check_case(case, cpu_run_case)
        except Exception as exc:
            ok, detail = False, f"execution error: {type(exc).__name__}: {exc}"
        detected = not ok
        results.append(detected)
        print(
            f"[selftest] {case.name:<18} detected: {'YES' if detected else 'NO'}"
            + ("" if detected else " (SELFTEST FAILED: broken kernel not caught!)")
        )
        if not detected:
            print(f"        {detail}")
    all_detected = all(results)
    print(
        f"[selftest] result: {'PASS - harness catches broken kernels' if all_detected else 'FAIL'}"
    )
    return 0 if all_detected else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU-Puzzles 双通道验证")
    parser.add_argument("--gpu", action="store_true", help="强制 GPU 通道")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU 通道")
    parser.add_argument("--selftest", action="store_true", help="负向自检")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    cases = build_cases()

    if args.gpu and args.cpu:
        print("Error: --gpu and --cpu are mutually exclusive")
        return 2

    if args.gpu:
        ok, diag = gpu_status()
        if not ok:
            print(f"GPU channel unavailable: {diag}")
            print("Hint: create a venv under solutions/ and install numba (see README.md), or use --cpu")
            return 2
        return run_all(cases, gpu_run_case, "GPU")

    if args.cpu:
        return run_all(cases, cpu_run_case, "CPU")

    # 自动模式：GPU 可用则 GPU，否则回退 CPU
    ok, diag = gpu_status()
    if ok:
        print(f"Auto-selected GPU channel ({diag})")
        return run_all(cases, gpu_run_case, "GPU")
    print(f"GPU channel unavailable ({diag}); falling back to CPU channel")
    return run_all(cases, cpu_run_case, "CPU")


if __name__ == "__main__":
    sys.exit(main())
