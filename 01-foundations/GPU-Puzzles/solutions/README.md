# GPU-Puzzles 解答与验证

本目录是 [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) 14 道题的独立解答，不修改上游任何文件。

## 文件

| 文件 | 说明 |
| --- | --- |
| `solutions.py` | 14 个 kernel（与上游 `GPU_puzzlers.py` 逐题同签名），唯一解答源 |
| `verify.py` | 17 用例双通道验证（唯一验证入口），不依赖上游 `lib.py` 的 chalk/colour/IPython 绘图链 |
| `.venv/` | GPU 通道专用虚拟环境（不影响全局 Python） |

## 用法

```powershell
# CPU 通道（纯 numpy，无需 numba；全局 Python 即可）
python verify.py --cpu

# GPU 通道（真实编译执行；Quadro RTX 5000 / sm_75 验证通过）
.venv\Scripts\python.exe verify.py --gpu

# 自动：GPU 可用则 GPU，否则回退 CPU
python verify.py          # 或 .venv\Scripts\python.exe verify.py

# 负向自检：故意写错的 kernel 必须被判定 FAIL
python verify.py --selftest
```

退出码：`0` 全部通过；`1` 有失败；`2` 环境错误（如强制 `--gpu` 但不可用）。

## 重建 .venv（GPU 通道）

关键约束：**pip 安装的 CUDA 工具链版本必须 ≤ 显卡驱动版本**。
本机驱动 556.18（CUDA 12.5）。若安装 12.9 组件，NVVM 生成 PTX `.version 8.8`，
驱动 ptxas 仅支持 8.5，报 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`。因此全部组件 pin 到 12.5.x：

```powershell
python -m venv .venv
.venv\Scripts\pip.exe install "numpy<2.2" numba numba-cuda
.venv\Scripts\pip.exe install "nvidia-cuda-nvcc-cu12==12.5.*"     # NVVM（编译）
.venv\Scripts\pip.exe install "nvidia-cuda-nvrtc-cu12==12.5.*"    # NVRTC
.venv\Scripts\pip.exe install "nvidia-cuda-runtime-cu12==12.5.*"  # cudart 运行时
.venv\Scripts\pip.exe install "cuda-bindings<13"                  # 与驱动 12.x 匹配
```

冒烟验证：

```powershell
.venv\Scripts\python.exe -c "from numba import cuda; print(cuda.is_available())"
```

> 备注：`numba` 要求 `numpy<2.2`，故必须在 venv 内使用，与全局 `numpy 2.4.6` 隔离。

## CPU 通道原理（`--cpu`）

block 内每个 CUDA 线程映射为一个 Python 线程；`cuda.shared.array` 返回该
block 所有线程共享的 numpy 数组；`cuda.syncthreads()` 为
`threading.Barrier(nthreads).wait()`，栅栏语义与 CUDA 硬件一致（GIL 与
Barrier 的锁语义保证共享内存可见性）。block 之间顺序执行。含栅栏死锁
超时检测（`syncthreads` 未被所有线程统一到达时判 FAIL，不挂死）。
