# gemm-lab — FP32 SGEMM 手写优化实验（AR002）

Quadro RTX 5000（sm_75）真机上的 GEMM 优化阶梯：**K0 naive → K1 shared tiling → K2 寄存器分块**，与 cuBLAS FP32/FP16 对照。6 kernel、36 基准配置、6 张可解释图（roofline / 机制因果 / tile 扫描 / tensor core 差距）。

- 逐图中文分析：[results.md](results.md)
- 面试导向笔记：[../notes/gemm-lab-notes.md](../notes/gemm-lab-notes.md)
- 需求/设计/验收：`../../specs/changes/AR002-gemm-lab/`

## 为什么是 numba CUDA 而不是 CUDA C++

本机无 nvcc 工具链与 MSVC 编译器（实测仅 ptxas.exe），CUDA C++ 编译路线不通；numba CUDA 在 sm_75 真机可编译执行 shared memory / syncthreads / 寄存器数组，算法结构与 CUDA C 一一对应（`cuda.shared.array` ≡ `__shared__`）。上游 LeetCUDA 的 `.cu` 文件零改动，作为设计参照。

## 文件

| 文件 | 运行环境 | 作用 |
|---|---|---|
| `kernels_numba.py` | venv | 3 级 6 个 kernel + 正确性门（对拍 f64 参考，含非整除 N） |
| `bench_numba.py` | venv | 手写 kernel 基准 → `results/bench_numba.json` |
| `bench_torch.py` | 全局 | cuBLAS FP32/FP16 基线 → `results/bench_torch.json` + `env.json` |
| `plot_results.py` | 全局 | 合并 JSON → `figs/fig1..fig6.png` |

## 复现步骤（Windows PowerShell）

```powershell
# 1) 正确性门（必须全 PASS 才继续）
& "D:\Infra\01-foundations\GPU-Puzzles\solutions\.venv\Scripts\python.exe" kernels_numba.py

# 2) 手写 kernel 基准（venv，约 5 分钟；pinned H2D + 3 遍×5 样本中位数）
& "D:\Infra\01-foundations\GPU-Puzzles\solutions\.venv\Scripts\python.exe" -u bench_numba.py

# 3) cuBLAS 基线（全局 python，约 2 分钟）
python bench_torch.py

# 4) 出图
python plot_results.py
```

## 计时纪律（复现性 <4%，实测最差漂移 3.98%）

1. CUDA Event 计时，**3 遍 × 5 样本 = 15 样本取中位数**
2. 每遍前 ~1s **持续发射 warmup**（显示 GPU 闲时降频，2 次发射拉不回 boost 时钟——冷启动重跑曾漂 25%）
3. 输入用 **pinned memory**（WDDM pageable H2D 实测慢 ~6×）
4. 报中位数不报均值（DWM 后台进程造成尾部离群）

## 一页结论

| 级 | @N=2048 TFLOPS | 达 FP32 峰值 |
|---|---|---|
| K0 naive（@1024） | 0.443 | 4.0% |
| K1 tiled T=16 | 0.885 | 7.9% |
| K2b regblock 64×64 | 2.036 | 18.2% |
| cuBLAS FP32 | 9.468 | 84.5% |
| cuBLAS FP16（tensor core） | 54.872 | 61.5% of 89.2 |

三个断层：手写三级 **3.6×** → cuBLAS **4.65×**（软件工程）→ FP16 TC **5.8×**（硬件路径）。
