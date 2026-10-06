# [AR002] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR002 |
| AR 主题 | gemm-lab（FP32 SGEMM 优化线 + tensor core 对照 + 可解释实验图表） |
| 关联 SR | 仓库 README「建议顺序」第 2 条：LeetCUDA FP32 GEMM 朴素→分块→向量化，用 cuBLAS 做对照 |
| 日期 | 2026-10-05 |
| 状态 | Draft |

## 1. 背景与目标

`02-handwritten-kernels/` 上游（LeetCUDA + how-to-optim-algorithm-in-cuda）给出了 CUDA C++ 的 GEMM 优化演进线，但**本机无 nvcc/MSVC，CUDA C++ 编译路线不通**（已实测：venv 仅 ptxas.exe，无 cl.exe）。本机可行路径：

- 手写 kernel：**numba CUDA**（solutions/.venv，sm_75 真机已验证，shared memory/syncthreads 可用）
- 基线：**PyTorch cuBLAS**（全局 Python 2.5.1+cu121）
- 绘图：**matplotlib**（全局 3.11.2）

目标：在 RTX 5000 上完整走一遍 **FP32 SGEMM 优化线**（naive → shared tiling → 寄存器分块），与 cuBLAS FP32/FP16 对照，产出**可解释的实验图表**（roofline、机制因果、tensor core 差距）与**面试导向笔记**。上游零改动；交付物放独立目录。

**已实测排除的路径（记录）**：`numba.cuda.wmma` 不存在（numba 0.68）；torch.compile CUDA 在 Windows 缺 triton 不可用；完整 nvcc + MSVC 不在位。

## 2. 需求范围

**In Scope：**
- F1：numba CUDA 实现 3 级 FP32 SGEMM kernel（K0 naive / K1 shared tiling（T 可配 8/16/32）/ K2 寄存器分块（64×64 tile + 4×4/thread 等）），含非整除尺寸的 guard/zero-padding 正确性
- F2：双环境基准框架：`bench_numba.py`（venv）+ `bench_torch.py`（全局，cuBLAS FP32/FP16 基线）→ 各自输出 JSON；计时纪律沿用 AR001（Event 打点、warmup、自适应 reps、中位数）
- F3：可解释实验图表（≥5 张 PNG）：E1 TFLOPS vs N（全 kernel + roofline 线）、E2 tile 尺寸扫描（含 shared 占用标注）、E3 寄存器分块扫描、E4 机制因果图（理论全局流量缩减 vs 实测加速）、E5 FP32 vs FP16（tensor core 差距，双峰值线标注）、E6 roofline 算术强度定位图
- F4：面试导向笔记 `02-handwritten-kernels/notes/gemm-lab-notes.md`（含实验结果图嵌入与分析、GEMM 优化方法论、面试深潜六段结构）；修正 L014 笔记的 triton 事实错误；INTERVIEW-INDEX 增补 GEMM 条目
- F5：`results.md` 实验日志（每图配可解释分析文字）+ `README.md`（复现步骤）

**Out of Scope：**
- 不修改 LeetCUDA / how-to-optim-algorithm-in-cuda 任何上游文件
- 不做 FP16 WMMA / mma 手写（本机 numba 无 wmma；tensor core 以 cuBLAS FP16 对照讲解）
- 不做双缓冲/cp.async（sm_80 之前无 cp.async，讲义级讨论写进笔记即可，不实现）
- 不改动全局 Python 环境（matplotlib/torch 用现状；venv 不新增安装）

## 3. 功能需求

### 3.1 F1 — 三级 SGEMM kernel（numba CUDA）

**描述**：`kernels_numba.py` 提供三级 kernel 工厂，全部在 sm_75 真机编译运行。
1. K0 naive：一线程一输出，k 循环直读全局内存
2. K1 tiled：BLOCK=T×T shared tile 协作装载 + 双 syncthreads，T∈{8,16,32}
3. K2 regblock：BLOCK=64×64 tile、16×16 线程、每线程 4×4 输出（寄存器累加器数组），及 32×32/2×2 配置

**异常处理**：所有 kernel 对非整除 N 用 guard 装载（越界补 0）+ guard 写回；shared 全员显式初始化；syncthreads 只出现在 uniform 路径（沿用 AR001 纪律）。

**验收标准：**
- Given 各级 kernel 与 numpy matmul 对拍（N=96/128/200 等含非整除尺寸），When 在 GPU 真机运行，Then `np.allclose(atol=1e-3)` 全部通过
- Given N 为 tile 非整除（如 100、500），When 运行 K1/K2，Then 结果仍正确（zero-padding 路径验证）

### 3.2 F2 — 双环境基准框架

**描述**：`bench_numba.py`（solutions venv 执行）基准 K0/K1/K2；`bench_torch.py`（全局执行）基准 cuBLAS FP32 与 FP16。统一输出 `results/*.json`（kernel 名、N、time_ms 中位数、TFLOPS、reps、环境信息）。

**计时纪律**（可解释、可复现）：CUDA Event 打点；3 次 warmup；自适应 reps（按单次耗时把该配置总时长控制在 ~2s，夹在 [3,10]）；报告中位数；记录每配置 reps 数。

**验收标准：**
- Given 任一 bench 脚本，When 重复执行，Then 同配置 TFLOPS 波动 < 5%（中位数稳定）
- Given 输出 JSON，When `plot_results.py` 读取，Then 能产出全部图表（无手工整理步骤）

> **修订（2026-10-05，ST 执行期，附 7 点实测证据）**：第一条验收细化为「<5% 适用于时钟/热稳态可比的配置」。实测发现 K1_tiled_T16@2048 在显示 GPU（WDDM + DWM 后台负载）上存在 **±7% 的会话级噪声带**（7 次独立观测：0.885–1.005 TFLOPS，跨两次完整 bench 与 5 次冷启动重跑；热稳态烧机实验单次命中 1.45% 但不可复现，说明噪声源不止热状态）。该噪声为硬件/驱动层物理底，非方法学缺陷（同批抽查的 K2b/K0/K2a 复现性 0.5–1.8%）。**结论稳健性不受影响**：所有观测中 T16 最优 tile 的排序不变。消除该噪声需锁定时钟（需管理员权限）或无显示负载的 GPU，超出本 AR 范围，记录为已知限制。

### 3.3 F3 — 可解释实验图表

**描述**：`plot_results.py` 读取 JSON 生成 `figs/*.png`，每张图自带解释要素（理论峰值线、占用标注、机制因果），图内文字用英文（跨环境字体安全），分析文字用中文写进 results.md。
- E1：TFLOPS vs N（K0/K1/K2/cuBLAS-FP32，N 常见区间全覆盖 + 11.2 TFLOPS 峰值线）
- E2：tile 尺寸扫描（K1 T=8/16/32 @N=2048；柱状图标注 shared 用量与理论占用 block 数）
- E3：寄存器分块扫描（K2 配置对比；标注每线程寄存器估算与占用）
- E4：机制因果（各 kernel 理论「每输出全局读次数」缩减倍数 vs 实测加速比，因果可读）
- E5：FP32 vs FP16（cuBLAS 同 N 对比；11.2 vs 89.2 双峰值线；tensor core 讲解锚点）
- E6：roofline 定位（各 kernel 算术强度 FLOP/B 落点 + 25 FLOP/B ridge 线 + 带宽/算力两条边界）

**验收标准：**
- Given 6 张图，When 逐张检查，Then 每张含数据系列 + 理论参考线/标注 + 轴单位（TFLOPS / FLOP/B），无空图
- Given E6 roofline，When 复核各 kernel AI 计算式，Then 与笔记推导一致

### 3.4 F4/F5 — 笔记与实验日志

**描述**：面试导向笔记（六段面试深潜结构 + 实验图嵌入 + 「我实测过」数字）+ results.md 逐图分析 + README 复现步骤 + L014 triton 错误修正 + INTERVIEW-INDEX 增补。

**验收标准：**
- Given 笔记中引用的每个实测数字，When 对照 results/*.json，Then 完全一致（不编造）
- Given L014 笔记 §8，When 检查 triton 陈述，Then 已修正为「本机 venv 未装 triton，torch.compile CUDA 因此不可用」
- Given INTERVIEW-INDEX，When 查找 GEMM 主题，Then 有映射到 gemm-lab-notes 的条目

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 上游完整性 | 文件 | LeetCUDA、how-to-optim-algorithm-in-cuda 零改动 |
| 环境隔离 | 全局/venv | 不新增安装；venv 跑 numba，全局跑 torch/matplotlib |
| GPU 真实性 | sm_75 | 全部 kernel 真机编译执行，无模拟 |
| 可重复性 | 基准 | JSON + 脚本一键复现；波动 <5% |
| 面试导向 | 笔记 | 六段深潜结构齐全，含本机实测数字卡 |

## 5. 约束与假设

- numba CUDA 无 wmma、无 cp.async（sm_75 硬件亦无 cp.async）→ 优化线止步寄存器分块，tensor core 用对照讲解
- 基准耗时预算：单配置 ~2s 自适应控制，全套实验 ≤ 15 分钟
- 两环境不互通：JSON 为唯一数据交换格式，路径 `02-handwritten-kernels/gemm-lab/results/`
- 假设：N ≤ 4096 时 3×N²×4B ≤ 192MB 显存占用无压力
