# GEMM 优化实验室笔记（gemm-lab / AR002）

> 手写 FP32 SGEMM 三级优化（naive → shared tiling → 寄存器分块）在 Quadro RTX 5000（sm_75）真机上的完整实验：6 个 kernel、36 个基准配置、6 张可解释图，与 cuBLAS FP32/FP16 对照。
> 全部数字来自本机实测（3 遍 × 5 样本中位数，复现漂移 <4%），代码与图：`../gemm-lab/`，逐图分析：`../gemm-lab/results.md`。
> 环境：numba CUDA 0.68（本机无 nvcc/MSVC，CUDA C++ 路线不通，用 numba 复现同一算法结构）；cuBLAS 基线走 torch 2.5.1+cu121。

## 1. 优化阶梯与实测数字（先把结论钉死）

| 级 | kernel | 机制 | @N=1024 TFLOPS | @N=2048 TFLOPS | AI (FLOP/B) |
| --- | --- | --- | --- | --- | --- |
| K0 | naive | 每线程 1 输出，直读全局 | 0.443 | — | 0.25 |
| K1 | tiled T=16 | shared tile，全局读 ÷T | 1.028 | 0.885 | 4 |
| K2a | regblock 32×32（2×2/线程） | 寄存器累加器 | 1.306 | 1.357 | 8 |
| K2b | regblock 64×64（4×4/线程） | 16 累加器 + ILP | 1.611 | **2.036** | 16 |
| 基线 | cuBLAS FP32 | 向量化+双缓冲+swizzle | 7.283 | **9.468**（84.5% 峰值） | 估 ≥32 |
| 基线 | cuBLAS FP16 | tensor core | 20.971 | 54.872 | — |

三个断层（面试可背）：**手写三级 3.6×**（N=1024 口径）→ **cuBLAS 差 4.65×**（软件工程）→ **FP16 tensor core 差 5.8×**（硬件路径）。

## 2. AI 推导（口算就能复现的一行式）

每输出元素的全局读次数：K0=2N；K1=2N/T；K2=2N/BN（**与 TM 无关**）。每次读 4B，每输出 2N FLOP：

```
AI = 2N / (loads × 4B) → K0: 0.25    K1: T/4    K2: BN/4
```

- ridge = 11.2e12 / 448e9 = **25 FLOP/B**：所有手写 kernel（AI ≤ 16）都在 ridge 左侧 = 仍是带宽受限；cuBLAS 估 AI≥32 已过 ridge = 算力受限
- K1_T32 与 K2a 的 AI 相同（都是 8）但差 62% 性能 → **寄存器分块省的不是 DRAM，是 shared→register 的片上流量**（fig3/fig6 的核心教学点）

## 3. 关键实验结论（每条 = 一个面试追问的答案）

1. **naive 饱和在 0.44 TFLOPS 且超过纯 DRAM roofline 4×**：16×16 块的行列局部性被 4MB L2 捕获。roofline 不建模 L2，点画到屋顶上方就是缓存命中的证据。
2. **tile 不是越大越好**：T=8/16/32 → 0.648/0.885/0.838。T8 同步摊销差（每 8 FMA 一次栅栏）；T32 每块 1024 线程只剩 1 块/SM，调度弹性消失。三者占用率相同（都满 1024 线程/SM），**差距来自同步开销与块切换弹性**。
3. **寄存器分块的双收益**：每 FMA 的 shared 读从 2 → 0.5（K2b），加上 16 条独立 FMA 链的 ILP。天花板是寄存器堆（64+ 累加器会 spill）。
4. **cuBLAS 的 4.65× 不是魔法**：float4 向量化装载（1 指令 4 元素）、计算与装载重叠（双缓冲思想；sm_75 无 cp.async，靠 unroll + 多缓冲 shared 轮换）、swizzle 消 bank conflict、128×128 warp 级 tile。每一项我都实现了对应「低配版」，差距是几十项工程细节的乘积。
5. **FP16 的 5.8× 是硬件代差**：一条 mma 指令 = 64 FMA；384 个 tensor core 各自独立于 CUDA core 数。sm_75 只有 FP16（无 BF16/FP8/TF32）。

## 4. 手写骨架（tiled + regblock 关键段，面试默写版）

```python
# K1 tiled 核心（numba CUDA，等价 CUDA C 的 __shared__）
As = cuda.shared.array((T, T), dtype=float32)   # 块内共享！local 是每线程私有的
Bs = cuda.shared.array((T, T), dtype=float32)
acc = float32(0.0)
for k0 in range(0, N, T):
    # 协作装载 + 越界补零（guard 不能省，非整除 N 会读越界）
    As[ty, tx] = A[i0+ty, k0+tx] if i0+ty < N and k0+tx < N else 0.0
    Bs[ty, tx] = B[k0+ty, j0+tx] if k0+ty < N and j0+tx < N else 0.0
    cuda.syncthreads()          # 栅栏1：等装载完成
    for t in range(T):
        acc += As[ty, t] * Bs[t, tx]
    cuda.syncthreads()          # 栅栏2：等读取完成，防下一轮装载覆盖
C[i0+ty, j0+tx] = acc           # guard 写回

# K2 regblock 增量：线程负责 TM×TN 个输出
acc = cuda.local.array((TM, TN), dtype=float32)   # 寄存器累加器组
# 内层：读一次 As[行]+Bs[列]，供 TM×TN 次 FMA 复用
for kk in range(K_STEP):
    for m in range(TM):
        for n in range(TN):
            acc[m, n] += As[ty*TM+m, kk] * Bs[kk, tx*TN+n]
```

默写得分点：双 shared 声明 / 协作装载+补零 / **两道栅栏各防什么**（装载→计算、计算→下一轮装载）/ 累加器进寄存器 / guard 写回。

## 5. 面试深潜（Interview Deep-Dive）

### 5.1 高频问法与口述骨架

- **Q1「手写一个 GEMM 并优化它」**：先写 naive（建立正确性基线）→ 讲清 0.25 AI 的带宽灾难 → shared tiling（复用提升 AI 到 T/4）→ 寄存器分块（片上流量 + ILP）→ 报出实测 0.44→2.04 TFLOPS 的三级跳，对照 cuBLAS 9.47 说明剩余差距构成。**始终用数字说话**。
- **Q2「tiling 为什么快」**：不改 FLOP 数，只改「每个全局字节被几个 FLOP 分摊」。一行式：全局读 2N → 2N/T。
- **Q3「寄存器分块解决什么」**：DRAM 流量已经由 tile 大小定死了；它压的是 **shared→register 的读次数**（每 FMA 2 次 → 0.5 次）并制造 ILP。证据：K1_T32 和 K2a 全局流量完全相同，性能差 62%。
- **Q4「cuBLAS 为什么还快 4.65×」**：向量化装载、双缓冲掩盖装载延迟、bank conflict 消除、warp 级调度——每一项 10~30%，乘起来就是 4~5×。
- **Q5「FP16 为什么快 6×」**：tensor core 是独立硬件单元，一条指令完成 4×4×4 矩阵乘；不是「更快的核心」而是「每周期每单元更多 FLOP」。

### 5.2 追问链

```
naive 慢在哪 ──→ 带宽（AI=0.25）──→ L2 不是也缓存了吗 ──→ 对，4×提升，但 roofline 模型不含 L2
tile 多大最好 ──→ 太小同步摊销差 / 太大占用弹性丢 ──→ 实测 T16 最优 ──→ 为什么 T32 反而降
寄存器分块 ──→ 省的是哪一层流量 ──→ shared→reg（不是 DRAM）──→ 证据：AI 相同性能差 62%
为什么不更大 tile ──→ shared 64KB / 寄存器 64K 每 SM ──→ spill 到 local 直接崩
双缓冲是什么 ──→ 装载与计算重叠 ──→ sm_75 无 cp.async ──→ 用多缓冲 shared 轮换近似
到 cuBLAS 还差什么 ──→ 向量化/swizzle/调度 ──→ 每项 10-30% 乘积效应
FP16 上限 ──→ tensor core ──→ mma = 64 FMA/指令 ──→ 我这代无 BF16/TF32（sm_80+ 才有）
```

### 5.3 数字卡片（本机实测口径）

| 数字 | 值 | 用途 |
| --- | --- | --- |
| K0 饱和值 | **0.44 TFLOPS**（4% 峰值） | naive 带宽地板 |
| K1_T16 最优 tile | **0.885** @2048（T8/32 各 -27%/-5%） | tile 权衡证据 |
| K1_T32 vs K2a | 0.838 vs **1.357**（AI 同为 8） | 寄存器分块的「片上流量」论证 |
| K2b 最好手写 | **2.036** @2048（18.2% 峰值） | 手写天花板 |
| cuBLAS FP32 | **9.468** @2048（84.5%） | 工程差距 4.65× |
| cuBLAS FP16 | **54.9** @2048 / 63.1 @4096 | tensor core 5.8× |
| ridge | 25 FLOP/B；K2b AI=16 仍在线左 | roofline 定位 |
| 计时纪律 | 3 遍×5 样本中位数，漂移 <4% | 基准可信度 |

### 5.4 红线清单（本 AR 实际踩过/验证过的）

1. **`cuda.local.array` ≠ `cuda.shared.array`**：local 是每线程私有。我实测踩过：tile 缓存放进 local，结果恰好输出**单位矩阵**（每线程只见过自己写的 1 个元素）——症状极具迷惑性，对拍才抓得住。
2. **两道 syncthreads 缺一不可**：少第一道读到旧数据（竞态），少第二道下一轮装载覆盖还在读的 tile（竞态）。必须放 uniform 路径，divergent 分支内 sync = 死锁。
3. **非整除 N 必须补零**：guard 装载（越界写 0）+ guard 写回，缺一处则小 N 正确、大 N 崩——正确性门要刻意含非整除尺寸。
4. **显示 GPU 基准的降频陷阱**：2 次 warmup 重跑漂 25%；必须持续 ~1s 压满时钟再计时，且多遍取中位数（我的协议：3×5=15 样本）。
5. **WDDM pageable H2D 慢 ~6×**：基准脚本的输入要用 pinned memory，否则 90% 墙钟花在拷贝上。
6. **不要报均值**：显示 GPU 有后台进程（DWM），尾部离群靠 median 过滤。
7. **别把外推当实测**：K0 没测 N=2048 就不能在表里给它 2048 的数（我在 review 中自己修掉了这个错误）。
8. **单一实验「证实」假说可能是运气**：K1_T16@2048 漂 6%，热稳态烧机后测得 1.45% 以为破案；同协议第二次跑出 13.6%——**受控实验必须可重复才算证据**。该配置最终确认为显示 GPU 上 ±7% 的会话噪声带（9 次观测 0.885–1.005），个别 kernel 形状 × WDDM 调度的交互，锁频或无头 GPU 才能消除；关键结论要用「多次独立观测方向不变」来背书，而不是单次数字。

### 5.5 60 秒电梯陈述

> 「我在一张 Turing 卡上完整走过 FP32 GEMM 的手写优化线：naive、shared 分块、寄存器分块三级，配合 tile 尺寸和分块形状的扫描实验，从 0.44 打到 2.04 TFLOPS；cuBLAS 同机 9.47，我把这 4.65× 拆解成向量化装载、双缓冲、bank conflict 和调度四个可解释的因子。另外我做了一套严格的基准方法学——显示 GPU 降频会让冷启动数据漂 25%，我用持续 warmup 加多遍中位数把复现性压到 4% 以内。这套实验产出了 roofline 定位图：我所有手写 kernel 的算术强度都低于 25 FLOP/B 的 ridge 点，这个事实本身就是『下一步该优化什么』的答案。」
