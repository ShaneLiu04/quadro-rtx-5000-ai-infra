# GEMM 形状空间与 CUTLASS 拆解——面试深潜笔记（AR003 gemm-sweep）

> 承接 gemm-lab-notes（AR002：手写优化线 + cuBLAS 方阵单点）。本篇回答下一层问题：
> **cuBLAS 在形状空间里怎么表现？库启发式的边界在哪？CUTLASS 的模板怎么表达这些自由度？**
> 全部数字为本机实测（`03-gemm/gemm-sweep/results/*.json`），CUTLASS 部分为源码拆解
> （`03-gemm/notes/cutlass-turing-dissection.md`，本机无 nvcc 不编译，只拆解）。

---

## 1. 高频问法（面试官怎么问）

1. 「cuBLAS 是不是所有形状都快？」→ 不是。**形状空间极不均匀**：64³ 只有 0.54 TF（4.8% 峰值，launch 主导），M,N≥512 区 16 格中 11 格 9.5-10.24 TF（85-91% 峰值，中位 9.71），但有 3 格「启发式凹陷」只有 8.58-8.90（(512,512)=8.58、(1024,512)=8.83、(1024,4096)=8.90）。「cuBLAS 快」默认指饱和区，且饱和区**并非没有洞**。
2. 「decode 场景 M=1 的 GEMM，瓶颈是什么？」→ 带宽，不是算力。M≤32 时实测 360-386 GB/s（HBM 的 80-86%），此时报 TFLOPS 没有意义。
3. 「为什么我的小输出大 K 矩阵乘还是很快？不是说 tile 少喂不满 SM 吗？」→ 库启发式**内部自动 split-K**。实测 256×256×16384 单发 8.2 TF，而 4-tile 饥饿模型上限只有 0.93 TF——差 8.8 倍就是证据。
4. 「那我自己切 K 用多流并发会不会更快？」→ 实测全败（s=2/4/8/16 → 6.8/5.6/3.7/2.0 TF，单调变差）。每路小 GEMM 又回饥饿态 + 归并开销；**E3seq 顺序对照实测多流反而比纯顺序还慢 -9.8%~-43.9%**——WDDM 跨流提交/事件同步是净开销，不是「打折」而是倒贴。
5. 「batch GEMM 是不是天然并行，比展平好？」→ 实测 flat 一致赢（B=64: 9.75 vs 8.61 TF，+13%）。bmm 的组间调度是纯开销，只有「无法堆叠/必须独立权重」时才用它。
6. 「CUTLASS 的 tile 层级怎么读？为什么是 2 级流水？」→ 三级 tile（128×256×64 → 64×64×64 → m8n8k16）；INT8 tile 双缓冲 = 48KB，Turing shared 64KB——**NumStages=2 是内存预算的必然**，不是选择。

## 2. 追问链（连环问路线图）

```
方阵峰值 ──→ 别的形状呢 ──→ 小形状为何崩（launch 主导，0.54 TF 地板）
           ──→ skinny 为何慢 ──→ 带宽平台 vs 算力平台 ──→ 拐点在哪（AI=25 → M*≈51，实测 32~64 之间）
                              ──→ M-skinny 和 N-skinny 一样吗（不一样：7.97 vs 5.92 TF，row-major 输出的列窄写回吃亏）
           ──→ wave 量化 ──→ 什么是坏形状（跨波不满：M=160 比 128/256 都低 ~10%）
                          ──→ 多少波后没事（≥4 波，损失 <5%）
split-K ──→ 何时需要（tile 数 << SM 数）──→ 怎么验证库已做（饥饿模型 0.93 vs 实测 8.2 = 8.8× 差）
        ──→ 手工为何输（小 GEMM 复饥饿 + 归并开销；多流实测比顺序还慢——E3seq 对照）
        ──→ CUTLASS 怎么暴露它（split_k_slices 模板参数 + workspace + reduction kernel = 显式化）
layout ──→ TN 是不是最快（本机 7 组合极差仅 5%：ldmatrix+swizzle 把布局差异吃掉了）
       ──→ 那 CUTLASS 默认为何是 TN（mma fragment 装载天然对齐；规范层偏好 ≠ 性能层差异）
       ──→ 为对齐做 transpose 划算吗（必亏：copy 成本 >> 3% 核内差）
dtype ──→ FP16 提速多少（6.4-6.9×@2048 两次观测，接近硬件 8×）──→ 为何达成率只有 ~70%（TC 更难喂 + 时钟态敏感）
      ──→ INT8 呢（硬件有 2× FP16；本机 Windows torch 缺 cublasLt INT8 路径没测成——诚实说明栈限制）
      ──→ FP16 数字为什么时高时低（跨会话 54.9~67.4：烧机程度决定时钟态；FP32 大形状仅 ±1.5%，且噪声带宽度 ∝ 测量窗口长度）
CUTLASS ──→ 模板参数怎么读（15 个：类型/布局/三级 tile/epilogue/swizzle/stages）
        ──→ NumStages 为何是 2（INT8 tile 24KB×2=48KB ≤ 64KB；3 级 72KB 超限——一个参数背后是整条内存预算）
        ──→ Turing 与 Ampere 差在哪（无 cp.async → global→寄存器→shared 两跳；shared 64KB vs 164KB）
```

## 3. 数字卡片（必背，全部本机实测）

### 3.1 形状空间（FP32，cuBLAS）

| 断言 | 数字 |
|------|------|
| 饱和区 | **M,N≥512 区 16 格中 11 格 9.5-10.24 TF（85-91%，中位 9.71）**；3 格凹陷 8.58-8.90（(512,512)/(1024,512)/(1024,4096)，cuBLAS 启发式选择差异） |
| 小形状地板 | 64³ = **0.54 TF**（launch 主导，「太小」≠「库慢」） |
| GEMV 带宽平台 | M≤32：**360-386 GB/s**（80-86% HBM） |
| 算力平台 | M≥128：**9.7-10.2 TF**；带宽跌到 <170 GB/s |
| roofline 拐点 | AI=25 FLOP/B → **M*≈51**（N=K=4096），实测拐在 32~64 之间 |
| skinny 不对称 | M-skinny **7.97** > N-skinny **5.92** TF（row-major：N 窄的写回吃亏） |
| wave 坏形状 | M=160/224 掉 ~10%（跨波不满最差）；**≥4 波损失 <5%** |

### 3.2 调度机制（split-K / batch）

| 断言 | 数字 |
|------|------|
| 库内部 split-K 证据 | 256×256×16384 单发 **8.20 TF** vs 4-tile 饥饿上限 **0.93 TF**（差 8.8×） |
| 手工 split-K 全败 | s=2/4/8/16 stream → **6.83/5.58/3.73/2.00 TF**；顺序对照 **7.57/6.68/5.30/3.56**——多流比顺序**还慢** -9.8%/-16.6%/-29.6%/-43.9%（WDDM 跨流净开销，随 s 放大） |
| batch vs flat | flat 稳赢：B=64 **+13%**、B=4 **+64%**；B≤2 持平（launch 区无意义） |

### 3.3 dtype 与缓存

| 断言 | 数字 |
|------|------|
| FP32→FP16 断层 | **6.4-6.9×**@2048（两次观测：9.49→65.46 / 10.46→67.43），硬件比 8× |
| 峰值达成率 | FP32 **85-93%**（跨观测带）vs FP16 **68-76%**（TC 更难喂饱 + 时钟态敏感） |
| FP16@4096 功率墙 | 两次观测均比 @2048 低（-5.9%/-10.0%）（61.6/60.7 vs 65.5/67.4）——持续负载时钟回落，方向稳健 |
| FP16 时钟态敏感性 | 跨会话 **54.9 / 58.4 / 65.5 / 67.4**；10s idle 掉到 **3.4 TF**；FP32 大形状仅 ±1.5% |
| 测量窗口效应 | @1024/@4096 复跑差 <1.5%，@512/@2048 差 10-17%——**噪声带宽度 ∝ 测量窗口长度**（1.7ms×10reps 的窗口短于时钟爬坡） |
| L2 驻留收益 | 仅 ws<L2：**×1.12**@3MB；ws≥12MB 后 flush 与否 ±2% 内 |
| layout 无关性 | 7 组合极差 **5.1%**；B 侧转置微慢 ~3%；为对齐做 copy 必亏 |

### 3.4 CUTLASS 08_turing 对账（源码推导，[源码]）

| 项 | 值 | 计算式 |
|----|----|--------|
| warp 数 / 线程 | 8 warp / 256 线程 | (128/64)×(256/64) |
| shared 用量 | **48KB** = NumStages 2 × (128×64 + 64×256)×1B | 2×24576B |
| 为何 2 级 | 3 级 = 72KB > Turing 64KB | **模板参数 = 内存预算** |
| mma.sync | m8n8k16 = **1024 MAC**/条 | warp k 步需 256 条 |
| INT8 峰值链 | 178.4 TOPS → 1024 MAC/SM/clk → 256/sub-core/clk | ÷(48×1.815GHz)，4 sub-core |
| epilogue | 16B 向量化（4×int32），只算 α·X+β·C | 128/32=4 |
| canonical 布局 | A 行主 + B 列主 = 「TN」 | mma fragment 免转置对齐 |

## 4. 手写骨架

### 4.1 CUTLASS `device::Gemm` 组装（08 示例核心，模板即文档）

```cpp
using ElementAccumulator = int32_t;          // INT8 乘积必须 32bit 累加
using ElementInputA = int8_t;  using LayoutInputA = cutlass::layout::RowMajor;
using ElementInputB = int8_t;  using LayoutInputB = cutlass::layout::ColumnMajor;  // TN
using ElementOutput = int32_t; using LayoutOutput = cutlass::layout::RowMajor;
using MMAOp   = cutlass::arch::OpClassTensorOp;   // tensor core 路线
using SmArch  = cutlass::arch::Sm75;              // 决定 mma 指令代际
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;
using ShapeMMAWarp        = cutlass::gemm::GemmShape< 64,  64, 64>;  // 8 warp 拼 TB
using ShapeMMAOp          = cutlass::gemm::GemmShape<  8,   8, 16>;  // mma.sync m8n8k16
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    int32_t, 128 / 32 /*4 元素=16B 向量*/, int32_t, int32_t>;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 2;  // 不是随手写：2×24KB=48KB ≤ 64KB shared，3 级超限
using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, /*B*/ ElementInputB,
    LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch,
    ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
// 生命周期：can_implement(args) → get_workspace_size → initialize → gemm_op()
```

### 4.2 wave 量化口算（面试白板题）

```
tiles = ⌈M/TM⌉ × ⌈N/TN⌉        // TM=TN=128 模型
waves = ⌈tiles / 48⌉             // 48 SM
坏形状 = tiles 刚跨过 48 的整数倍但没填满（多付一整波，只多算一点）
损失上界 ≈ (waves - tiles/48) / waves；waves ≥ 4 时 < 25%…实测 < 5%
```

### 4.3 手工 split-K 骨架（E3 实测的「反面教材」版，讲清为什么输）

```python
for i in range(s):                      # K 切 s 段
    with torch.cuda.stream(streams[i]): # 多流并发——E3seq 对照实测：比纯顺序还慢，WDDM 跨流提交/事件同步是净开销
        torch.matmul(A[:, i*K//s:(i+1)*K//s], B[i*K//s:(i+1)*K//s, :], out=C_part[i])
                                        # 每路小 GEMM 又回到饥饿态，库还会对它再 split——两层 split 打架
default_stream.wait_all(streams)
C = sum(C_part)                          # 归并串行在默认流，s 大时开销超过本体
# 实测：s=1: 8.2 TF（库已内部 split）→ s=16: 2.0 TF。教训：先证明基线没偷用你的优化。
```

## 5. 红线清单（说错一条降一档）

1. **方阵 TFLOPS 不能外推 skinny**：饱和区 9.5-10.5 TF，GEMV 区按 TFLOPS 看只有 0.2-2.9，但带宽已达 86%——**指标要跟瓶颈走**（带宽区报 GB/s）。
2. **「M 小和 N 小一样」是错的**：row-major 输出下 N-skinny（5.92）比 M-skinny（7.97）惨 26%——写回列窄 + B 复用差。
3. **手工优化前先验证基线**：cuBLAS 对 256×256×16384 已自动 split-K（饥饿模型 0.93 vs 实测 8.2 TF）。不识别这一点会得出「split-K 没用」或「我的优化有用」的双重错误。
4. **split-K 不是免费午餐**：多 launch + workspace/归并；多流在 WDDM 下实测为**负**贡献（顺序对照 E3seq：stream 比 seq 慢 -9.8%~-43.9%）；库只在饥饿时才启用它。并发归因必须有顺序对照，否则只是推测。
5. **batch 维不是免费并行**：bmm 有组间调度开销（B=64 时 -13%）；能堆叠就 flat。
6. **FP16/TC 数字必须报时钟态条件**：同一 2048³ FP16 跨会话 54.9~67.4 TF（±10%）；10s idle 掉到 3.4 TF。「我测过 X TF」后面永远跟着「什么烧机条件」。**窗口效应**：1.7ms×10reps 的测量窗口短于时钟爬坡——@2048 复跑差 10%，@4096（70ms 窗口）差 <1.5%。
7. **两套计时协议的数字不可混比**：循环协议（吞吐）vs per-iter event 协议（受控缓存态，含同步开销，绝对值低 ~30%）。
8. **「TN 最快」是过时经验**：本机 7 种布局组合极差 5%（ldmatrix + swizzled shared 的功劳）；为布局对齐做 transpose/copy 几乎必亏。
9. **CUTLASS 红线**：Turing **有** ldmatrix（CUDA 10.2 起）；Turing 双缓冲**不是** cp.async（走寄存器站两跳，Ampere 才直通 shared）；NumStages=2 在 INT8 下是 64KB shared 硬限的必然。
10. **模板库 vs 运行时库的启发式透明度**：CUTLASS 把 split-K/swizzle/stages 暴露成参数；cuBLAS 藏在运行时——「谁替你做决定」是选型/调试的第一问。

## 6. 60 秒电梯陈述

> GEMM 性能不是一个数，是**形状空间里的调度问题**。我在 RTX 5000 上扫过这个空间：
> 饱和区 9.5-10.5 TF，但 64³ 只有 0.54（launch 主导）；M≤32 是带宽平台（80-86% HBM），
> 拐点在 AI=25 对应的 M*≈51，和 roofline 推算对上；skinny 还分方向，N 窄比 M 窄惨 26%。
> 调度上，wave 量化让「刚跨波不满」的形状最亏（-10%，≥4 波后 <5%）；cuBLAS 对小输出
> 大 K 自动 split-K——我拿饥饿模型上限 0.93 TF 对实测 8.2 TF 才确认这点，手工切分全败，且加了
> 顺序对照实验证明多流在 WDDM 上是净开销（比纯顺序还慢）。
> batch 有组间调度开销，能 flat 就别 bmm。dtype 断层 FP32→FP16 实测 6.4-6.9×，
> 但 FP16 数字跨会话 ±10%（时钟态），报数必须带条件。CUTLASS 把这些自由度做成模板：
> 三级 tile、NumStages=2 是 INT8 tile 48KB 对 64KB shared 的预算必然——读它的源码等于
> 把 GEMM 的调度自由度清单过一遍。
