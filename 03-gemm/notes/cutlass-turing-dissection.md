# CUTLASS 08_turing_tensorop_gemm 拆解——Turing INT8 Tensor Core GEMM 的官方参考实现

对象：`03-gemm/cutlass/examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu`（357 行，单文件）
对照文档：`cutlass/media/docs/cpp/efficient_gemm.md`（Hierarchical Structure / Epilogue / Optimizations）
本机：Quadro RTX 5000（sm_75，本示例的精确目标硬件：Turing + INT8 TC + mma.sync + ldmatrix）

> 本文所有数字都可由示例源码常量 + 机器规格直接推导，计算式随文给出。区分两类事实：
> **[源码]** = 从示例/头文件读到的事实；**[本机]** = 本机 cuBLAS 实测（见 gemm-sweep/results.md）。

---

## 1. 一句话定位

CUTLASS 把「一个高性能 GEMM kernel」拆成**三级 tile 层级 + 流水线 + epilogue** 的模板组装题：你声明数据类型/布局/三级 tile 尺寸/流水级数，库推导出线程数、shared 用量、bank 冲突规避装载、mma 指令序列。08 示例 = 用 ~60 行类型别名在 Turing 上组装出 INT8 tensor core GEMM。

## 2. `device::Gemm` 的 15 个模板参数逐个释义（[源码] L179-193）

| # | 参数 | 示例取值 | 释义 |
|---|------|---------|------|
| 1 | `ElementInputA` | `int8_t` | A 元素类型——Turing TC 支持 INT8/FP16 mma |
| 2 | `LayoutInputA` | `RowMajor` | A 行主：A[m][k] |
| 3 | `ElementInputB` | `int8_t` | B 元素类型 |
| 4 | `LayoutInputB` | `ColumnMajor` | B 列主：B[n][k] |
| 5 | `ElementOutput` | `int32_t` | D 输出类型 |
| 6 | `LayoutOutput` | `RowMajor` | D 行主 |
| 7 | `ElementAccumulator` | `int32_t` | 累加器——INT8 乘积必须 int32 累加（8bit 会溢出） |
| 8 | `MMAOp` | `OpClassTensorOp` | 用 tensor core（对照 `OpClassSimt` = 纯 CUDA core 路线） |
| 9 | `SmArch` | `Sm75` | 目标架构——决定用哪代 mma 指令 |
| 10 | `ShapeMMAThreadBlock` | `GemmShape<128,256,64>` | threadblock tile |
| 11 | `ShapeMMAWarp` | `GemmShape<64,64,64>` | warp tile |
| 12 | `ShapeMMAOp` | `GemmShape<8,8,16>` | 单条 mma.sync 的形状（INT8: m8n8k16） |
| 13 | `EpilogueOp` | `LinearCombination<int32, 4, int32, int32>` | epilogue：D = α·(AB) + β·C |
| 14 | `SwizzleThreadBlock` | `IdentityThreadblockSwizzle<>` | threadblock 栅格化顺序 |
| 15 | `NumStages` | `2` | 流水线级数（double buffering） |

**注意 A 行主 + B 列主这个组合**：CUTLASS 的经典「TN」布局。B 列主即 B 按 [n][k] 存储——mma fragment 的装载路径（ldmatrix）对这个组合是「天然对齐、免转置」的。**与本机 E7 实测的呼应**：cuBLAS/torch 的 row-major 世界里 layout 差异已小到 5%（[本机] fig7），但 CUTLASS 的模板默认值仍然选择 TN——**库作者把「对 mma 最友好的布局」定为规范**，这是文档层的教学锚点。

## 3. tile 层级对账表（每个数字带计算式）

| 层级 | 形状 | 推导 | 本机约束对账 |
|------|------|------|-------------|
| Threadblock | 128×256×64 | — | 见下 |
| Warp | 64×64×64 | warps = (128/64)·(256/64) = **2×4 = 8 warps = 256 线程** | ≤1024 线程/SM ✓；一个 SM 可驻留 4 个这种 block（线程约束）但见 shared 约束 |
| MMA op | 8×8×16 | 每条 `mma.sync.m8n8k16` = 8·8·16 = **1024 MAC** | Turing INT8 TC 指令（CUDA 10.2+，[源码] L327-330 注释明确点名 mma.sync 与 ldmatrix 同版本引入） |

**shared memory 用量（本 AR 的核心可解释点）**：
```
每 stage：A tile 128×64 + B tile 64×256（int8 = 1B）
        = 8192 + 16384 = 24576 B
NumStages=2：24576 × 2 = 49152 B = 48 KB
```
Turing 每 SM shared 上限 64KB → **48KB 恰好放得下 2 级**；3 级需要 73728B = 72KB > 64KB ✗。
**[源码→硬件] 因果链完整**：示例的 `constexpr int NumStages = 2;`（L177）不是随手写的——它被 INT8 数据密度（1B/元素）和 Turing 的 64KB shared 硬限制**共同决定**。FP16 示例（07_volta）tile 更小、级数可更多；sm_80 的 A100 shared 提到 164KB + cp.async，同模板可跑 3-4 级。**一个模板参数背后是整条内存层级预算**。

**warp 内指令展开**：warp tile 64×64×64 每 k 步需要 (64/8)·(64/8)·(64/16) = 8·8·4 = **256 条 mma.sync**。这些指令的 A/B 操作数 fragment 由 `ldmatrix` 从 shared 装载（Turing 起可用）。fragment 的线程分布（对照 `cutlass/include/cutlass/arch/mma_sm75.h`：INT8 m8n8k16 特化 `FragmentC = Array<int, 2>`）——**每线程持 2 个 int32 累加器**（8×8=64 输出 ÷ 32 线程），A/B 操作数 fragment 各 4×int8（打包进 1 个 32 位寄存器）。注意 FP16 m8n8k4 / FP32 m8n8k8 特化的 FragmentC 是 `Array<_, 4>`——不同 dtype 的每线程累加器数不同。

**网格级对账（示例问题规模 5120×4096，[源码] L197-199）**：TB tile 128×256 → ⌈5120/128⌉×⌈4096/256⌉ = **40×16 = 640 个 threadblock**；640/48 SM ≈ **13.3 waves**——大问题的 wave 量化损失被摊薄（最后不满波占比 <8%）。

**178.4 INT8 TOPS 换算链**（峰值→指令节拍）：
```
178.4e12 ops/s ÷ (48 SM × 1.815e9 Hz) = 2048 ops/SM/clk = 1024 MAC/SM/clk
÷ 4 sub-core/SM = 256 MAC/sub-core/clk
= m8n8k16（1024 MAC）每 4 clk 一条 / sub-core
```
FP16 同链：89.2 TFLOPS → 512 MAC/SM/clk → 128/sub-core/clk（m16n8k8 = 16×8×8 = 1024 MAC/条 → 每 sub-core 8 clk 完成一条）。
**INT8 : FP16 = 2:1** 的 MAC 吞吐 = Turing tensor core 的代际规格；[本机] E5 实测 FP16 65.5-67.4 TF（@2048，73-76% 峰值，两次观测），INT8 路径 Windows torch 构建未实现未能实测（env.json 留档）——但本示例的模板参数 10-12 就是为跑满这条 178 TOPS 线而设的。

## 4. 数据流与流水线（[源码] L81-101 注释原文的相位图）

```
单 pipeline（同步，每 stage 等上一个）：
(1) global → (2) registers → (3) shared → (4) registers → (5) mma → (6) registers → (7) global

双 pipeline（NumStages=2，相位错位一半）：
pipeline A: (1)→(2)→(3)→(4)→(5)→(6)→(7)
pipeline B:        (1)→(2)→(3)→(4)→(5)→(6)→(7)
```
- Turing **没有 cp.async**（sm_80 才有）——所以 (2) registers 这一站存在：global 先进寄存器再写 shared，用「一次 global读 + 一次 shared 写」把不定期延迟吸收在寄存器站。**这就是 [源码] 数据流里 registers 出现两次的原因**，也是 Turing 与 Ampere 流水线的本质分界。
- 第 2 条 pipeline 的 global 装载与第 1 条的 mma 计算重叠 → 「用已装载数据的计算时间，藏下一段的装载延迟」。
- 对照 `efficient_gemm.md` Optimizations → Pipelining 一节：双缓冲是「多级流水」的最小形式；sm_80+ 的 multistage（cp.async + 更深队列）是同一思想的深化。

## 5. epilogue：`LinearCombination`（[源码] L166-174）

```cpp
LinearCombination<int32_t, 128 / sizeof_bits<int32_t>, int32_t, int32_t>
//              ↑输出类型   ↑128/32 = 4 元素/次向量访问      ↑α/β 计算类型
```
- **D = α·(A·B) + β·C**：主循环只算 A·B；α/β/C 的逐元素运算全部推迟到 epilogue——**这是「GEMM 可组合性」的关键**：主循环形状与 epilogue 解耦，GELU/bias/残差都挂在 epilogue 模板上（CUTLASS 例 12 bias+ReLU 即换一个 EpilogueOp）。
- 向量宽度 4×int32 = 16B/访问：epilogue 的 global 写回按 16B 向量化，与主循环的 coalescing 同级纪律。
- 对照 `efficient_gemm.md` Epilogue 一节：epilogue 与主循环共用 tile 迭代器抽象，输出 tile 的分块方式自动对齐主循环。

## 6. swizzle、split-K 与生命周期

**SwizzleThreadBlock = IdentityThreadblockSwizzle**（L164）：threadblock 按 linear 顺序铺。CUTLASS 另有 swizzled 栅格（如按对角线分组）——目的是**提高 L2 命中**：相邻 block 共享 A 行/B 列，把「同时活跃的 block」安排成共享输入的邻居。[本机] E6 显示 GEMM 迭代内 L2 复用本来就活跃（flush 只差 12%），swizzle 正是把这个复用「调度出来」的工具。对照 `efficient_gemm.md` Optimizations → Threadblock Rasterization。**batch 维度的对照**：CUTLASS batched GEMM（例 05）用 `GemmBatchedIdentityThreadblockSwizzle` 把 batch stride 纳入同一个调度抽象——batch index 进 tile 坐标第三维，每个 batch 组的 tile 独立栅格化。[本机] E4 的 flat-vs-bmm 对比给出了这个抽象的运行时成本：组间索引/边界调度是 batched kernel 相对 flat 的额外开销（B=64 时 -13%），CUTLASS 把同一权衡做成「显式 batch 维」的模板选择。

**split_k_slices = 1**（L254）：K 不切。设为 s>1 时：s 个 block 各算一段 K 的部分和 → 写入 workspace → 再起 reduction kernel 归并。适用场景 = **输出 tile 太少喂不满 SM**（skinny/小输出大 K）。[本机] E3 实测了同一权衡的运行时版本：cuBLAS 对 256×256×16384 自动做了内部 split-K（8.2 TF >> 0.93 TF 饥饿界）；CUTLASS 把它暴露为显式模板/运行时参数——**启发式透明度**是两类库的哲学差异。对照 `efficient_gemm.md` Optimizations → Parallelized Reductions。

**生命周期**（L276-285）：
```cpp
gemm_op.can_implement(arguments)   // 静态检查：布局/对齐/形状与模板是否相容
Gemm::get_workspace_size(arguments) // split-K workspace 等
gemm_op.initialize(arguments, workspace.get())
gemm_op()                           // launch
```
`can_implement` 是模板库的「编译期约束运行时化」：模板选错了在这里报错而不是产出错误结果。

## 7. 与 efficient_gemm.md 层级理论的对照表

| 文档概念 | 08 示例中的对应物 |
|---------|------------------|
| Hierarchical Structure: Threadblock-level GEMM | `ShapeMMAThreadBlock 128×256×64` + shared 双缓冲 mainloop |
| Hierarchical Structure: Warp-level GEMM | `ShapeMMAWarp 64×64×64`，8 warp 拼 128×256 |
| Hierarchical Structure: Thread-level GEMM | 每线程持 C fragment（m8n8 输出/每 warp 32 线程 → 每线程 2 个 int32，`mma_sm75.h` FragmentC=Array<int,2>） |
| Optimizations: Pipelining | `NumStages=2`（相位错位，见 §4） |
| Optimizations: Threadblock Rasterization | `IdentityThreadblockSwizzle`（可换 swizzled） |
| Optimizations: Parallelized Reductions | `split_k_slices`（workspace + reduction） |
| Optimizations: Hopper Warp Specialization | sm_90 专属（08 不涉及；生产者/消费者 warp 分工是 sm_80+ 的进一步演化） |
| Epilogue | `LinearCombination` + 16B 向量化 |

## 8. sm_80+ 差异红线（「Turing 为什么止步于此」）

| 差异 | Turing sm_75 | Ampere+ sm_80 | 对本示例的含义 |
|------|--------------|---------------|----------------|
| 异步拷贝 | 无（global→寄存器→shared 两跳） | **cp.async** 直达 shared | NumStages 受 shared 容量硬限 → 2 级 |
| shared/SM | 64KB | 164KB（A100，可配） | 48KB 双缓冲 vs 3-4 级深流水 |
| mma 指令 | m8n8k16（INT8）/ m16n8k8（FP16，f16/f32 累加两种特化） | m16n8k8 / m16n8k16 等 | ShapeMMAOp 参数化掉架构差异 |
| TF32 | 无 | 有（m16n8k8 TF32） | Turing FP32 路线只有 SIMT |
| ldmatrix | **有**（CUDA 10.2 引入，[源码] 注释） | 有 | fragment 装载效率的保障 |

**红线**：说「Turing 没有 ldmatrix」是错的（CUDA 10.2 起就有）；说「双缓冲就是 cp.async」是错的（Turing 的双缓冲走寄存器站，cp.async 是 Ampere 的免寄存器直达路径）。

## 9. 60 秒复述脚本（面试用）

> CUTLASS 的 GEMM = 三级 tile 的模板组装：threadblock 128×256×64 由 8 个 64×64 的 warp tile 拼成，warp 内部用 m8n8k16 的 INT8 mma.sync 铺满。数据在 global→寄存器→shared→寄存器→mma 的流水线上跑两级 double buffering——Turing 没有 cp.async，所以必须经寄存器中转，且 64KB shared 只够 INT8 tile 放两级（2×24KB=48KB），NumStages=2 是内存预算的必然。累加只算 A·B，α·X+β·C 推给 16B 向量化的 epilogue；split-K 和 swizzle 是两个正交旋钮，分别解决「tile 太少」和「L2 复用调度」。我在 RTX 5000 上实测过这套权衡的运行时版本：cuBLAS 对 256×256×16384 自动 split-K 打到 8.2 TF（73% FP32 峰值），手工 stream 并发反而全输——库启发式把模板参数干的事在运行时全干了。
