# AR004 triton-dsl-lab 实测结果与分析

> 数据来源：`results/*.json`（2026-10 复跑，编译/计时分离协议）；图为 `figs/fig1..fig9`。
> 环境：triton 3.8.0（Windows wheel，全局）、torch 2.5.1+cu121、numba 0.68（venv 子进程）、
> Quadro RTX 5000（sm_75，48 SM，HBM 448 GB/s，shared 64KB/SM，FP32 峰值 11.2 TF）。
> 本文档所有数字与 JSON 一致，未四舍五入处保留 3 位。

## 0. 协议：编译/计时分离（本 AR 最重要的方法论产出）

E3 首跑（旧协议：逐尺寸编译→立即计时）出现 cuBLAS@2048 = 3.44 TF 的异常
（正常带 ~9.5-10.3 TF）。根因：**编译阶段 CPU 侧忙、GPU 空闲掉频，WDDM 下时钟
漂移进入紧随其后的计时窗口**。修复后协议：

1. **Phase A**：预编译全部 config（CPU 侧，不计入任何数据）；
2. **烧机**：~2s 满载矩阵乘把时钟拉到工作频率；
3. **Phase B**：背靠背计时（cudaEvent，reps=10 取中位），每个尺寸前 fresh burn；
4. 全程 `nvidia-smi` 遥测（clock/temp/power 落 console 与 JSON）。

修复后：E3 扫描计时相仅 1s（旧协议 36s），全程时钟 1875-1920 MHz；
cuBLAS@2048 复测 9.95 TF（+189% vs 异常值）；E1/E2 用同协议复跑。
E1 大 N 两轮重复性：torch@64M 377.2 → 377.6 GB/s（0.1%），triton@64M 372.0 → 372.6（0.2%）。

## 1. E1 向量加法：三方框架（fig1, fig2）

### 大 N：全部收敛到 83% HBM（fig1）

| N=64M | torch | triton(bs1024) | numba |
|---|---|---|---|
| GB/s | 377.6 | 372.6 | 368.9 |
| % HBM(448) | 84.3% | 83.2% | 82.3% |

内存受限 kernel 的框架无关性：搬 2N 字节的活，谁来发指令都一样。
BLOCK 大小（256/1024/4096）在大 N 无差异（373.2/372.6/369.4 @64M）。

### 小 N：launch 开销分层（fig2）

实测时间下限（n≤65536 各尺寸最小值）：

- torch ~**25.6 µs**（各尺寸跳动 25.6-39.9 µs，WDDM 抖动）
- triton ~**45.6 µs**（极其稳定：45.6-46.8 µs，逐尺寸几乎不变）
- numba ~**69.4 µs**（69.4-70.4 µs，32768 一格 108.5 µs 离群）

三层台阶 = 各框架 launch 路径的成本（WDDM 用户态提交队列的贡献远大于 Linux）。
triton 的稳定性和 torch 的抖动形成对比：torch 走 cuBLAS/elementwise dispatch，
WDDM 提交路径时有批处理停顿；triton launcher 是薄封装。
首跑 numba@32M 曾出现 10.4 ms 异常（复跑未再现），同 AR003 结论：单点异常一律复跑取证。

## 2. E2 融合 softmax（fig3, fig4）

M=1024 行固定，行宽 N 扫描。有效带宽口径 = 2MN 字节 / 时间。

### 教程 kernel ≠ 最优 kernel（fig3）

| N=4096 | triton 教程 | torch 原生 | naive 5-pass |
|---|---|---|---|
| 有效 GB/s | 148.2 | 211.2 | 60.3 |

- triton 教程 kernel：**142-154 GB/s 平台**，几乎不随行宽变化（nw4 稳定 142-149）；
- torch 原生 softmax：183-221 GB/s，**快 1.2-1.6×**；
- num_warps 4 vs 8 影响很小；nw8@4096 一格 107.5 GB/s 为单次抖动（同 config 其余尺寸 146-154）。

### 为什么教程 kernel 慢：PTX 取证（写进 F4 文档的证据）

排除「缺向量化」假设：softmax kernel（BLOCK=4096, nw8）PTX 中
`ld.global.v4` ×4（256 线程 ×16 元素 = 4×v4/thread）、`ex2.approx` ×16
（快速 exp 已用上）、37 regs、0 spills。mask 也不阻止向量化
（对照实验：masked/unmasked 行加载同样产生 v4×8）。

剩余差距的机制：教程 kernel 每行做 **block 级规约**（`bar.sync`×4、shared 往返），
96 CTA = 每 SM 只 2 个 CTA，规约停顿无法被足够多的并发 CTA 掩盖 →
146 GB/s ≈ 33% HBM。原生实现（多行/CTA + 更好的 ILP 结构）把这一层吃掉了。
结论：**教程 kernel 是教学产物，离该 GPU 上的最优还差 1.2-1.6×**——
"会用 triton" 和 "用 triton 写到最优" 是两个台阶。

### naive 5-pass：流量模型上界 4×，实测只有 2.45×（fig4）

naive 有效带宽 ~60 GB/s（平坦），triton 教程 142-154 GB/s → **实测有效比仅
2.4-2.5×**，低于 8MN+4M 流量模型给出的 4× 上界。两个软化机制（口径诚实）：

1. **L2 吸收**：本实验总量固定 2^20 元素，每个中间张量（z、exp 结果）恰为
   4MB ≈ 本机 L2 容量——5 个 pass 之间的中间数据相当部分由 L2 供血，
   实际 DRAM 流量低于 8MN 模型（模型上界 239-252 GB/s 是「全部来自 DRAM」的假设）；
2. **多 kernel 间隙**：naive 是 4-5 个独立 kernel 串联，launch/收尾间隙
   在 0.13-0.14ms 的总时长里占比不可忽略。

结论修正为更准确的教学点：**融合省的是流量，但小工作集下 L2 会部分
替 naive 买单；naive 的 2.45× 劣势 = 流量差 × L2 折扣 + 多 kernel 间隙**。
若把总量放大到中间张量远超 L2（如 64MB），4× 模型上界才会逼近。

## 3. E3 FP32 matmul：config 全扫描 + 三方对比（fig5, fig6, fig7）

### 三方对比（fig5）

| N | triton 最优 | cuBLAS | triton/cuBLAS | numba K2b | numba K1 |
|---|---|---|---|---|---|
| 256 | 0.557 | 0.885 | 63% | 0.319 | 0.393 |
| 512 | 2.945 | 4.178 | 70% | 1.161 | 0.892 |
| 1024 | 6.404 | 8.549 | 75% | 1.661 | 1.072 |
| 2048 | 8.666 | 9.950 | **87%** | 2.008 | 0.944 |

- **triton 达成 cuBLAS 的 87%**（@2048）——tiled matmul + 自动配置扫描即可接近手调库；
- numba K2b（AR002 寄存器分块手写最优）= cuBLAS 的 20%；同一算法骨架，
  triton 的代码生成（软件流水 stages、更优的 shared 访问模式）值 4×；
- 跨 AR 对账：cuBLAS@2048 9.950 vs AR002 基线 9.468（+5.1%，带内）；
  numba K2b 2.008 vs 2.036（-1.4%），**复跑在基线带内**。

### config 扫描 @1024：76 valid / 32 pruned，spread 2.67×（fig6）

- 网格 BM,BN ∈ {32,64,128}²，BK ∈ {16,32,64}，warps {4,8}，stages {2,3}；
  剪枝 shared = stages×BK×(BM+BN)×4B ≤ 64KB，32 个 config 被剪（如 128×128×BK64）；
- **最优 6.90 TF = 64×128×BK16 w4 s2**；最差 2.59 TF = 32×32×BK16 w8 s3；
- 规律：BM×BN 大的方向快（更多计算复用），但 32×32 全系垫底（复用差），
  128×128 只有 BK16/32 能活（寄存器/shared 压力）；
- **w4 系统性优于 w8**（top-6 中 5 名是 w4，第 4 名 64×128×32_w8 仅差 2.3%）：FP32 FMA
  路径上，每线程更多累加元素 = 更长依赖链可被 ILP 填充；w8 把每线程的活摊薄了。

### 最优 config 随 N 迁移（fig7）+ 机器资源对账

@256/512 最优 64×128×32_w8_s2 → @1024 64×128×16_w4_s2 → @2048 128×64×16_w4_s3。
**autotune 是 per-size 的**：小尺寸 grid 少、wave 效应主导（需要大 tile 减少 tile 数）；
大尺寸算力主导（tile 大到寄存器压力可控即可）。cuBLAS 的 heuristic 每个尺寸都
压住 triton 扫描出的最优——库的启发式就是一张预扫好的表。

资源对账（shared = stages×BK×(BM+BN)×4B；wave = tiles/48 SM）：

| N | 最优 config | shared 用量 | tiles | waves |
|---|---|---|---|---|
| 256 | 64×128×32 w8 s2 | 48 KB（75% 上限） | 8 | 0.17（不满一波） |
| 512 | 64×128×32 w8 s2 | 48 KB | 32 | 0.67（不满一波） |
| 1024 | 64×128×16 w4 s2 | 24 KB（38%） | 128 | 2.67 |
| 2048 | 128×64×16 w4 s3 | 36 KB（56%） | 512 | 10.67 |

规律可读：小尺寸选**大 tile 填满 shared**（把 8 个 tile 做厚，摊薄每 tile 的
固定开销）；大尺寸转向**浅 BK 多 stages**（2.67-10.7 波时算力主导，
流水深度比 tile 面积更重要）。64KB 预算没有一个最优 config 用满——
寄存器压力（fp32 累加器）先于 shared 成为约束。

### GROUP_M {1,8}：±5% 以内，@2048 gm1 反而快 4.5%

gm8/gm1 = 1.002 / 0.999 / 1.017 / 0.955（@256→2048）。L2 分组重排在
本机 4MB L2、这些尺寸下收益有限；@2048 gm8 慢 4.5% 的方向值得注意
（分组使部分 wave 跨步访问，L2 命中改善不足以抵消）。
教程把 GROUP_M 当关键参数，实测在本设置里是二阶效应。

### 扫描相 vs 隔离相：-7%

同 config（64×128×16_w4_s2@1024）在扫描循环里 6.90 TF，三方隔离测量 6.40 TF。
背靠背循环里 L2/时钟处于「热」状态。写结果时要说明口径——这是
AR003「多观测带」教训的又一实例：**数字必须连同测量上下文一起报告**。

## 4. E4 FP16：triton wheel 在 sm_75 不触发 Tensor Core（fig8, fig9）

### 实测：26 config × 3 尺寸全部 mma.sync = 0

每个 config 的 PTX/中间表示计数（落 e4.json）：`mma_count=0`、
`ldmatrix_count=0`、`n_spills=0`，乘法是标量 `fma.rn.f32`
（静态计数 = BM×BN×BK/threads，如 64×64×16 w4 = 512 条/循环体，与实测一致）。

### 后果一：fp16 比 fp32 还慢（fig8）

| N | triton FP16 最优 | triton FP32 | 差 | cuBLAS FP16(TC) | TC 差距 |
|---|---|---|---|---|---|
| 512 | 2.90 | 2.94 | -1.4% | 8.738 | 3.0× |
| 1024 | 5.38 | 6.40 | **-16%** | 22.467 | 4.2× |
| 2048 | 6.64 | 8.67 | **-23%** | 63.072 | **9.5×** |

fp16 输入没换来任何 TC 收益，还要付 fp16→fp32 转换与寄存器搬动——**净亏**。
「把数据转成 fp16」本身不提速；提速的是「TC 路径被触发」。
cuBLAS FP16@2048 = 63.07 TF，是 FP32 的 6.3×、triton FP16 的 9.5×，
落在 AR002 基线带（60.7-67.4）内。

### 后果二：小 tile 全面获胜（fig9）

FP16 最优全是 64×64×16_w4（196 regs），128×128 系无一胜出——
FMA 路径下大 tile 的累加器寄存器压力（fp32 acc）没有 TC 的分摊机制，
寄存器压力直接压垮吞吐。与 FP32 扫描（64×128/128×64 获胜）对照，
** lowering 路径改变了最优 config 的形状**——这是「autotune 必须在
真实 lowering 下做」的实证。

### 源码有、构建未触发（F4 文档核心素材）

上游 triton 3.9 源码存在 MMAv2 Turing 路径
（`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp`：`mmaInstrPtxTuring`
m16n8k8、`callMmaTuringFp16`，L337-463），但本机 3.8.0 Windows wheel
的构建未为 sm_75 fp16 dot 选择该路径。「源码里写着支持」和
「你的二进制里启用了」是两件事——版本/平台构建矩阵直接决定特性可用性，
**用 PTX 计数验证，不要信文档推断**。

## 5. 结论与跨 AR 沉淀

1. **带宽型 kernel**（E1/E2）：框架差异在 launch 层（25.6/45.6/69.4 µs 三台阶），
   大 N 的稳态吞吐三方一致（82-84% HBM）；融合的价值在流量（模型 4×；
   本设置因中间张量 ≈ L2 被 4MB 吸收，实测 2.45×），不在每字节效率。
2. **计算型 kernel**（E3）：triton + config 扫描 = cuBLAS 的 87%（FP32/FMA 路径），
   同骨架 numba 只有 20%——代码生成质量的差距是真实的。
3. **精度路径**（E4）：lowering 决定一切。sm_75 + triton 3.8 wheel 的 fp16 dot
   = 标量 FMA，比 FP32 还慢；cuBLAS TC 同题快 9.5×。
   「换精度」不是开关，是「 lowering 是否变」的问题。
4. **协议**：WDDM 下编译/计时必须分离 + 烧机 + 时钟遥测；
   单点异常复跑取证；数字必须连同上下文（扫描相/隔离相差 7%）报告。
