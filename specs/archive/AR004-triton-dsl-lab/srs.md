# [AR004] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| AR 主题 | triton-dsl-lab（Triton 教程线 vector add→fused softmax→matmul+autotune 真机实验 + numba/cuBLAS 三方对照 + sm_75 lowering 拆解） |
| 关联 SR | 仓库 README「建议顺序」第 4 条：04-kernel-dsl Triton 学 block 编程/autotune/编译流水线；AR002/AR003 延伸（三方语言栈对照） |
| 日期 | 2026-10-06 |
| 状态 | Draft |

## 1. 背景与目标

`04-kernel-dsl/triton/` 是 Triton 完整源码克隆（3.9.0），含官方教程线（01-vector-add → 02-fused-softmax → 03-matrix-multiplication+autotune）。本机 triton 3.8.0.post29 已解锁（CRT 修复，真机 sm_75 验证通过）。AR002 用 numba 手写了 GEMM（K0/K1/K2b，最高 2.04 TF），AR003 扫了 cuBLAS 形状空间（饱和 9.5-10.2 TF）——**三方语言栈（numba CUDA / Triton DSL / cuBLAS 库）在同一 GPU 上的对照**是面试高价值素材：同一算法在不同抽象层的代价与天花板。

**smoke 预实验（2026-10-06，已去风险）**：
- Triton 3.8 在 sm_75 编译/运行全通（fp32 dot 误差 1.7e-5，fp16 0.0156 均符合舍入预期）
- **重大发现**：fp16 `tl.dot` 在 16 组 config 扫描中 **0 个 mma.sync/ldmatrix**——全部降级为标量 FMA（fma.rn.f32 数 ∝ BM×BN×BK，大 tile spill 最高 4856）——**该构建不用 Turing tensor core**；上游 3.9 源码存在 MMAv2 Turing 路径（`mmaInstrPtxTuring` m16n8k8），是「源码有、构建未选择」的差距，E4/F4 的核心素材
- fp32 dot → FFMA（Turing 无 TF32，预期内）

目标：(a) 教程线三 kernel 真机基准（vs torch 原生）；(b) matmul 全 config 扫描的可解释 autotune 数据；(c) numba/cuBLAS 三方对照；(d) sm_75 lowering 拆解（PTX 级证据）；(e) 面试导向笔记。

## 2. 需求范围

**In Scope：**
- F1：基准框架 `bench_dsl.py`（全局 python：triton + torch cuBLAS；numba 经 AR001 venv 子进程同会话跑，JSON 落盘合并；沿用 AR002/003 计时纪律）
- F2：实验组：E1 vector-add（triton BLOCK_SIZE 扫描 × torch.add × numba add）、E2 fused-softmax（triton 教程 kernel × torch 原生 softmax × naive 5-pass 展开对照）、E3 matmul FP32 全 config 扫描（triton 手动 config 网格 + cuBLAS + numba K1/K2b）、E4 matmul FP16（triton FMA 降级路径 vs cuBLAS TC 54.9-67.4 TF 基线）+ PTX/mma 证据
- F3：`plot_results.py` ≥8 张可解释图（带宽模型线/峰值线/config 热图/spill 曲线）
- F4：`notes/triton-lowering-sm75.md`：tl.dot 在 sm_75 的实际 lowering（PTX 证据链：fp32→FFMA、fp16→FFMA 无 mma、上游 MMAv2 路径存在但未选择的源码定位）、block 编程模型要点（AGENTS.md 的 block-uniform 约束）、autotune 机制（cache/编译开销）、`python -c` 无法定义 @jit 的源文件约束
- F5：`notes/triton-notes.md` 六段面试笔记 + INTERVIEW-INDEX 增补 + results.md + README

**Out of Scope：**
- 不修改 triton/tvm 上游任何文件（零改动，mtime 核查）
- 不编译 triton/tvm 源码（只做源码定位 + 安装版实测）
- 不做 bf16/tf32（Turing 不支持/无 TF32）；不跑 06-fused-attention 教程（AR008 flash-attention 主题）
- 不改全局 Python 环境（numba 用现有 venv 子进程，不 pip install）
- 不用 triton.testing.do_bench（自建 Event 计时保持与 AR002/003 同协议，可跨 AR 对照）

## 3. 功能需求

### 3.1 F1 — 三方运行时基准框架

**描述**：`bench_dsl.py` 支持 `--exp E1..E4/all`；全局 python 跑 triton/torch 实验；`--exp E3` 时经子进程调用 numba venv python（`01-foundations/GPU-Puzzles/solutions/.venv`）跑 AR002 同款 kernel（K1 T16 / K2b 64×64 regblock 4×4）落独立 JSON，由主进程合并；计时纪律沿用：2s 烧机、3 次 warmup、自适应 reps 中位数、CUDA Event、JSON 落盘。

**验收标准：**
- Given 任一 triton 实验，When 连跑两次，Then 长窗口配置 TFLOPS/GB/s 波动 <5%（WDDM 会话噪声带口径同 AR002/003）
- Given numba 子进程实验，When 主进程合并 JSON，Then 记录含 venv 版本信息且与全局实验同会话（时钟态可比）
- Given 正确性门，When 每个配置首次执行，Then 先过门再计时：add 精确相等；softmax vs double 参考 ≤1e-6（fp32 下溢/上溢稳定性构造）；matmul FP32 vs torch 参考 rel-err ≤1e-5、FP16 ≤1.5 ulp@max|ref|（2026-10-06 修订：原定 ≤1e-2 绝对门在 fp16 输出量级下物理不可达——1 ulp@128 = 0.125 已超 1e-2，首跑实测触发；改用 AR003 E5 同款原理化门）

### 3.2 F2 — 四组实验

**E1 vector-add**：N ∈ 2^12..2^26 对数扫描；triton BLOCK_SIZE ∈ {256,1024,4096}；三方（triton/torch/numba）GB/s 对比；含 L2（≤4MB）与 DRAM 区的带宽台阶；有效流量模型 = 3×N×4B。
- 验收：三方曲线 + 448 GB/s 峰值线 + L2 4MB 拐点标注；triton 最优 BLOCK_SIZE 有结论

**E2 fused-softmax**：形状 M×N，N ∈ {1K,2K,4K,8K,16K} 行宽 × M 取 2^20/N 行（总量固定）；triton 教程 kernel（单遍融合，BLOCK=next_pow2(N)）vs `torch.nn.functional.softmax`（原生融合 kernel）vs naive 5-pass torch 展开（8MN+4M 流量）；GB/s 有效流量 = 2×M×N×4B（融合口径）。
- 验收：融合收益量化（triton vs naive 的 GB/s 差 ≥3× 预期方向验证；2026-10-06 修订：实测 2.45×，低于模型上界——中间张量恰 4MB ≈ L2 被部分吸收 + 多 kernel 间隙，方向成立但量级须按「流量差 × L2 折扣」口径报告）；triton vs torch 原生差距有分析（行宽/BLOCK/num_warps 效应）；正确性含极端值（±large 构造）

**E3 matmul FP32 全 config 扫描**：N ∈ {256,512,1024,2048} 方阵；triton 手动 config 网格（BLOCK_M/N/K ∈ {32,64,128}³ 剪枝 + num_warps {4,8} + num_stages {2,3,4}（2026-10-06 修订：实际网格 {2,3}，与 design.md 一致；srs 原文 {2,3,4} 系起草笔误，s=4 未跑，76 valid + 32 pruned = 108 仅与 {2,3} 闭环），失败 config 记录 OutOfResources 原因落 JSON）全量计时；三方对照（triton 最优 config vs cuBLAS vs numba K1/K2b）；GROUP_M swizzle 对比（固定最优 config，GROUP_M ∈ {1,8}）。
- 验收：config 网格热图（每格 TFLOPS，含失败格标注）；最优 config 随 N 的变化表 + 机器资源对账（shared 用量/wave 数）；三方差距结论（相对 cuBLAS 的达成率）

**E4 matmul FP16 + lowering 证据**：N ∈ {512,1024,2048}；triton fp16（FMA 降级路径实测）vs cuBLAS fp16（TC 54.9-67.4 TF 基线）；每 config 记录 n_regs/n_spills/mma.sync 计数/ldmatrix 计数落 JSON；PTX 证据（fp16 最优 config 的 PTX 片段存档）。
- 验收：量化「无 TC 的 fp16」与 cuBLAS TC 的差距倍数；mma=0/spill 数据支撑降级结论；与 smoke 预实验一致性

### 3.3 F3 — 图表

**描述**：`plot_results.py` 读 JSON 产 ≥8 图：fig1 add 三方带宽曲线（含 448 线 + L2 拐点）、fig2 add BLOCK_SIZE 对比、fig3 softmax 三方 GB/s、fig4 softmax 融合流量模型对照（naive 理论 4× vs 实测）、fig5 matmul 三方 TFLOPS vs N、fig6 config 网格热图（N=1024）、fig7 最优 config 变化 + shared/wave 对账、fig8 fp16 三方对比 + TC 峰值线、fig9 regs/spills vs config 散点（降级证据可视化）。英文图内文字，300 DPI。

**验收标准：** 每图含数据 + 参考线/标注 + 单位；无空图；数字与 JSON 一致。

### 3.4 F4 — sm_75 lowering 拆解文档

**描述**：`notes/triton-lowering-sm75.md`：(1) tl.dot 三 dtype 在 sm_75 的实测 lowering（fp32→fma.rn.f32；fp16→fma.rn.f32 无 mma——PTX 计数证据 + spill 数据）；(2) 上游 MMAv2 Turing 路径源码定位（`MMAv2.cpp` mmaInstrPtxTuring m16n8k8 / callMmaTuringFp16）与「构建未选择」差距的诚实记录；(3) block 编程模型（block-uniform 标量约束、mask/OOB、constexpr 元编程）；(4) autotune 机制（config 尝试、cache 键、首跑编译开销实测）；(5) jit 源文件约束（inspect.getsourcelines → `python -c` 不可用）；(6) 与 CUTLASS/CUDA C++ 的心智模型对照（tile ↔ program、shared ↔ 编译器管理）。

**验收标准：**
- Given 每个 lowering 结论，When 附 PTX 计数/源码行号/实测 JSON，Then 逐项可复核（[源码]/[本机] 分类标注，沿用 AR003 纪律）
- Given fp16 无 mma 结论，When 读者追问「为什么」，Then 文档给出源码路径存在性 + 实测未触发 + 开放问题三层表述（不编造根因）

### 3.5 F5 — 面试笔记

**描述**：`notes/triton-notes.md` 六段结构；数字卡全部来自本 AR JSON + AR002/003 衔接；INTERVIEW-INDEX 增补「DSL/编译栈/三方对照」条目；results.md 逐图中文分析；README 复现步骤。

**验收标准：**
- 笔记数字与 JSON 完全一致（不编造）；六段齐全；红线含「fp16 输入 ≠ tensor core 被使用」「do_bench 与自建协议不可混比」类条目

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 上游完整性 | 文件 | triton/tvm 克隆零改动（mtime 聚类核查） |
| 环境稳定 | 全局 python | 不 pip install 任何包；numba 走既有 venv 子进程 |
| GPU 真实性 | sm_75 | 全部数据真机实测，无模拟 |
| 可重复性 | 基准 | JSON + 脚本一键复现；波动 <5%（长窗口口径） |
| 时间预算 | 全套实验 | E1-E4 总 wall time ≤ 60 分钟（config 全扫描是大头，需留余量） |
| 面试导向 | 笔记 | 六段结构 + 实测数字卡 |

## 5. 约束与假设

- triton 3.8.0.post29 已知限制：fp16 dot 不走 TC（smoke 已证）——E4 以此为**实测对象**而非阻塞项；若开发中发现某 config 触发 mma 则如实更新结论（证据优先）
- `@triton.jit` 需源文件（`python -c` 不可用）——bench 脚本必须落盘运行；PowerShell 多行 heredoc 禁用（AR002 教训）
- numba venv（0.68）无 torch：numba 实验独立计时（numba.cuda 事件），JSON 由主进程合并；三方对比标注「同会话背靠背，时钟态相近但非同一进程」
- WDDM 显示 GPU 会话噪声（AR002/003 教训）：短窗口配置预期噪声大，抽查记录；跨实验比较须同协议
- 假设：numba K2b 复跑应落在 AR002 记录（2.036 TF@1024）的会话噪声带内——超出则调查（venv/驱动变化）
- tvm 文件夹本 AR 不动（后续 AR 或不覆盖，见仓库 README 顺序）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| program | Triton 的 SPMD 实例（≈ CUDA threadblock，grid 的一个元素） |
| BLOCK_SIZE / BLOCK_M... | tl.constexpr 块参数：单个 program 处理的元素/子矩阵尺寸 |
| autotune | @triton.autotune 对 config 集合（block/warps/stages）逐个试跑取最优的机制 |
| lowering | DSL 语句到 GPU 指令（PTX/SASS）的编译下降路径 |
| MMAv2 | Triton 源码中面向 Turing 的 mma.sync m16n8k8 lowering 路径 |
| 三方对照 | numba CUDA（AR002）vs Triton DSL（本 AR）vs cuBLAS（AR002/003） |
