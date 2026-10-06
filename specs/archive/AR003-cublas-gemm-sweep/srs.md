# [AR003] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| AR 主题 | gemm-sweep（CUTLASS Turing tensorop 拆解 + cuBLAS 形状/数据类型真机扫描实验） |
| 关联 SR | 仓库 README「建议顺序」第 3 条：03-gemm CUTLASS 学 tile 层级/epilogue/split-K；AR002 延伸（cuBLAS 单点 → 形状空间扫描） |
| 日期 | 2026-10-05 |
| 状态 | Draft |

## 1. 背景与目标

`03-gemm/cutlass/` 提供了 NVIDIA 官方 GEMM 模板库的完整源码与 94 个示例，其中 `examples/08_turing_tensorop_gemm/` **恰好瞄准本机 GPU**（Turing sm_75、INT8 TC、mma.sync m8n8k16、NumStages=2 双缓冲）。但本机无 nvcc/MSVC，CUTLASS 无法编译运行——只能**源码拆解 + 文档对照**。

同时，AR002 只测了 cuBLAS 在方阵（N=N=N）下的单点性能（FP32 9.5 / FP16 54.9 TFLOPS@2048）。真实工作负载里 GEMM 形状千变万化（decode 的 skinny GEMV、训练的方阵、MoE 的 batch 小矩阵），cuBLAS 在形状空间里的行为（tile 调度、split-K 触发、wave 量化、L2 命中）是面试高频考点，且**本机可真机实测**（torch cuBLAS + `torch._int_mm` INT8 TC 路径）。

目标：(a) 把 CUTLASS Turing 示例拆解成可讲解的 tile 层级/流水线/epilogue 知识；(b) 用 cuBLAS 在形状×数据类型空间做系统性真机扫描，产出可解释图表（wave 量化、skinny 边界、split-K 收益、INT8/FP16/FP32 TC 断层），全部数字真机实测；(c) 面试导向笔记。

## 2. 需求范围

**In Scope：**
- F1：基准框架 `bench_sweep.py`（全局 python，torch cuBLAS）：形状/数据类型参数化扫描，CUDA Event 计时，JSON 输出；含 dtype 路径探测（`torch._int_mm` INT8 可用性）与正确性对拍门
- F2：实验组 E1–E7（详见 §3），覆盖：形状网格热图、skinny GEMV 边界、split-K 代价/收益、batch GEMM、数据类型 TC 断层、L2 效应、layout（TN/NT/NN）效应
- F3：`plot_results.py` 生成可解释图（≥7 张 PNG：热图 + wave 量化标注 + 峰值线 + 带宽模型线），图内英文、分析中文（results.md）
- F4：CUTLASS 08_turing 拆解文档：tile 层级（TB 128×256×64 → warp 64×64×64 → mma 8×8×16）与本机硬件参数逐项对账（shared 用量、TB 数、wave 数），epilogue/split-K/swizzle 机制，`device::Gemm` 15 参数模板逐个释义；对照 `media/docs/cpp/efficient_gemm.md` 层级理论
- F5：面试笔记 `03-gemm/notes/gemm-sweep-notes.md`（六段结构 + 实测数字卡）+ INTERVIEW-INDEX 增补 + `results.md` + `README.md`

**Out of Scope：**
- 不修改 cutlass 任何上游文件（零改动）
- 不编译运行 CUTLASS（无 nvcc；只做源码拆解 + 文档对照）
- 不做 Triton kernel（AR004 主题；本 AR 仅 cuBLAS/`_int_mm`）
- 不做 FP8/BF16/TF32（sm_75 硬件不支持，笔记中仅作代际对比叙述）
- 不改动全局 Python 环境（torch/numpy/matplotlib 现状即可）

## 3. 功能需求

### 3.1 F1 — 扫描基准框架

**描述**：`bench_sweep.py` 支持 `--exp E1..E7` 子实验，统一 bench 纪律：pinned H2D、~1s 持续 warmup（时钟爬坡）、3 遍×自适应 reps 取中位数、JSON 落盘 `results/*.json`（实验名、形状、dtype、time_ms 中位数、TFLOPS/GB/s、reps、环境）。

**验收标准：**
- Given 任一子实验，When 重复执行两次，Then 同配置 TFLOPS 波动 < 5%（沿用 AR002 修订口径：显示 GPU 会话噪声带已知，抽查记录）
- Given `torch._int_mm`，When 启动时探测，Then 记录可用性并做与 int32 参考对拍（容差 0，INT8 精确）

> **修订（2026-10-05，ST 执行期，附 E5 两次完整复跑证据）**：第一条验收细化为「<5% 适用于**测量窗口足够长**（时钟爬坡完成后）的配置」。实测 E5 复跑：@1024/@4096 差 <1.5% ✓；@512/@2048 差 10-17% ✗——机制是 2048³ 单次 1.7ms、10 reps 循环 ~17ms，**测量窗口短于显示 GPU 的时钟爬坡时间**，短窗测的是「当前时钟态」而非稳态；@4096（70ms 窗口）稳定。**结论稳健性不受影响**：FP32→FP16 断层两跑 6.9×/6.4×、FP16@4096 功率墙回落两跑方向一致。消除需锁时钟（管理员）或加长窗口（reps 上限 10 受 srs 计时纪律约束），记录为已知限制，全部观测值入 results.md 多观测表。

### 3.2 F2 — 七组实验

**E1 形状网格热图**：M,N ∈ {64,128,256,512,1024,2048,4096}× 同集合（K=2048 固定），FP32 TFLOPS 热图；标注 wave 量化边界（TB tile 128×128 估算：⌈M/128⌉×⌈N/128⌉ vs 48 SM 整数倍）。
- 验收：热图 ≥49 格全实测；wave 边界带可解释（同 wave 内 TFLOPS 平坦、跨 wave 台阶）

**E2 skinny 边界**：M ∈ {1,2,4,8,16,32,64,128,256,512,1024}（N=4096,K=4096 固定），FP32；TFLOPS + GB/s 双轴，标 GEMV 带宽极限模型线（2·M·K+2·M·N+2·K·N 字节 / 448 GB/s）。
- 验收：M=1 实测带宽 ≥ 40% 峰值可解释；M 增大时算术强度拐点与 roofline 拐点（25 FLOP/B）一致性有分析

**E3 split-K 代价/收益**：固定 (M=64,N=64,K∈{4096,8192,16384})：full-K 单 GEMM vs 手工 2/4/8 路 split（各路独立 GEMM + 相加）耗时对比。
- 验收：能展示「并行度饥饿时 split 有收益 / epilogue 归并代价」的 crossover；与 CUTLASS split-K 机制（reduction workspace）对照分析

**E4 batch GEMM**：bmm B∈{64} 个 (256×256)×(256×256) vs 单个大 GEMM 等价 flop 对比；小 batch {1,2,4,8,16,32,64} 扫描。
- 验收：batch 调度开销/收益可量化；与 CUTLASS batched swizzle 对照

**E5 数据类型断层**：方阵 N∈{512,1024,2048,4096}，FP32 vs FP16（半精度存储）vs INT8（`torch._int_mm`，int8 存储）；三峰值线（11.2 / 89.2 / 178.4 TOPS）。
- 验收：INT8 路径真机跑通（否则记录降级为 FP16 对比 + INT8 理论分析）；断层倍数与 AR002 口径衔接

**E6 L2 效应**：4096³ FP32 GEMM，迭代间 flush L2（写 64MB dummy buffer）vs 不 flush；对照 L2 4MB 容量的命中模型。
- 验收：两种条件 TFLOPS 差异实测并给出可解释分析（或差异 <2% 亦有结论：工作集远超 L2 时 flush 影响小）

**E7 layout 效应**：同形状 (2048³)，A/B 的 row/col major 四组合（torch 转置视图模拟 TN/NT/NN/TT）。
- 验收：四组合差异实测；与 cuBLAS 各 layout kernel 路径/共享内存装载效率对照分析

### 3.3 F3 — 图表

**描述**：`plot_results.py` 读 JSON 产 `figs/*.png`；E1 热图（含 wave 网格叠加）、E2 双轴+带宽模型线、E3 柱状 crossover、E4 batch 曲线、E5 dtype 断层 + 三峰值线、E6 对比柱、E7 layout 柱状。英文图内文字，300 DPI。

**验收标准：** 每张图含数据 + 参考线/标注 + 单位；无空图；数字与 JSON 一致。

### 3.4 F4 — CUTLASS 拆解文档

**描述**：`notes/cutlass-turing-dissection.md`（或并入 gemm-sweep-notes 一章）：以 08 示例为线索，逐层拆解：
1. 数据流：global→register→shared→register→mma→register→global（示例注释原文的 pipeline 相位）
2. tile 层级对账表：TB 128×256×64（shared 2 stages×(128×64+64×256)×1B=48KB ≤ 64KB ✓）、warp 64×64×64（4 warp/128 线程…按 128×256/64×64=8 warp 核算线程数）、mma 8×8×16（INT8 mma.sync，每 SM 8 个 TC × 什么节拍 → 178 TOPS 换算）
3. `device::Gemm` 15 模板参数逐个释义（dtype/layout/MMAOp/SmArch/三级 tile/epilogue/swizzle/stages）
4. epilogue LinearCombination 与 vector width（128/32=4 元素/访问）
5. swizzle（identity）与 wave、split-k 参数、`can_implement`→`initialize`→launch 生命周期
6. 对照 `efficient_gemm.md` 的 Hierarchical Structure/Optimizations 章节

**验收标准：**
- Given 拆解中每个数字（shared 用量、warp 数、寄存器估算），When 对照示例源码与本机规格，Then 逐项可复核（列出计算式）
- Given mma 8×8×16，When 换算到本机 178.4 INT8 TOPS，Then 换算链展示（含时钟/SM/TC 数）

### 3.5 F5 — 笔记与日志

**描述**：`03-gemm/notes/gemm-sweep-notes.md` 六段面试结构（高频问法/追问链/数字卡片/手写骨架/红线清单/60秒电梯陈述）；数字卡片全部来自本 AR 实测 JSON + AR002 衔接；INTERVIEW-INDEX 增补「GEMM 形状空间/split-K/CUTLASS 层级」条目；`results.md` 逐图中文分析；`README.md` 复现步骤。

**验收标准：**
- 笔记引用数字与 JSON 完全一致（不编造）
- 六段结构齐全；红线清单含「方阵 TFLOPS 不能外推 skinny」「split-K 不是免费午餐」类陷阱条目

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 上游完整性 | 文件 | cutlass 零改动（mtime 聚类核查） |
| GPU 真实性 | sm_75 | 全部数据真机实测，无模拟 |
| 可重复性 | 基准 | JSON + 脚本一键复现；波动 <5%（会话噪声带口径同 AR002） |
| 时间预算 | 全套实验 | E1–E7 总 wall time ≤ 45 分钟 |
| 面试导向 | 笔记 | 六段结构 + 实测数字卡 |

## 5. 约束与假设

- 无 nvcc → CUTLASS 只拆解不运行；「本机 CUTLASS 实测」不存在，笔记表述须区分「源码事实」与「本机 cuBLAS 实测」
- `torch._int_mm` 可用性未验证：若不可用，E5 降级为 FP32/FP16 + INT8 理论换算（并在 results.md 记录降级原因）
- WDDM 显示 GPU 会话噪声（AR002 教训）：E6 的 L2 flush 对比属微小效应量级，须多遍观测方向稳健再下结论
- 假设：最大形状 4096³ FP32 ≈ 3×64MB ≈ 192MB 显存无压力；16GB 充裕
- triton 已解锁（3.8.0.post29 全局）但本 AR 不用（AR004 主题）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| wave | 一波能装满全部 SM 的 threadblock tile 集；跨 wave 出现量化台阶 |
| split-K | 把 K 维切给多个 CTA 并行累加，换取并行度、付出归并代价 |
| skinny GEMM | M 或 N 远小于 K 的长条形 GEMM（decode 场景），算术强度低 |
| TN/NT/NN/TT | cuBLAS 布局组合记法（本 AR 用 torch 转置视图模拟） |
| mma.sync m8n8k16 | Turing INT8 tensor core 指令形状（08 示例 ShapeMMAOp） |
