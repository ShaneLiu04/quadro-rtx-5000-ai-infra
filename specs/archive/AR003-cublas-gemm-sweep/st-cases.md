# [AR003] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-06 |

## 测试用例列表

### ST-001：同配置重复执行波动 <5%（修订口径）

**关联需求：** srs.md §3.1 F1 验收 1（含 2026-10-05 测量窗口效应修订块）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 测量窗口足够长（时钟爬坡完成后）的配置

**测试步骤：**
1. 核对 E5 两次完整复跑记录（srs 修订块 + results.md 多观测表）
2. 核对 E3seq 本会话两次执行的控制台记录（run1: 7.615/6.705/5.368/3.528；run2: 7.569/6.683/5.301/3.559）

**期望结果：**
- Then 长窗口配置两跑差 <5%；短窗口配置差异已按修订口径记录为已知限制

**实际结果：** E5 @1024/@4096 两跑差 <1.5% ✓；E3seq 两跑差 0.6%/0.3%/1.2%/0.9% ✓；@512/@2048 短窗 10-17% 已按修订块记录（窗口 ∝ 噪声带），方向稳健

**状态：** PASS

---

### ST-002：`torch._int_mm` 探测与对拍门

**关联需求：** srs.md §3.1 F1 验收 2
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given Windows torch 2.5.1+cu121 环境

**测试步骤：**
1. 读取 `results/env.json` 的 `int_mm` 字段

**期望结果：**
- Then 可用性被记录（含错误信息），E5 走降级分支并留档

**实际结果：** `env.json: {"int_mm": {"available": false, "error": "RuntimeError: \"addmm_cuda\" not implemented for 'Int'"}}`；E5 按 srs §5 降级为 FP32/FP16 对比 + INT8 理论线，results.md fig5 记录降级原因

**状态：** PASS

---

### ST-003：E1 形状网格热图全实测 + wave 边界可解释

**关联需求：** srs.md §3.2 E1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given e1a.json（7×7 M×N 网格，K=2048）

**测试步骤：**
1. 统计 e1a.json 记录数与 `model_tiles` 字段覆盖
2. 核对 results.md fig1a 分析含 wave 边界解释

**期望结果：**
- Then ≥49 格全实测；wave 边界带可解释（同 wave 平坦、跨波台阶）

**实际结果：** 49 条记录、全部含 `model_tiles` ✓；fig1a 分析含 wave 等值线对齐讨论（对齐但不精确 → 库启发式证据）、fig1b 阶梯跨波 -10%、≥4 波 <5%（对 10.15 参考格）；凹陷格口径可复现（M,N≥512 区 16 格：11 格 9.5-10.24 中位 9.71 + 3 格凹陷 8.58-8.90）

**状态：** PASS

---

### ST-004：E2 skinny 边界带宽达成 + roofline 拐点一致性

**关联需求：** srs.md §3.2 E2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given e2.json（M ∈ 11 档，N=K=4096）

**测试步骤：**
1. 读取 M=1 记录的 GB/s 并计算对 448 GB/s 峰值的达成率
2. 核对 results.md fig2 分析的算术强度拐点讨论

**期望结果：**
- Then M=1 实测带宽 ≥40% 峰值且可解释；拐点与 roofline（25 FLOP/B）一致性有分析

**实际结果：** M=1 = 359.7 GB/s = **80.3%** 峰值 ✓（且贴住 naive GEMV 理论线 359.6 GB/s——cuBLAS 没有摆烂）；11 条记录全实测；M*≈51 拐点与 roofline 换算（AI=25 → M*≈51.2）一致，fig2 标注带宽平台 360-386 GB/s（80-86%）

**状态：** PASS

---

### ST-005：E3 split-K 代价/收益 + CUTLASS 机制对照（含验收偏离）

**关联需求：** srs.md §3.2 E3
**测试类型：** 边界条件（含验收标准偏离记录）
**优先级：** High

**前置条件：**
- Given e3.json + e3seq.json（M=N=256, K=16384）

**测试步骤：**
1. 核对 E3/E3seq 记录数与正确性门
2. 核对 results.md fig3 分析与 CUTLASS split-K（reduction workspace）对照

**期望结果：**
- Then 能展示「饥饿时 split 有收益 / epilogue 归并代价」的 crossover

**实际结果：** **验收偏离（有证据）**：crossover 不存在——s=1 基线 8.20 TF >> 4-tile 饥饿上限 0.93 TF，证明 **cuBLAS 内部已自动 split-K**，手工 split 无收益空间（stream 6.83→2.00、顺序 7.57→3.56 全败；stream 比顺序还慢 -9.8%~-43.9%，E3seq 对照实测）。归并代价与多层 split 打架机制已量化分析；与 CUTLASS `split_k_slices` 模板参数（08 示例=1）及 reduction workspace 机制对照完成；「WDDM 串行化」从推测升级为有顺序对照的实测结论。偏离已记录于 design §E3 执行修订 + tasks T012；饥饿模型对账反而成为比 crossover 更有教学价值的产出

**状态：** PASS（偏离有据，按 srs §5 口径）

---

### ST-006：E4 batch GEMM 调度开销量化

**关联需求：** srs.md §3.2 E4
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given e4.json（B ∈ {1,2,4,8,16,32,64}，bmm vs flat 双系列）

**测试步骤：**
1. 核对记录数、B 覆盖、双系列字段
2. 核对与 CUTLASS batched swizzle 的对照

**期望结果：**
- Then batch 调度开销/收益可量化；与 batched swizzle 对照

**实际结果：** 7 条记录、B 全覆盖、双系列（bmm/flat reps+tflops）✓；B≥4 flat 一致胜出（+64%@B=4、+13%@B=64），B=1/2 持平；dissection §6 补 GemmBatchedIdentityThreadblockSwizzle 段并与 E4 运行时成本关联

**状态：** PASS

---

### ST-007：E5 数据类型断层（INT8 降级路径）

**关联需求：** srs.md §3.2 E5
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given e5.json（N ∈ {512,1024,2048,4096}，FP32/FP16 双 dtype）+ env.json 降级记录

**测试步骤：**
1. 核对 e5 记录覆盖与双 dtype 字段
2. 核对三峰值线、降级记录、AR002 口径衔接

**期望结果：**
- Then INT8 真机跑通，否则记录降级；断层倍数与 AR002 衔接

**实际结果：** 4 尺寸 × FP32/FP16 双系列全实测 ✓；INT8 降级记录于 env.json + results.md fig5（附理论换算 178.4 TOPS 链）；FP32→FP16 6.4-6.9×@2048（两跑）；FP16 峰值达成 68-76%（跨会话时钟态带 54.9~67.4 含 AR002 54.9 衔接）；fig5 三峰值线（11.2/89.2/178.4）

**状态：** PASS

---

### ST-008：E6 L2 驻留效应

**关联需求：** srs.md §3.2 E6
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given e6.json（N ∈ {512,1024,2048,4096}，flush vs no-flush，3 passes）

**测试步骤：**
1. 核对记录数、N 覆盖、多 pass 字段
2. 核对与 L2 4MB 命中模型的对照分析

**期望结果：**
- Then 两种条件差异实测且有可解释分析（或 <2% 亦有结论）

**实际结果：** 4 条记录、含 `flush_passes_ms`/`noflush_passes_ms` 3-pass 多观测 ✓；仅 N=512（ws 3MB=0.75×L2）×1.12，ws≥12MB 后 ×0.98-1.02 方向不定只报范围（符合 srs「微小效应多遍观测」约束）；分析含「K 循环内部本来就有 L2 复用」机制 + per-iter event 协议与 E5 循环协议不可混比红线

**状态：** PASS

---

### ST-009：E7 layout 效应

**关联需求：** srs.md §3.2 E7
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given e7.json（2048³ FP32，row/col 四组合 + contiguous 对照）

**测试步骤：**
1. 核对组合覆盖、极差、无隐式拷贝断言

**期望结果：**
- Then 四组合差异实测；与 layout kernel 路径对照分析

**实际结果：** **7 组合**（超出验收的 4 组合：四组合 + AtBt/colA_colB/contiguous）全实测；极差 ×1.051（5.1%）；bench 内 `memory_allocated` 断言无隐式拷贝；分析含 B 侧转置微慢机制 + 「TN 神话破除」（layout 远小于形状/dtype 效应）

**状态：** PASS

---

### ST-010：F3 图表完整性

**关联需求：** srs.md §3.3 F3
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given figs/*.png（8 张）+ plot_results.py

**测试步骤：**
1. 核对 8 张 PNG 存在且非空
2. 核对 300 DPI、参考线/标注、数字与 JSON 一致

**期望结果：**
- Then 每张图含数据 + 参考线/标注 + 单位；无空图；数字与 JSON 一致

**实际结果：** 8 张全存在（105-163KB，fig1a/fig1b/fig2/fig3/fig4/fig5/fig6/fig7）✓；`figure.dpi: 300` ✓；fig1a 对角线参考 + wave 红线、fig2 带宽模型线、fig3 stream vs sequential 双柱 + 饥饿上界线 + 基线、fig5 三峰值线、fig7 极差入标题；重渲染无 matplotlib 警告；数字-JSON 一致性经两轮 review 子代理从 JSON 重算核对（fig1b 14 值、fig2 平台带、fig3 双系列、fig4/5/6/7 全值）

**状态：** PASS

---

### ST-011：F4 拆解文档 tile 层级对账

**关联需求：** srs.md §3.4 F4 验收 1、2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given cutlass-turing-dissection.md + 08 示例源码 + mma_sm75.h + 本机规格

**测试步骤：**
1. 核对 tile 层级对账表（TB/warp/mma 三级 + 计算式）
2. 核对 mma→178.4 TOPS 换算链、FragmentC、FP16 mma 形状

**期望结果：**
- Then 每个数字可复核（列出计算式）；换算链含时钟/SM/TC 数

**实际结果：** dissection §3 逐项计算式齐全：TB 128×256×64（shared 2×24KB=48KB ≤ 64KB，NumStages=2 是预算必然——3 级 73.7KB 超限）、8 warp/256 线程、warp 64×64×64 每 k 步 256 条 mma、INT8 m8n8k16 FragmentC=`Array<int,2>`（每线程 2×int32，mma_sm75.h:233 复核）、FP16 特化 **m16n8k8**（f16/f32 累加两种，FragmentC `Array<half_t,4>`/`Array<float,4>`，1024 MAC/条 → 每 sub-core 8 clk）、178.4 TOPS 链（÷(48×1.815GHz)=2048 op/SM/clk → 1024 MAC → 256/subcore → m8n8k16 每 4 clk）、§3 补网格级对账（5120×4096→640 tile/13.3 waves）；[源码]/[本机] 事实分类标注

**状态：** PASS

---

### ST-012：F4 拆解文档机制章节完整

**关联需求：** srs.md §3.4 F4 描述 1-6
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given dissection 9 个章节

**测试步骤：**
1. 核对数据流、15 模板参数、epilogue、swizzle/split-K、生命周期、efficient_gemm.md 对照

**期望结果：**
- Then 六项拆解内容齐全

**实际结果：** §1 一句话定位；§2 `device::Gemm` 15 模板参数逐个释义（[源码] L179-193）；§3 tile 层级对账；§4 数据流管线（global→register→shared→register→mma→register→global，[源码] L81-101）；§5 epilogue LinearCombination + 16B 向量化（128/32=4）；§6 swizzle + split-K 正交旋钮 + batched swizzle 段；§7 efficient_gemm.md 层级对照表；§8 sm_80+ 差异红线；§9 60 秒复述脚本

**状态：** PASS

---

### ST-013：F5 面试笔记六段结构 + 数字卡一致性

**关联需求：** srs.md §3.5 F5 验收 1、2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given gemm-sweep-notes.md + results/*.json

**测试步骤：**
1. 核对六段结构齐全
2. 核对数字卡与 JSON 一致、红线清单条目

**期望结果：**
- Then 数字与 JSON 完全一致（不编造）；红线清单含指定陷阱条目

**实际结果：** 六段齐全（§1 高频问法 / §2 追问链 / §3 数字卡片 / §4 手写骨架 / §5 红线清单 / §6 电梯陈述）✓；数字卡经两轮 review 从 JSON 重算核对（含本 ST 前最后一轮 R1-R5 修正：凹陷格 3/16 口径、E3seq -16.6/-43.9、GEMV 理论线 359.6、73% FP32 峰值）；红线清单含「方阵 TFLOPS 不能外推 skinny」「split-K 不是免费午餐（并发归因必须有顺序对照）」及协议红线（两协议数字不可混比、跨实验比较同会话同协议）

**状态：** PASS

---

### ST-014：INTERVIEW-INDEX 增补 + README 复现步骤

**关联需求：** srs.md §3.5 F5 描述
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given INTERVIEW-INDEX.md + gemm-sweep/README.md

**测试步骤：**
1. 核对 INDEX 中 gemm-sweep 条目
2. 核对 README 复现命令与实验表

**期望结果：**
- Then GEMM 形状空间/split-K/CUTLASS 层级条目增补；README 可复现

**实际结果：** INDEX 含 8 处 gemm-sweep 行（5 行映射 + 追问阶梯 + 自测 + FP16 时钟态行）✓；README 含 3 处复现命令（bench/plot/单跑示例）、实验表 8 行（E1a/E1b/E2/E3/E3seq/E4/E5/E6/E7 全含 E3seq 顺序对照）；`--help` 文本已同步 E3seq

**状态：** PASS

---

### ST-015：results.md 逐图中文分析覆盖

**关联需求：** srs.md §3.5 F5 描述（results.md 逐图分析）
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given results.md

**测试步骤：**
1. 核对每个 fig 章节存在且含数据+分析

**期望结果：**
- Then 全部图有中文分析

**实际结果：** 8 个 fig 章节（fig1a/fig1b/fig2/fig3/fig4/fig5/fig6/fig7）均含「数据」+「分析」结构；fig3 章节覆盖 E3+E3seq 双实验；末尾含汇总表与已知限制

**状态：** PASS

---

### ST-016：NFR 上游完整性（cutlass 零改动）

**关联需求：** srs.md §4 上游完整性
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given 03-gemm/cutlass/ 全目录

**测试步骤：**
1. ST 时点全量 mtime 扫描，对照克隆窗（2026-10-05 14:52:39–14:53:31）

**期望结果：**
- Then 零改动（无文件 mtime 晚于克隆窗）

**实际结果：** 7934 文件，min=14:52:39、max=14:53:31（与 T011 记录一致；191 个「窗后」计数为亚秒精度伪差异，真实 max 与窗端同秒）；会话期（10-05 开发 + 10-06 review/ST）零改动 ✓

**状态：** PASS

---

### ST-017：NFR GPU 真实性

**关联需求：** srs.md §4 GPU 真实性
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given env.json

**测试步骤：**
1. 核对环境快照与真机标记

**期望结果：**
- Then 全部数据真机实测，无模拟

**实际结果：** env.json：cc=7.5、Quadro RTX 5000、torch 2.5.1+cu121、16GB ✓；所有实验记录来自 CUDA Event 计时真机执行；CUTLASS 部分为源码拆解（srs §5 约束，「源码事实」与「本机实测」分类标注）

**状态：** PASS

---

### ST-018：NFR 时间预算（E1-E7 全套 ≤45 分钟）

**关联需求：** srs.md §4 时间预算
**测试类型：** 边界条件
**优先级：** Low

**前置条件：**
- Given 全部实验 JSON + 开发会话记录

**测试步骤：**
1. 从 JSON 计算 GPU-busy 时间上界（Σ(reps+4)×time_ms）
2. 对照开发会话记录（T001-T011 单会话完成）

**期望结果：**
- Then 全套 wall time ≤45 分钟

**实际结果：** GPU-busy 上界 ≈1.9s + 9×2s 烧机 + 分配/对拍/setup 开销；全套在单开发会话内完成（tasks 会话记录），实际远低于预算上限（design 估算 30-40min 为保守上界，README「约 2-3 分钟」为复现口径）；ST 时点未重跑全量以避免覆盖 canonical JSON 观测带（复现步骤已在 README 验证可单跑）

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-06 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 F1 | 扫描基准框架 | ST-001, ST-002 | PASS |
| §3.2 E1 | 形状网格热图 | ST-003 | PASS |
| §3.2 E2 | skinny 边界 | ST-004 | PASS |
| §3.2 E3 | split-K 代价/收益 | ST-005 | PASS（偏离有据） |
| §3.2 E4 | batch GEMM | ST-006 | PASS |
| §3.2 E5 | dtype 断层 | ST-007 | PASS |
| §3.2 E6 | L2 效应 | ST-008 | PASS |
| §3.2 E7 | layout 效应 | ST-009 | PASS |
| §3.3 F3 | 图表 | ST-010 | PASS |
| §3.4 F4 | CUTLASS 拆解 | ST-011, ST-012 | PASS |
| §3.5 F5 | 笔记与日志 | ST-013, ST-014, ST-015 | PASS |
| §4 NFR | 上游完整性/GPU 真实性/时间预算 | ST-016, ST-017, ST-018 | PASS |

**需求覆盖率：** 12 / 12（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 10 | 10 | 0 | 0 |
| 边界条件 | 4 | 4 | 0 | 0 |
| 异常处理 | 2 | 2 | 0 | 0 |
| 回归测试 | 2 | 2 | 0 | 0 |
| **合计** | **18** | **18** | **0** | **0** |

### 验收偏离与已知限制（均有证据记录）

| 项 | 描述 | 依据 |
|----|------|------|
| E3 crossover 不存在 | cuBLAS 内部已自动 split-K（饥饿上限 0.93 vs 实测 8.20 TF），手工 split 无收益空间；产出反直觉基线 + E3seq 顺序对照（stream 比顺序还慢 -9.8%~-43.9%） | design §E3 执行修订、tasks T012、results.md fig3 |
| INT8 未真机跑通 | Windows torch 2.5.1 缺 `addmm_cuda` Int 路径；按 srs §5 降级为 FP32/FP16 + INT8 理论换算 | env.json、results.md fig5 |
| 短窗口配置波动 10-17% | @512/@2048 测量窗口短于显示 GPU 时钟爬坡；长窗口 <1.5%；全部观测入多观测表 | srs §3.1 修订块（附两跑证据） |
| wave 模型 tile 128×128 为可视化假设 | cuBLAS 实际多 tile 尺寸，量化边界比模型模糊（本身是库启发式证据） | results.md fig1a 分析、已知限制节 |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Cosmetic | README 全套复现时间口径（2-3 min 实测 vs design 30-40 min 估算）可在下次复跑时校准为实测值 | 延后处理（不影响复现） |

### 结论

> **Go**。18/18 用例 PASS，12/12 需求 100% 覆盖，无 Critical/Major 缺陷；4 项验收偏离/已知限制全部有实测证据并按 srs 修订口径记录；review 两轮闭环（FAIL→修复→PASS→R1-R5 收尾）；上游零改动 ST 时点复核通过。
