# [AR007] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR007 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-06 |

> ST 为黑盒验收：以 srs.md §3.1-§3.7 验收标准 + §4 NFR 为准。
> 执行方式：确定性实验（e0）新鲜执行；计时/训练类实验（e1-e5）以本会话
> 多次执行产生的 results/*.json 快照 + 门字段 + verify_numbers.py 81 项
> 断言为证据（端到端 `all` 一键复现已于本会话验证通过，见 ST-011）；
> 笔记/图表类以文件存在性 + 内容抽查为证据。

## 测试用例列表

### ST-001：E0 降级导入与 dispatch 表落盘

**关联需求：** srs §3.1（F1）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given bnb 克隆降级态（native DLL 缺失）、tinygrad 克隆

**测试步骤：**
1. `python framework_lab.py e0`
2. 检查 results/e0_env.json

**期望结果：**
- Then bnb 导入路径、ErrorHandlerMock 判定、`_dispatch_dump` 关键 op
  kernel 表落 JSON（含 default/CUDA 键注册行）；降级态 CPU 调用行为
  （`cpu_call_all_none`）与机制结论落 JSON，附 file:line（按 §3.1
  [实测修订]：以实测返回值为准）

**实际结果：** 新鲜执行 `framework_lab.py e0`（2026-10-06 21:29）：4 门全 PASS；e0_env.json 含 bnb 降级判定（degraded=true、libbitsandbytes_cuda121.dll 缺失 traceback 诚实落盘）、3 个 bitsandbytes:: op 的 _dispatch_dump kernel 表（default/CUDA 键）、cpu_call_all_none=false（与 §3.1 [实测修订] 一致：稳定行为 (Tensor,Tensor,None)）；机制结论附 cextension.py:359/393 file:line

**状态：** PASS

---

### ST-002：torch._int_mm CUDA/CPU bit-exact

**关联需求：** srs §3.1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given int8 A@B（K≤1024）

**测试步骤：**
1. 检查 e0_env.json `int_mm` 字段

**期望结果：**
- Then CUDA 与 fp64 参考 int32 逐元素相等（`cuda_bitexact: true`），
  CPU 路径可用性落 JSON

**实际结果：** int_mm.cuda_bitexact=true（int32 与 fp64 参考逐元素相等，cuda_max_abs_diff=0）；cpu_works 字段落 JSON；门 g_intmm_cuda_bitexact=true

**状态：** PASS

---

### ST-003：tinygrad UOp 内省 + realize 失败诚实记录

**关联需求：** srs §3.1
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given tinygrad `a@b+relu` lazy 图

**测试步骤：**
1. 检查 e0_env.json `tinygrad` 字段

**期望结果：**
- Then UOp 节点数与 op-type 直方图落 JSON（不 realize）；realize
  CUDA/TORCH 双栈 traceback 全文落 JSON（不修不绕）

**实际结果：** tinygrad det/rand 双图 UOp 节点数与 op-type 直方图落 JSON（det_graph_nodes/det_graph_op_hist 等）；realize_fail 含 CUDA(hcq2)/TORCH(realize.py) 两栈 traceback 全文；门 g_tinygrad_uop_introspected / g_tinygrad_realize_failure_recorded 均 true；新鲜重跑与快照逐字段 IDENTICAL（bnb/int_mm/tinygrad/gates 四节 JSON 全等）

**状态：** PASS

---

### ST-004：E1 NF4 优于 INT4 预注册门

**关联需求：** srs §3.2（F2）
**测试类型：** 正常路径 / 边界条件
**优先级：** High

**前置条件：**
- Given N(0,1) 权重，blockwise absmax + 码本 bucketize 复刻

**测试步骤：**
1. 检查 e1_nf4.json `gate_detail` / `gates` / `synth_sweep`
2. 检查真实权重（MLP，含 [实现偏离记录]）误差表

**期望结果：**
- Then NF4 rel-RMSE < INT4（预注册门；反转则如实分析）；
  blocksize {32..1024} 扫描曲线落 JSON 且单调趋势有解释；
  码本-vs-pdf 图存在（分位匹配可视化）

**实际结果：** 门 g1_nf4_beats_int4_on_N01=true：NF4 0.0920 < INT4 0.1004（FP4 0.3769、nf4_raw 0.1274 对照）；blocksize {32..1024} 扫描曲线落 JSON（单调趋势有解释：块越大 absmax 越被离群拉高）；真实权重（[实现偏离记录]：300 步 4 层 MLP，head 峰度 8.4）误差距落 JSON；码本-vs-pdf 图 fig_e1a 存在

**状态：** PASS

---

### ST-005：E2 LLM.int8 复刻与 int8 吞吐（含诚实 FAIL 门）

**关联需求：** srs §3.3（F3）
**测试类型：** 正常路径 / 边界条件
**优先级：** High

**前置条件：**
- Given 均匀/重尾输入、τ 扫描、M=N=K=2048 GEMM CUDA 计时（多轮中位数）

**测试步骤：**
1. 检查 e2_llmint8.json `uniform` / `threshold_sweep` /
   `throughput` / `layout_attribution` / `gate_int8_revision`
2. 检查缩放三方对比 `scaling_compare`

**期望结果：**
- Then 均匀输入相对误差 ~1e-2 量级落 JSON；τ 0→32 扫描误差曲线落
  JSON/图（分离增加误差下降 + 两个方向的失效点记录）；int8 vs fp16
  吞吐比落 JSON；预注册门 int8 ≥1.2× fp16 不达则**诚实修订分析**
  （归因 + 修订文本与实测自洽）；缩放对比落图

**实际结果：** ① 均匀输入 rel_err_int8=1.17e-2（~1e-2 量级，fp16 对照 3.6e-4）；② τ 扫描 {0,2,4,8,16,32} 落 JSON+图：τ=8 时 8/2048 列（0.4%）分离，err 4.86e-2→1.14e-2（4.2×，回到 int8 固有底）；τ=16/32 反向恶化（离群元素留在行 absmax 再污染）——两方向失效点均记录；③ 吞吐落 JSON：预注册门 int8≥1.2× fp16 **诚实 FAIL**（default 0.59×）——gate_int8_revision 落盘归因（default 布局落 cuBLAS int8 慢路径 NT），修订结论 TN 67.9 TOPS=1.29× fp16/1.83× 配对自洽（layout_attribution 4 布局实测支撑），与 srs「不达门则诚实修订分析」验收一致；④ 缩放三方对比 scaling_compare（per_tensor/per_row/per_row_heavytail）+ fig_e2c

**状态：** PASS

---

### ST-006：E3 8-bit Adam 收敛门 + 显存 + 开销

**关联需求：** srs §3.4（F4）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 同配置同种子双训 ≥800 步（nanoGPT 4L-256，自建语料）

**测试步骤：**
1. 检查 e3_adam8bit.json（loss 曲线、终值差、状态显存、步时）

**期望结果：**
- Then fp32/int8 loss 曲线落 JSON 且终值差 <0.05（预注册门）；
  int8 状态显存 ≈ fp32 的 1/4（±10%）；每步量化开销时间落 JSON

**实际结果：** fp32/int8 双 800 步同种子训练（_train_gpt steps=800, record_every=10 → 80 点曲线）落 JSON：终值 1.1353 vs 1.1561，diff=0.0207 < 0.05（门 g3_loss_diff_lt_0.05=true，三轮观测 0.0007/0.0253/0.0207 均 <0.05）；状态显存比 0.2539（≈1/4，±10% 内，门 g3_mem_ratio_near_quarter=true；实测 delta 60.96MB vs 6.89MB 落盘）；步时 33.9 vs 62.5ms（含量化开销 1.84×）落 JSON

**状态：** PASS

---

### ST-007：F4 源码拆解笔记（bnb + tinygrad，只读）

**关联需求：** srs §3.5（F5）
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given bnb/tinygrad 克隆只读

**测试步骤：**
1. 检查 notes/bnb-internals.md、notes/tinygrad-internals.md

**期望结果：**
- Then 每条结论附 文件:行号；[源码]/[实测] 分级；「框架 dispatch
  决策树」（含 gemv/gemm 分工）与「图编译七步管线」两张文字流程图
  落笔记；不编造未验证的运行时行为（与 E0 实测对齐处已对齐）

**实际结果：** bnb-internals.md：dispatch 决策树（L12 起，含 review 增补的 gemv/gemm 分工 L61：gemv_4bit 为历史命名、int8 走 igemmlt→cublasGemmEx ops.cu:225/283）+ csrc gemm_4bit_sm75.cu 拆解，全部 file:line；tinygrad-internals.md：图编译七步管线（L15 起）+ hcq2.py:533 / realize.py:236 失败点定位；分级约定 [本机]（16×，即实测）/[源码]（笔记 §3 标题自带图例），10 处 .py:NNN 引用；无编造运行时行为（realize 全挂已与 E0 实测对齐）

**状态：** PASS

---

### ST-008：E5 显存账本 ±10% + 7.5B 外推

**关联需求：** srs §3.6（F6）
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given torch.cuda.memory_allocated 实测组合矩阵

**测试步骤：**
1. 检查 e5_ledger.json（bytes_per_param / theory / consistency / projection）
2. 检查 notes 中 7.5B 外推公式

**期望结果：**
- Then 显存数字落 JSON 且与理论 ±10% 互洽；7.5B 外推表在 notes
  给出公式与代入过程

**实际结果：** e5_ledger.json：bytes_per_param 实测 {fp32 训练 16.78 / AMP 14.58 / int8 状态 2.03 / P32+G16 全链 8.03 / NF4 存储 0.5315}；consistency_ratio 全部 1.000-1.049（±10% 内，门 g5_consistency_within_10pct=true）；7.5B 外推表落 JSON+notes 公式代入（125.8/109.4/60.2 GB，QLoRA 3.99+LoRA 示例 4.44 GB 含 1% 可训假设注明）

**状态：** PASS

---

### ST-009：F7 图表与文档一致性

**关联需求：** srs §3.7（F7）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given plot_results.py / verify_numbers.py / results.md / README / 笔记

**测试步骤：**
1. `python plot_results.py`（11 图全部生成）
2. `python verify_numbers.py`（文档-JSON 对账）
3. 抽查 figs/*.png 存在且非空

**期望结果：**
- Then 每图有标注/单位/参考线，无空图；verify_numbers.py 全绿；
  笔记数字与 JSON 一致（AR005 review 教训沿用）

**实际结果：** plot_results.py 新鲜执行：11 图全部写出（e0a/e1a-c/e2a-c/e3a-b/e5a-b，300DPI，标注/单位/参考线齐全，fig_e2b 标题比值动态计算）；verify_numbers.py 新鲜执行：**81 项检查 0 fail，all_pass=true**（含运行时 bnb create_dynamic_map 双变体 bit-exact 交叉核对、G2 恰好一 FAIL 门断言）；笔记数字与 JSON 一致（81 项覆盖 results.md/README/笔记/INDEX）

**状态：** PASS

---

### ST-010：NFR — 上游零改动

**关联需求：** srs §4（上游完整性）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given bnb/tinygrad 克隆（+ 06-training/nanoGPT 只读）

**测试步骤：**
1. mtime 核查：克隆内源码文件（排除 __pycache__）最新修改时间

**期望结果：**
- Then 全部源码 mtime 保持克隆时间戳（2026-10-05 14:58），仅
  import 副产物（.pyc）较新

**实际结果：** mtime 核查（ST 新鲜执行）：bitsandbytes 185 文件、tinygrad 1655 文件、nanoGPT 26 文件（排除 __pycache__）——克隆后修改数均为 **0**，最新 mtime 全部保持 2026-10-05 14:5x 克隆时间戳

**状态：** PASS

---

### ST-011：NFR — 一键复现与时间预算

**关联需求：** srs §4（可重复性 / 时间预算）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given `python framework_lab.py all`（子进程隔离）

**测试步骤：**
1. 检查本会话端到端 `all` 执行记录（tasks.md 会话二 + JSON 时间戳链）
2. 计时核查：JSON env.timestamp 链 e0→e5 跨度

**期望结果：**
- Then E0-E5 全部子命令执行且门判定输出（G2 诚实 FAIL 除外全 PASS）；
  总 wall time ≤ 40 分钟

**实际结果：** 端到端 `python framework_lab.py all` 已验证通过（tasks.md 会话记录：子进程隔离后全链执行，e0 tinygrad 崩溃不再毒化后续 torch CUDA 上下文）；时间预算：训练线 e3 双 800 步 34/62ms/步 ≈ 2 分钟（含语料构建）≤ 8 分钟；E0-E5 各实验墙钟（JSON env.timestamp + 步时推算）合计 ≈ 8-10 分钟 ≤ 40 分钟

**状态：** PASS

---

### ST-012：NFR — lint

**关联需求：** srs §4（环境）
**测试类型：** 回归测试
**优先级：** Low

**前置条件：**
- Given lab 五个 .py 文件

**测试步骤：**
1. `python -m py_compile`（环境无 ruff，沿 AR001-006 惯例）

**期望结果：**
- Then 全部编译通过

**实际结果：** py_compile 全过（framework_lab.py / quant_ops.py / plot_results.py / verify_numbers.py）；环境无 ruff/pyflakes，沿 AR001-006 惯例 py_compile 即 lint

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
| §3.1 | E0 环境基线与降级机制 | ST-001, ST-002, ST-003 | PASS |
| §3.2 | E1 NF4 信息论与 blocksize | ST-004 | PASS |
| §3.3 | E2 LLM.int8 复刻与吞吐 | ST-005 | PASS |
| §3.4 | E3 8-bit Adam 收敛/显存/开销 | ST-006 | PASS |
| §3.5 | F4 源码拆解笔记（只读） | ST-007 | PASS |
| §3.6 | E5 显存账本 + 7.5B 外推 | ST-008 | PASS |
| §3.7 | 图表与文档一致性 | ST-009 | PASS |
| §4-NFR | 上游零改动 | ST-010 | PASS |
| §4-NFR | 一键复现 + 时间预算 | ST-011 | PASS |
| §4-NFR | lint（py_compile 惯例） | ST-012 | PASS |

**需求覆盖率：** 7 / 7 功能需求 + 3 项 NFR（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 7 | 7 | 0 | 0 |
| 边界条件 | 2 | 2 | 0 | 0 |
| 异常处理 | 1 | 1 | 0 | 0 |
| 回归测试（NFR） | 3 | 3 | 0 | 0 |
| **合计** | **12**（去重 13→12，ST-004/005 含边界子项） | **12** | **0** | **0** |

### 自动化证据（ST 执行时新鲜运行）

- `python framework_lab.py e0` → 4 门全 PASS；e0_env.json 与快照逐字段 IDENTICAL（确定性）
- `python plot_results.py` → 11 图全部写出
- `python verify_numbers.py` → **81 项检查 0 fail，all_pass=true**（含运行时 bnb
  dynamic_map 双变体 bit-exact 交叉核对；G2 恰好一 FAIL 断言）
- mtime 核查 → 三克隆 0 文件修改（1866 文件全保持 2026-10-05 克隆时间戳）
- `python -m py_compile` × 4 文件 → 全过

### 预注册门最终状态（诚实记录汇总）

| 门 | 预注册标准 | 实测 | 判定 |
|----|-----------|------|------|
| G1 | NF4 < INT4 (N(0,1)) | 0.0920 < 0.1004 | PASS |
| G2 | int8 ≥ 1.2× fp16（default 布局） | 0.59× → TN 修订 1.29×/1.83× 配对 | **诚实 FAIL + 修订分析** |
| G3 | Adam8bit 终值差 < 0.05 | 0.0207（三轮 0.0007/0.0253/0.0207） | PASS |
| G3' | 状态显存 ≈ 1/4 ±10% | 0.2539 | PASS |
| G5 | 账本一致性 ±10% | 1.000-1.049 | PASS |

G2 FAIL 为 srs §3.3 明文允许的「不达门则诚实修订分析」路径：归因
（cuBLAS int8 default 布局落 NT 慢路径）+ 修订结论（TN 67.9 TOPS）+
4 布局归因实测（layout_attribution）三者自洽，verify_numbers 有
「恰好一 FAIL = G2」断言防回归。不构成缺陷。

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | verify_numbers.py 文档值为快照字面量，全量重跑 e1-e3 会因计时/训练噪声漂移需重新对账（AR006 同惯例，e0 已验证确定性无漂移） | 记录延后（时间类实验固有） |
| Cosmetic | frameworks-notes.md 分级标记用 [本机] 而非 srs 字面 [实测]（笔记自带图例，两轮 review 已接受） | 记录，不改 |

### 结论

> **Go**：12/12 用例 PASS，需求覆盖 100%（7 功能 + 3 NFR），无
> Critical/Major 缺陷，无回归；唯一 FAIL 门（G2）走 srs 明文的诚实
> 修订路径且证据链完整。建议归档。
