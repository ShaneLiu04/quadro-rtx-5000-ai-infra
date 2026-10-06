# [AR004] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-06 |

> 本 AR 为 GPU 实验型交付（Python）。ST 为黑盒验证：以 srs §3.1-3.5 验收标准与
> §4 非功能需求为基准，采用「新鲜 GPU 执行 + JSON/文件对账」两类证据。
> 执行器：`st-verify.py`（本轮新鲜执行）+ 既有运行记录（今日全量 E1-E4 复跑）。

## 测试用例列表

### ST-001：E1 长窗口重复性 <5%

**关联需求：** srs §3.1 F1 验收 1（连跑两次波动 <5%，WDDM 会话噪声带口径）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given E1 干净数据（e1.json），When 新鲜测量同配置

**测试步骤：**
1. st-verify.py 新鲜测量 torch add @64M 与 triton BLOCK=1024 @64M
2. 与 e1.json 记录值对比

**期望结果：**
- Then GB/s 波动 <5%

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-002：numba 子进程 JSON 合并与会话标注

**关联需求：** srs §3.1 F1 验收 2（venv 版本信息 + 同会话）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given e1na.json / e3na.json 存在

**测试步骤：**
1. 读取两个 JSON 的 env 与 same_session 字段

**期望结果：**
- Then 含 numba 版本（0.68.x）且 same_session=true

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-003：正确性门新鲜执行（四类）

**关联需求：** srs §3.1 F1 验收 3（add 精确 / softmax ≤1e-6 含极端值 / FP32 ≤1e-5 / FP16 1.5 ulp@max|ref|，含 2026-10-06 修订）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 全新进程，未复用任何缓存判定

**测试步骤：**
1. st-verify.py 新鲜执行：triton add vs torch（torch.equal）
2. softmax 含 ±1000 极端行 vs double 参考（≤1e-6）
3. matmul FP32 vs torch 参考（rel-err ≤1e-5）
4. matmul FP16 vs fp16 参考（≤1.5 ulp@max|ref|）

**期望结果：**
- Then 四类门全部通过

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-004：E1 交付完整性（三方曲线 + 参考线 + BLOCK 结论）

**关联需求：** srs §3.2 E1 验收
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. figs/fig1_e1_bandwidth.png 存在；plot_results.py 含 448 GB/s 峰值线与 L2 4MB 拐点线
2. e1.json 含三方变体全尺寸记录；results.md 含 BLOCK_SIZE 结论

**期望结果：**
- Then 曲线/参考线/拐点标注齐全，BLOCK 有结论（实测：大 N 三 BLOCK 无差异）

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-005：E2 融合收益量化与原生差距分析

**关联需求：** srs §3.2 E2 验收（含 2026-10-06 修订：实测 2.45× 口径）
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. e2.json 含 triton_nw4/nw8、torch_native、naive_5pass 四变体 × 5 行宽
2. results.md 含 naive 2.45×（L2 折扣口径）与 triton vs 原生 1.2-1.6× 分析

**期望结果：**
- Then 融合收益按修订口径量化；原生差距有机制分析（PTX 取证 + 占用率）

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-006：E3 config 扫描与失败记录

**关联需求：** srs §3.2 E3 验收（热图 + 失败格原因 + 最优 config 表 + 资源对账 + 三方达成率）
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. e3cfg.json @1024：valid 记录 + oor_reason 记录合计 = 网格总数（76+32=108）
2. oor_reason 全部为 shared 剪枝公式触发
3. figs/fig6_e3_heatmap.png 存在；fig7 存在
4. results.md 含最优 config 随 N 迁移 + shared/wave 对账表 + 87% 达成率

**期望结果：**
- Then 网格精确闭环；失败原因落盘；热图/对账/达成率齐全

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-007：E3 GROUP_M 对照

**关联需求：** srs §3.2 E3 验收（GROUP_M ∈ {1,8}）
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. e3.json 含 triton_best_gm1 与 triton_best_gm8 记录（4 尺寸 × 2）

**期望结果：**
- Then 对照数据存在且有结论（实测 ±5%，@2048 gm1 快 4.5%）

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-008：E4 无 TC 差距量化 + mma=0 证据

**关联需求：** srs §3.2 E4 验收（差距倍数 + mma=0/spill 支撑 + 与 smoke 一致）
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. e4.json：每 config 记录 mma_count/ldmatrix_count/n_regs/n_spills，全 26 config × 3 尺寸 mma=0
2. cublas_fp16 记录存在；@2048 差距 = 63.072/6.639 ≈ 9.5×
3. fma 静态计数 = BM×BN×BK/threads 对账

**期望结果：**
- Then 降级结论有完整计数证据；差距倍数量化；与 smoke 预实验（16 config mma=0）方向一致

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-009：E4 PTX 证据存档

**关联需求：** srs §3.2 E4 验收（PTX 片段存档；交付形式偏离已记录于 tasks T005）
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. results/best_fp16_kernel.ptx 存在且头部含计数注释
2. 文件内 mma.sync=0、fma.rn.f32=512、cvt.f32.f16=16

**期望结果：**
- Then PTX 证据可独立复核

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-010：F3 图表数量与规格

**关联需求：** srs §3.3 F3（≥8 图、数据+参考线+单位、数字与 JSON 一致）
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. figs/ 下 9 个 png 存在
2. plot_results.py 含 448/11.2/89.2 参考线与 L2 拐点
3. Review C1/C2 已对账（30 项抽查 28+2 修正后全一致）

**期望结果：**
- Then ≥8 图齐全、无空图、口径一致

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-011：F4 lowering 文档六要素

**关联需求：** srs §3.4 F4（lowering 证据/源码定位/block 模型/autotune/jit 约束/心智对照 + [本机]/[源码] 标注 + fp16 三层表述）
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. notes/triton-lowering-sm75.md 存在
2. 含：PTX 证据链、MMAv2 源码路径（third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp）、block 模型对照表、autotune 机制、jit 源文件约束（inspect.getsourcelines）、CUDA C++ 对照
3. 含 [本机]/[源码] 标注；fp16 结论为「源码存在 + 实测未触发 + 开放问题」三层表述

**期望结果：**
- Then 六要素齐全、标注纪律执行、无编造根因

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-012：F5 六段笔记 + INDEX 增补 + README

**关联需求：** srs §3.5 F5（六段结构、数字一致、红线含 fp16≠TC 与 do_bench 条目）
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. notes/triton-notes.md 含：高频问法/追问链/数字卡片/手写骨架/红线清单/60 秒电梯陈述
2. 红线含「fp16 输入 ≠ tensor core 被使用」「do_bench 与自建协议不可混比」
3. INTERVIEW-INDEX.md 含 triton-lab 条目；triton-lab/README.md 含复现步骤

**期望结果：**
- Then 六段齐全、红线到位、INDEX/README 增补完成

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-013：上游零改动（NFR）

**关联需求：** srs §4 上游完整性（triton/tvm 零改动，mtime 聚类）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. 全库 mtime 扫描：最新 mtime 与克隆时刻聚类（triton ≤14:54:17 / tvm ≤14:54:22）

**期望结果：**
- Then 零偏离集群的文件（无任何后于克隆分钟的修改）

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-014：环境稳定性（NFR）

**关联需求：** srs §4 环境稳定（不 pip install）
**测试类型：** 回归测试
**优先级：** Medium

**测试步骤：**
1. 核对本 AR 全部执行记录（bench/plot/spot-check）无任何 pip install 调用
2. 全局 site-packages 的 triton/torch 版本与 env.json 一致

**期望结果：**
- Then 全局环境未被改动

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

### ST-015：可重复性与时间预算（NFR）

**关联需求：** srs §4 可重复性（JSON+脚本一键复现）+ 时间预算（E1-E4 ≤60 分钟）
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. README 复现命令与实际脚本入口一致（--exp E1..E4 + plot）
2. 核对今日全量执行记录的 wall time（E3 33.6s、E4 25.5s、E1/E2 约各 1 分钟）

**期望结果：**
- Then 一键可复现；总时长远低于 60 分钟

**实际结果：** [见下方各用例执行记录]

**状态：** PASS

---

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 15 | 15 | 0 | 0 |



## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-06 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮（ST-009 首轮 FAIL 系验证脚本自身计数 bug——PTX 头部注释行被计入字符串计数；修正为只数非注释行后复跑，15/15 PASS。交付物本身无缺陷） |

> 执行器：`st-verify.py`（本目录，归档时不随行——属过程文件）。每用例的实测
> 证据即执行器输出，关键数字摘录：

- ST-001：torch add@64M 379.7 vs 377.63 GB/s（+0.6%）；triton 377.6 vs 372.55（+1.4%）——均 <5%
- ST-002：e1na/e3na 含 numba 0.68 版本 + same_session=true，8 条 e3na 记录
- ST-003：四类门新鲜执行全过（fp16 实测 err 0.0625 ≤ 1.5 ulp 门 0.0938）
- ST-005：4 变体 × 5 行宽；2.45× + L2 折扣口径；原生差距 1.2-1.6× 有机制分析
- ST-006：76 valid + 32 pruned = 108 精确闭环，pruned 全为 shared 公式；热图/迁移图/对账表/87% 齐全
- ST-008：78 config 记录全 mma=0/ldmatrix=0；fma=BM×BN×BK/threads 逐条对账；9.50×
- ST-009：best_fp16_kernel.ptx 头部溯源注释 + 正文 mma=0/fma=512/cvt=16
- ST-013：上游 mtime 聚类（triton 14:54:17 / tvm 14:54:22 < 14:55:00 克隆截止）零改动
- ST-015：E3 33.6s / E4 25.5s / E1/E2 各约 1 分钟——总时长远低于 60 分钟预算

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 F1 | 三方基准框架（重复性/子进程合并/正确性门） | ST-001, ST-002, ST-003 | PASS |
| §3.2 E1 | vector-add 三方对照 | ST-004 | PASS |
| §3.2 E2 | fused-softmax 三方 + 流量模型 | ST-005 | PASS |
| §3.2 E3 | matmul FP32 config 全扫描 | ST-006, ST-007 | PASS |
| §3.2 E4 | matmul FP16 + lowering 证据 | ST-008, ST-009 | PASS |
| §3.3 F3 | ≥8 张可解释图 | ST-010 | PASS |
| §3.4 F4 | sm_75 lowering 拆解文档 | ST-011 | PASS |
| §3.5 F5 | 六段笔记 + INDEX + README | ST-012 | PASS |
| §4 NFR | 上游完整性/环境稳定/可重复性/时间预算 | ST-013, ST-014, ST-015 | PASS |

**需求覆盖率：** 9 / 9（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 9 | 9 | 0 | 0 |
| 边界条件 | 3 | 3 | 0 | 0 |
| 回归测试 | 3 | 3 | 0 | 0 |
| **合计** | **15** | **15** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | fig6 剪枝格以空白表示（非打叉）；fig7 对账表在 results.md 而非图面 | 已在 README 已知限制记录口径，不阻塞 |
| Minor | fig1 L2 拐点位置按工作集 3N×4B=4MB 口径（N≈350K），与 design 起草时 N≈2^20 的表述有口径差 | 已在 README 记录口径（工作集口径为正确物理口径） |

### 结论

> **Go**。15/15 用例 PASS，需求覆盖 9/9（100%），无 Critical/Major 缺陷，
> 两个 Minor 为图面口径说明类（已记录在案）。核心数据链经 review 30 项抽查
> + ST 78 config 逐条对账双重验证。AR004 可归档。
