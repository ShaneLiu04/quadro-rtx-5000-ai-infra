# [AR001] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR001 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-05 |

> ST 为黑盒验证，基于 srs.md §3 验收标准；执行环境：Quadro RTX 5000（sm_75，驱动 556.18/CUDA 12.5）+ 全局 Python 3.11.9（无 numba）+ solutions/.venv（numba 0.68.0 + CUDA 12.5 组件）。

## 测试用例列表

### ST-001：GPU 真机全量对拍（F1 核心验收）

**关联需求：** srs.md §3.1 F1 验收标准 1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given solutions venv 已就绪且 GPU（sm_75）可用

**测试步骤：**
1. 在 solutions 目录以 .venv Python 执行 `python verify.py --gpu`

**期望结果：**
- Then 14 题 17 用例全部 `Passed`（真实 GPU 编译执行），退出码 0

**实际结果：** `[GPU] summary: 17/17 passed`，EXIT=0。.venv 环境实测：numba 0.68.0，`cuda.is_available() == True`。

**状态：** PASS

---

### ST-002：CPU 模拟通道全量对拍（F1 验收标准 2）

**关联需求：** srs.md §3.1 F1 验收标准 2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 全局 Python（无 numba 环境）

**测试步骤：**
1. 以全局 Python 执行 `python verify.py --cpu`

**期望结果：**
- Then 17 用例全部 `Passed`（CPU 栅栏模拟），退出码 0

**实际结果：** `[CPU] summary: 17/17 passed`，EXIT=0（全局 Python，无 numba）。

**状态：** PASS

---

### ST-003：上游零改动（F1 验收标准 3）

**关联需求：** srs.md §3.1 F1 验收标准 3 + §4 非功能需求「上游完整性」
**测试类型：** 回归/完整性
**优先级：** High

**前置条件：**
- Given 上游两目录应为原始内容

**测试步骤：**
1. 校验 `GPU_puzzlers.py` 含恰好 14 处 `# FILL ME IN`、`!pip`/`!wget` 头部语句、14 个 `*_test(cuda)` 工厂
2. 校验 `lib.py` 含 `import chalk` 与 `def run_cuda`
3. 校验 sha256 与 acceptance.md 留档一致
4. 扫描上游目录无 `.bak/.orig/.rej/.tmp` 编辑残留

**期望结果：**
- Then 全部校验通过，上游零改动成立

**实际结果：** FILL=14 ✓、pip/wget ✓、14 工厂 ✓、chalk/run_cuda ✓；sha256：`GPU_puzzlers.py=31574999…babd`、`lib.py=90096e75…a8b3` 与留档一致；编辑残留扫描为空。

**状态：** PASS

---

### ST-004：负向自检——坏 kernel 必被检出（F2 验收标准 3）

**关联需求：** srs.md §3.2 F2 验收标准 3；design §6.4「负向用例」
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given 故意写错的 kernel（错误数值 / 去掉 guard）

**测试步骤：**
1. 执行 `python verify.py --selftest`

**期望结果：**
- Then bad_map 与 bad_guard 均 `detected: YES`，自检退出码 0

**实际结果：** `Map detected: YES`、`Guard detected: YES`、`result: PASS - harness catches broken kernels`，EXIT=0。

**状态：** PASS

---

### ST-005：端到端负向注入——错误 kernel 全量运行必 FAIL 且退出码非 0

**关联需求：** srs.md §3.2 F2 验收标准 3
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given 通过 monkeypatch 注入错误数值（+20）的 Map kernel

**测试步骤：**
1. 注入后调用 `verify.main(['--cpu'])`，检查输出与退出码

**期望结果：**
- Then Map 用例 FAIL，退出码 1，打印失败题目清单

**实际结果：** `[CPU] Map FAIL`、`failed cases: Map`、退出码 1。
*执行备注：首轮以「无 guard」kernel 注入判 FAIL 系测试设计错误——Map 用例 4 线程=4 数据，无 guard 行为与正确实现等价，不构成错误；改用 +20 错误数值后按预期检出。产品无缺陷。*

**状态：** PASS

---

### ST-006：强制 --gpu 但环境不可用——明确报错不静默降级（F2 异常处理）

**关联需求：** srs.md §3.2 异常处理 + design §6.2/6.4「环境异常」
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- Given 全局 Python（无 numba / 无 CUDA）

**测试步骤：**
1. 以全局 Python 执行 `python verify.py --gpu`

**期望结果：**
- Then 打印明确诊断，退出码 2，不静默降级

**实际结果：** `GPU channel unavailable: numba 未安装（No module named 'numba'）；请在 solutions/.venv 中安装` + Hint，EXIT=2。

**状态：** PASS

---

### ST-007：栅栏死锁超时保护（design §6.4「栅栏纪律」）

**关联需求：** design §6.4；srs §3.2 异常处理
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- Given `cuda.syncthreads()` 位于非 uniform 分支（仅 local_i==0 调用）的 kernel

**测试步骤：**
1. 构造死锁 kernel，经 `cpu_run_case(case, timeout=5.0)` 执行

**期望结果：**
- Then 在超时上限内抛 TimeoutError，不挂死

**实际结果：** 5s 内抛出 `TimeoutError: CPU emulator barrier deadlock: syncthreads must be called uniformly by all threads in the block`。

**状态：** PASS

---

### ST-008：自动模式通道选择（F2 期望行为 3）

**关联需求：** srs.md §3.2 期望行为 3
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given 全局 Python（GPU 通道不可用）

**测试步骤：**
1. 以全局 Python 执行 `python verify.py`（无参数）

**期望结果：**
- Then 打印回退信息，CPU 通道 17/17，退出码 0

**实际结果：** `GPU channel unavailable (numba 未安装…); falling back to CPU channel` → `[CPU] summary: 17/17 passed`，EXIT=0。
*执行备注：首轮 EXIT=-1 为测试命令管道截断（`Select-Object -First 1` 提前终止 python 进程）的测量伪影；去除截断后复测为 0。*

**状态：** PASS

---

### ST-009：可重复性（NFR）

**关联需求：** srs.md §4「verify.py 可重复执行，结果确定」
**测试类型：** 回归测试
**优先级：** Medium

**前置条件：**
- Given 同一环境连续两次全量执行

**测试步骤：**
1. GPU 通道连续两轮；CPU 通道连续两轮

**期望结果：**
- Then 结果完全一致（17/17），无随机波动

**实际结果：** GPU 两轮均 17/17（EXIT=0）；CPU 两轮均 17/17（EXIT=0）。

**状态：** PASS

---

### ST-010：环境隔离（NFR）

**关联需求：** srs.md §4「不修改全局 site-packages」
**测试类型：** 回归测试
**优先级：** Medium

**前置条件：**
- Given 交付完成后检查全局环境

**测试步骤：**
1. 全局 Python `import numba`（应失败）；.venv Python `import numba` + `cuda.is_available()`

**期望结果：**
- Then 全局无 numba；.venv 内 numba + CUDA 可用

**实际结果：** 全局 `ModuleNotFoundError: No module named 'numba'`（导入失败 ✓）；.venv：numba 0.68.0、`cuda_available True`。

**状态：** PASS

---

### ST-011：六讲笔记齐全与结构验收（F3 验收标准 1）

**关联需求：** srs.md §3.3 F3 验收标准 1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given `01-foundations/notes/` 下应有 6 篇

**测试步骤：**
1. 枚举 notes 目录校验 6 文件；2. 校验每篇含 sm_75 适配标注；3. 校验每篇含 puzzles 关联

**期望结果：**
- Then 6 篇齐全，结构要素齐备

**实际结果：** 6 文件齐全（001/002/003/004/012/014，顺序与命名符合）；每篇均含 sm_75 标注与 Puzzle 引用（脚本全量校验）。

**状态：** PASS

---

### ST-012：题号关联正确性（F3 验收标准 3）

**关联需求：** srs.md §3.3 F3 验收标准 3
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 题号映射：P10=Dot、P11=1D Conv、P12=Prefix Sum、P13=Axis Sum、P14=Matmul

**测试步骤：**
1. 提取 6 篇笔记全部带名称的题号引用，逐一比对题号↔名称配对

**期望结果：**
- Then 无任何错配

**实际结果：** 全量正则扫描（`Puzzle N（名称）` 形态），mismatches=NONE。

**状态：** PASS

---

### ST-013：笔记事实性抽查（F3 验收标准 2）

**关联需求：** srs.md §3.3 F3 验收标准 2
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given 笔记关键数字/常量应可在讲义原件找到

**测试步骤：**
1. L012 笔记的 flash_attention.cu 常量（B_r=8/B_c=32/blockDim 128×8）↔ 原件
2. L012 笔记的 spilling 数字（8320B、13.02ms vs 2.10ms）↔ flash_attention.ipynb
3. L004 笔记的 occupancy/字面量事实（64k/2k→32、1536/256→6、0.2989f）↔ cuda-mode-session-4.ipynb
4. L003 笔记的灰度加权（0.2989/0.5870/0.1140）↔ pmpp.ipynb
5. L014 笔记的 triton_util.py 函数名（7 个工具）↔ 原件

**期望结果：**
- Then 全部命中，无事实性错误

**实际结果：** 五组抽查全部命中（True×5）。
*执行备注：初版用例把 spilling 数字出处误标为 L004（实为 lecture_012 notebook 的事实），已更正出处后复测。*

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-05 |
| 执行结果 | **PASS** |
| 执行轮次 | 第 1 轮（含 2 个测试脚本缺陷的当场修正与复测，无产品缺陷） |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 F1 | 14 道 GPU-Puzzles 解答 | ST-001, ST-002, ST-003 | PASS |
| §3.2 F2 | 双通道验证脚本 | ST-004, ST-005, ST-006, ST-007, ST-008 | PASS |
| §3.3 F3 | 六讲逐节笔记 | ST-011, ST-012, ST-013 | PASS |
| §4 NFR | 可重复性 / 环境隔离 / sm_75 / 上游完整性 | ST-009, ST-010, ST-001, ST-003 | PASS |

**需求覆盖率：** 4 / 4（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 5 | 5 | 0 | 0 |
| 边界条件 | 2 | 2 | 0 | 0 |
| 异常处理 | 4 | 4 | 0 | 0 |
| 回归/完整性 | 3 | 3 | 0 | 0 |
| **合计** | **13** | **13** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| INFO | acceptance.md §3.3 分讲关联清单为摘要性质，未穷尽笔记全部题号引用（review 第 2 轮提出，所列条目本身全部正确） | 已记录，不阻塞 |
| Minor | solutions.py 中 TPB8/TPB3 常量位于题旁而非文件顶部；CPU 通道 block 枚举顺序（x 外层）与上游 Coord.enumerate()（y 外层）相反——均不影响正确性（review 第 2 轮判定不阻塞） | 记录延后处理，不阻塞归档 |

### 结论

> **Go**。Go 条件全部满足：需求覆盖率 100%（4/4）；无 Critical/Major 缺陷（0 个产品缺陷，2 个 ST 测试脚本缺陷当场修正）；GPU 真机（sm_75）17/17 + CPU 17/17 + 负向自检 + 全部异常路径通过，无回归；NFR（可重复性、环境隔离、上游完整性、sm_75 实跑）全部验证达标。遗留 2 条 INFO/Minor 均已记录且经 review 确认不阻塞。
> （Go 判定依据用户会话级预授权「自动化、每一步无需确认」执行。）
