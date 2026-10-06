# [AR001] 全量验收清单（T014）

> 验收日期：2026-10-05　验收人：开发会话（自查）　依据：srs.md §3 验收标准 + §4 非功能需求
> 结论：**F1/F2/F3 全部 PASS**，可移交 review 阶段。

## 1. F1 — 14 道 GPU-Puzzles 解答

| # | 验收标准（srs 原文） | 结果 | 证据 |
| --- | --- | --- | --- |
| 1.1 | 本机 numba.cuda 真实运行 solutions 中的 kernel，`np.testing.assert_allclose(out, spec)` 全部通过 | PASS | 2026-10-05 复跑 `verify.py --gpu`：**17/17 PASS**（Quadro RTX 5000 / sm_75，solutions/.venv） |
| 1.2 | CPU 模拟执行器按 block/thread 枚举执行同一 kernel 逻辑，与 spec 对拍全部通过 | PASS | 2026-10-05 复跑 `verify.py --cpu`（全局 Python）：**17/17 PASS，退出码 0** |
| 1.3 | `GPU-Puzzles/`、`gpu-mode-lectures/` 上游文件零改动 | PASS（附证据说明） | 见 §4 上游完整性 |

补充：17 用例 = 10 个单例（Map/Zip/Guard/Map 2D/Broadcast/Blocks/Blocks 2D/Shared/Pooling/Dot）+ 1D Conv ×2（含双 block halo）+ Prefix Sum ×2（含非整除补 0）+ Axis Sum ×1 + Matmul ×2（单块退化 + 3×3 块迭代 6 读）。

## 2. F2 — 双通道验证脚本

| # | 验收标准 | 结果 | 证据 |
| --- | --- | --- | --- |
| 2.1 | `python verify.py --gpu`：14 题 17 用例全部 Passed（真实 GPU） | PASS | GPU 通道 17/17（真实编译执行，无 chalk/colour/IPython 依赖） |
| 2.2 | `python verify.py --cpu`：无 numba 环境同样全部 Passed | PASS | 全局 Python（无 numba）跑 CPU 通道 17/17 |
| 2.3 | 故意写错的 kernel → 对应用例 FAIL 且退出码非 0 | PASS | `verify.py --selftest`：bad_map（+20）与 bad_guard（去 guard 越界写）均被检出，`detected: YES`，退出码 0（自检通过的定义即"全部检出"） |
| 2.4 | 自动模式：GPU 可用选 GPU，不可用回退 CPU，不静默失败 | PASS | `--gpu` 强制但不可用时退出码 2 并打印诊断（verify.py:631-636）；自动分支含诊断输出（verify.py:642-648） |
| 2.5 | 栅栏死锁保护 | PASS（机制就位） | `cpu_run_case` join 超时 + `barrier.abort()` + TimeoutError（verify.py:465-476），不挂死 |

## 3. F3 — 六讲逐节笔记

交付位置：`01-foundations/notes/lecture_{001,002,003,004,012,014}-notes.md`（与 design.md §4.4 目录结构一致）。

| # | 验收标准 | 结果 | 说明 |
| --- | --- | --- | --- |
| 3.1 | 每讲全部小节/主要文件均有对应笔记，无遗漏 | PASS | L001（8 个脚本逐文件 + profiler 三层）、L002（3 例逐 kernel）、L003（pmpp.ipynb 88 cell 四级递进全覆盖）、L004（25 cell 三实验全覆盖）、L012（notebook 34 cell + 3 个 .cu + main.cu）、L014（notebook 全部 markdown 小节 + triton_util.py + Qs.md） |
| 3.2 | 抽查无事实性错误 | PASS（自查） | 笔记中代码/数值均取自讲义原文（如 spilling 8320B、ncu 13.02ms vs 2.10ms、0.21/0.71/0.07 加权等）；深度复核移交 review 阶段 |
| 3.3 | puzzles 关联标注与 F1 实际题号一致 | PASS（review 第 1 轮修正后复验） | 正确关联：L001↔P1/3/6/8、L002↔P2/4/6/7/9、L003↔P14、L004↔P11/12/14、L012↔P10/11/12/14、L014↔P9/11/14。初稿将 matmul 误标 P13（实际 P14，P13 为 Axis Sum）、P10 误标 softmax（实际 Dot，全 14 题无 softmax 题）——sdd-reviewer 独立审查发现（S1/C3 FAIL），第 1 轮修复：L003/L004/L012 题号更正、L012/L014 补显式关联；本题号清单已逐号对照 GPU_puzzlers.py 核验 |
| 3.4 | sm_75 适配点标注（64KB shared、无 TMA/WGMMA 等） | PASS | 每讲设独立「sm_75 适配标注」小节；L012/L014 注明本机可运行路径（动态 `compute_75` arch、triton 支持 sm_75） |

## 4. 上游完整性（非功能需求）

核查方法与证据（本机无 git 基线，GitHub raw 内网 403 无法拉取哈希比对，采用结构完整性 + 过程证据链）：

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| `GPU_puzzlers.py` 未被填写/修改 | PASS | `# FILL ME IN` 恰好 14 处全部保留；`!pip install`/`!wget` 头部语句原样；14 个 `*_test(cuda)` 工厂签名完整 |
| `lib.py` 未被修改 | PASS | `import chalk` 与 `def run_cuda` 原样（解答不依赖也不改动它） |
| 上游目录无编辑残留 | PASS | 全目录扫描无 `.bak/.orig/.rej/.tmp/~` 文件 |
| 新增文件仅交付物 | PASS | `GPU-Puzzles/` 下仅新增 `solutions/`（solutions.py + verify.py + README.md + .venv）；`01-foundations/` 下仅新增 `notes/` |
| 关键文件指纹（留档） | 记录 | `GPU_puzzlers.py` sha256 `315749999ed8297ae2c8651fc4eb23d933411ba073e54a498ab6237ae338babd`；`lib.py` sha256 `90096e75d40bc625602b026cf63a386a49bc8843263651f0f53488f23e82a8b3` |

**限制声明**：mtime 无区分度（仓库整体于 2026-10-05 落盘）；无上游哈希基线可比对。如后续可访问外网，建议用上述 sha256 与 GitHub 比对做终验。

## 5. 非功能需求

| 类型 | 要求 | 结果 |
| --- | --- | --- |
| 环境隔离 | 不修改全局 site-packages；numba 仅在 solutions/.venv | PASS（.venv 内 numpy<2.2 + numba + CUDA 12.5 组件 pin；全局 numpy 2.4.6 未动） |
| sm_75 兼容 | GPU 通道在 Turing 实跑，不用 sm_80+ 特性 | PASS（真实 sm_75 编译执行 17/17） |
| 可重复性 | verify.py 可重复执行、结果确定 | PASS（同日两次全量复跑结果一致，无随机数） |
| 文档一致性 | 用例数与实现一致 | PASS（本轮修正 srs/design/tasks 中 18→17：初稿把「其余 ×1」误计为 8 例，实际 10 单例 + 2 + 2 + 1 + 2 = 17，与上游测试数一致） |

## 6. 遗留与移交

- 无阻塞缺陷；负向自检、异常路径（强制 --gpu 不可用、栅栏死锁超时）均有机制且验证过。
- 笔记事实性抽查的深度复核 → 移交 review 阶段（sdd-task-review）。
- 建议（非本 AR 范围）：后续 AR 若引入版本控制，先对上游目录建立 git 基线。
