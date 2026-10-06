# [AR001] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR001 |
| AR 主题 | gpu-puzzles-solutions |
| 关联 SR | 无（学习任务；对应本仓库 README「建议顺序」第 1 条） |
| 日期 | 2026-10-05 |
| 状态 | Draft |

## 1. 背景与目标

`01-foundations/` 是本 AI Infra 资料库的入门层，包含 Sasha Rush 的 GPU-Puzzles（14 道 Numba CUDA 交互练习，每题留有 `# FILL ME IN` 待实现）与 GPU MODE 讲义。本 AR 的目标：

1. 完成 GPU-Puzzles 全部 14 道题的 kernel 实现；
2. 在本机（Quadro RTX 5000，sm_75）上**真实运行 GPU kernel** 与 numpy spec 对拍验证，深度利用本机 CPU + GPU；
3. 对 lecture_001~004、012、014 六讲产出详细逐节学习笔记，建立与 puzzles 的知识关联。

**环境事实澄清**：仓库根 README 声称「本机无 nvidia-smi / 无法跑 kernel」已过时。实测：
- `C:\Windows\System32\nvidia-smi.exe` 存在：Quadro RTX 5000，驱动 556.18，CUDA Version 12.5；
- 全局 Python 3.11.9 已装 PyTorch 2.5.1+cu121，`torch.cuda.is_available() == True`；
- CUDA Toolkit 目录 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5` 仅剩 `extras`，**无 nvcc/NVVM**，需通过 pip 镜像在独立 venv 中补齐 numba 运行所需组件。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- F1：14 道 GPU Puzzles 的完整解答（numba CUDA 语法，与上游题目签名一致），含各题全部子测试用例（1D Conv 2 组、Prefix Sum 2 组、Matmul 2 组）
- F2：验证脚本：优先用 numba.cuda 在本机 RTX 5000 上真实编译运行 14 题并与 numpy spec 对拍；同时提供纯 Python CPU 模拟对拍作为回退与交叉验证
- F3：6 讲详细逐节笔记（lecture_001~004 + 012 + 014）

**Out of Scope（本 AR 不做的）：**

- 不修改 `GPU-Puzzles/`、`gpu-mode-lectures/` 任何上游文件
- 不覆盖 lecture_005/008/009/013/017+ 等其余讲义
- 不做 02 目录及以后的内容，不为 sm_75 编写性能优化 kernel
- 不改动全局 Python 环境（numpy 2.4.6、torch 等保持原样）

## 3. 功能需求

### 3.1 F1 — 14 道 GPU Puzzles 解答

**描述：** 新建独立 solutions 文件给出全部 14 题 kernel 实现，不改动上游。

**触发条件：** 运行 `verify.py`（或直接阅读 solutions.py）。

**期望行为：** 每题提供与上游题目同签名的 `*_test(cuda)` 闭包，`# FILL ME IN` 处逻辑完整：
1. Map：`out[i] = a[i] + 10`
2. Zip：`out[i] = a[i] + b[i]`
3. Guards：越界线程直接返回，不写 `out`
4. Map 2D：`out[i,j] = a[i,j] + 10`，二维线程索引 + guard
5. Broadcast：`out[i,j] = a[i,0] + b[0,j]`
6. Blocks：全局索引 `blockIdx.x * blockDim.x + threadIdx.x` + guard
7. Blocks 2D：二维全局索引 + guard
8. Shared：经 shared memory 中转后写 out，遵循 syncthreads
9. Pooling：shared memory 存块内数据，每线程算窗口 3 的局部和，每线程 1 次全局读 + 1 次全局写
10. Dot Product：shared memory 归约，每线程 2 次全局读 + 1 次全局写
11. 1D Convolution：一般情形（含跨块 halo），每线程 2 次全局读 + 1 次全局写
12. Prefix Sum：shared memory 上的并行前缀和（每步合并剩余一半），块内求和写出
13. Axis Sum：二维 batch × 列，blockIdx.y 对应 batch，列方向块内归约
14. Matmul：分块拷入 shared memory，迭代推进部分点积（困难情形 6 次全局读以内）

**异常处理：** kernel 内不使用 numpy 属性、列表推导等 CUDA 不支持语法；shared memory 尺寸用字面常量；写后读 shared 必须 `cuda.syncthreads()`。

**验收标准：**
- Given 14 题的上游测试数据与 spec 函数, When 在本机以 numba.cuda 真实运行 solutions 中的 kernel, Then `np.testing.assert_allclose(out, spec)` 全部通过
- Given CPU 模拟执行器按 block/thread 枚举在 numpy 数组上执行同一 kernel 逻辑, Then 与 spec 对拍同样全部通过
- Given `git status`（如启用版本控制）或文件比对, Then `GPU-Puzzles/`、`gpu-mode-lectures/` 上游文件零改动

### 3.2 F2 — 验证脚本（GPU 真机 + CPU 模拟双通道）

**描述：** `verify.py` 一键验证 14 题解答，GPU 优先、CPU 模拟回退。

**触发条件：** 在 solutions venv 中执行 `python verify.py`。

**期望行为：**
1. 复刻 `lib.py::CudaProblem.run_cuda` 的调用方式（`numba.cuda.jit(fn)[grid, block](out, *inputs, *args)`），不依赖 chalk/colour/IPython 绘图链
2. 每题运行全部上游子测试（Conv ×2、Sum ×2、Matmul ×2，其余 ×1，共 17 个用例；初稿误计为 18，10 个单例 + 2 + 2 + 1 + 2 = 17，已按上游实际测试数修正），逐题打印 PASS/FAIL 与期望/实际对照
3. GPU 通道不可用（numba/NVVM 装配失败）时自动回退纯 Python CPU 模拟通道：按 `Coord.enumerate()` 语义枚举 block/thread，用等价 Python 对象（threadIdx/blockIdx/blockDim/shared 数组 + 同步栅栏语义）在真实 numpy 数组上执行 kernel
4. 退出码：全部通过 = 0，任一失败 = 1，并输出失败题目清单

**异常处理：** venv 未创建 / numba 不可导入 / 无 CUDA 设备时给出明确诊断信息并切换 CPU 通道，不静默失败。

**验收标准：**
- Given solutions venv 已就绪且 GPU 可用, When `python verify.py --gpu`, Then 14 题 17 用例全部 `Passed`（真实 GPU 执行）
- Given 无 numba 环境, When `python verify.py --cpu`, Then 同样全部 `Passed`（CPU 模拟）
- Given 任一 kernel 写错（如故意去掉 guard）, When 运行 verify, Then 对应用例 FAIL 且退出码非 0

### 3.3 F3 — 六讲详细逐节笔记

**描述：** 对 gpu-mode-lectures 的 lecture_001~004、012、014 产出详细逐节 Markdown 笔记。

**触发条件：** 阅读 `01-foundations/notes/` 下笔记文件。

**期望行为：**
- 每讲一篇 `lecture_XXX-notes.md`，按讲义实际小节逐节展开：核心概念、代码/示例讲解、关键结论
- 标注与本机硬件（sm_75、64KB shared memory、无 TMA/WGMMA）的适配点与讲义中超出本机能力的内容（如 Hopper 专属特性）
- 与 GPU-Puzzles 对应题号建立显式关联（如 lecture_001 的线程索引 ↔ Puzzle 1/3/6）
- 012（FlashAttention 算法）、014（Triton）笔记需覆盖讲义的主算法/主流程，并注明本机可运行路径（Triton 支持 sm_75）

**异常处理：** 讲义文件为空或缺失时，在笔记中显式记录缺失情况，不虚构内容。

**验收标准：**
- Given 6 篇笔记, When 逐篇对照讲义目录, Then 每讲全部小节/主要文件均有对应笔记内容，无遗漏
- Given 笔记中出现的代码或结论, When 抽查对照讲义原文, Then 无事实性错误
- Given 笔记中的 puzzles 关联标注, Then 与 F1 实际题号一致

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 环境隔离 | 全局 Python | 不修改全局 site-packages；numba 及依赖仅装入 solutions/.venv |
| 兼容性 | sm_75 | GPU 通道必须在 Turing (sm_75) 上实际跑通，不使用 sm_80+ 专属特性 |
| 可重复性 | 验证脚本 | `verify.py` 可重复执行，结果确定（不依赖随机数） |
| 上游完整性 | 文件哈希 | 上游两目录所有文件零改动 |

## 5. 约束与假设

**约束：**
- pip 仅能走华为内网镜像（mirrors.tools.huawei.com / cmc-cd-mirror.rnd.huawei.com）；numba 需 numpy < 2.2，与全局 numpy 2.4.6 不兼容 → 必须独立 venv
- 本机无 nvcc/NVVM 系统安装 → GPU 通道依赖 pip 包 `nvidia-cuda-nvcc-cu12`（含 NVVM）或等效方式供 numba 使用；若装配失败，CPU 模拟通道为兜底
- numba CUDA kernel 语法限制：无 numpy 属性访问、无列表推导、shared memory 尺寸为字面常量、跨线程数据依赖需 syncthreads

**假设：**
- 华为镜像可提供 numba、llvmlite、nvidia-cuda-nvcc-cu12 等 wheel
- GPU-Puzzles 上游 `lib.py` 的 `run_cuda` 调用语义（jit、grid/block 配置、参数顺序）可直接复刻，不需要 chalk 绘图链
- lecture 讲义文件（README/ipynb/cu）内容完整可读

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| TPB | Threads Per Block，每 block 线程数（上游题目常量） |
| Guard | 越界保护：线程数多于数据量时，多余线程不执行写入 |
| Shared memory | block 内共享显存，写入后需 `cuda.syncthreads()` 才能被同 block 其他线程读取 |
| Prefix Sum | 并行前缀和：每步将剩余元素两两合并，log₂(TPB) 步完成块内归约 |
| Halo | 卷积分块时，本 block 计算所需但落在相邻 block 范围内的元素 |
| sm_75 | Turing 架构 compute capability，Quadro RTX 5000 |
| NVVM | numba CUDA 编译所需的 NVIDIA IR 编译库（libnvvm） |
