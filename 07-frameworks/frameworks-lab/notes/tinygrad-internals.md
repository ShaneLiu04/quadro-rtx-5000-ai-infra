# tinygrad 图编译器拆解（AR007 F4 笔记）

> 分级：**[源码]** = 读克隆源码（文件:行号）；**[实测]** = 本机实验
> （Quadro RTX 5000 sm_75 / torch 2.5.1 / tinygrad main 克隆，零改动）。
> 本机 realize 失败 = 诚实记录的教学样本（不修不绕，AR005 E7 惯例）。

## 0. 一句话总结

tinygrad 是「**一张 UOp 图走到底**」的编译型框架：Tensor 只是指向 UOp 图的
句柄，所有操作（含随机数、buffer 分配、schedule 决策）都是图节点；执行 =
把图经 pattern-matching 重写成后端可编译的线性程序，再由 renderer 生成
PTX/LLIR/METAL 源码。面试一句话：**lazy + 单一 IR + 图重写 + 多后端渲染**，
对比 PyTorch eager（op 级立即执行）和 Inductor（FX 图 + Triton 生成）。

## 1. 图编译管线（七步）

| 步骤 | 位置 [源码] | 说明 |
|---|---|---|
| 1. Tensor → UOp | `tensor.py:104` `_apply_uop`：每个算子返回新 Tensor 包新 UOp，**不物化不执行** | `t.uop` 即图根（本版 API 由 lazydata 改名 uop [实测]） |
| 2. 图构造 | `function.py`（如 matmul→mul+sum 的代数分解） | [实测] `(a@b).relu()+1` 确定性图 **56 节点**：CONST 15 / RESHAPE 9 / STACK 8 / PERMUTE 4 / CAST 4 / MUL 3...；`randn` 图 **236 节点**（THREEFRY PRNG 节点可见——**随机数也是图**）（results/e0_env.json） |
| 3. 调度 | `schedule/__init__.py:31` `create_schedule` → `schedule/` 7 个模块 | 把"算子森林"变成带 buffer 语义的 schedule 图 |
| 4. 图重写 | `uop/ops.py` `graph_rewrite` + `uop/` 模式匹配器；新 HCQ 路径 `runtime/support/hcq2.py:303` `sched_batches`、`:524` `hcq_compile` | [实测] 本机 CUDA 失败点 = `hcq2.py:533` `graph_rewrite(sched_batches(linear), pm_encode, name="encode")` 抛 **StopIteration**（pattern matcher 耗尽 = 编译器内部 bug/兼容性断裂，全 traceback 落 e0_env.json） |
| 5. codegen | `codegen/`（3 文件） | 线性程序 → 后端 IR |
| 6. 渲染 | `renderer/`（7 文件：cuda/llvm/clang/webgpu/metal...） | 生成 PTX/LLVM IR/... |
| 7. 执行 | `runtime/ops_cuda.py` 等（HCQ = host command queue 抽象） | [实测] **`runtime/ops_torch.py` 文件不存在**——TORCH 后端整体缺失，`Tensor.randn(128,128).to('TORCH')` realize 抛 ModuleNotFoundError（比 CUDA 的 StopIteration 更根本） |

## 2. UOp：万物皆节点

- [实测] UOp 属性：`u.op`（Ops 枚举）、`u.src`（前驱元组）、`u.dtype`、`u.shape`；
  图遍历 = 递归 `src` 去重（E0 探针 5 行代码）。
- [源码] `uop/` 10 个模块 = **统一 IR + pattern matching 引擎**：ops.py 定义
  Ops 枚举与 graph_rewrite；每个优化（fusion、代数化简、to_register 移动）
  都是一条 rewrite 规则——与 AR004 Triton 的 `tt.compile` 阶段同思想。
- [面试] 对照表：
  - **PyTorch eager**：op 级立即执行，无图（AR006 的 profiler 归因就是 op 粒度）
  - **torch.compile/Inductor**：FX graph（symbolic trace）→ Triton/C++ 生成
  - **tinygrad**：全程序单图（连 buffer 分配 ALLOC、拷贝 COPY、同步 AFTER 都在图内，
    [实测] 直方图可见 STORE/ALLOC/BUFFER/AFTER 节点）
  - **bnb**：不是图框架——是 op 级 dispatch（见 bnb-internals.md §2）

## 3. 本机失败模式解剖（诚实记录）

- [实测] CUDA 栈：`realize()` → `engine/realize.py:280` `run_linear` →
  `:273` `compile_linear` → `hcq_compile`（hcq2.py:524）→ **`:533` encode 阶段
  graph_rewrite 抛 StopIteration**。StopIteration 从 pattern matcher 的
  `next()` 耗尽冒泡——内部编译器在 sm_75/WDDM/torch 2.5.1 组合上的兼容性断裂。
- [实测] TORCH 栈：`realize.py:234` `lower_and_compile` → `:236` dictcomp 中
  `_get_call_to_compile` → `runtime/ops_torch.py` **ModuleNotFoundError**（文件
  在本克隆中不存在——TORCH 后端被移除/重构未完成）。
- [实测] 能跑的部分：import、`Device.DEFAULT='CUDA'`（检测到 GPU）、图构造与
  内省全部正常——**失败精确发生在"图 → 机器码"边界**，这本身就是教学数据：
  编译型框架的失败模式和 eager 完全不同（eager 会在第一个 op 就报错，
  编译型在编译期深处炸）。
- [实测] **附带伤害（端到端复现发现）**：hcq2 崩溃后同进程内任何后续
  torch CUDA kernel 都报 "invalid argument"——tinygrad 绕过 torch 的
  CUDA 上下文管理（自己 cuCtx/驱动调用），崩溃时也不负责恢复现场，
  进程级 CUDA 上下文被留在不可用状态。工程结论：**跨框架混跑必须
  进程隔离**（framework_lab.py all 已改为子进程逐实验执行）。
- [红线] 不修不绕不臆测未验证行为；与 AR005 E7（Windows 无编译器、只拆源码）、
  AR006 E7（inductor 现状）同一诚实记录惯例。

## 4. 面试高频问答卡

| 问题 | 回答要点 |
|---|---|
| tinygrad 和 PyTorch 根本区别？ | lazy 单图 IR vs eager op 流；[实测] 56 节点图不 realize 零执行成本；执行才编译（JIT） |
| 为什么随机数也在图里？ | randn = THREEFRY 可逆哈希 PRNG 的纯函数展开（[实测] 直方图 THREEFRY/SHL/SHR/OR 节点）——保证可复现 + 可调度进 kernel |
| 图重写引擎怎么工作？ | pattern matching（uop/ops.py graph_rewrite）：规则集 pm_encode 等逐层 apply 直到不动点；本机失败 = encode 阶段 matcher 耗尽抛 StopIteration（[实测] traceback） |
| 和 Inductor 比？ | Inductor：trace Python → FX 图 → Triton 源码生成 → 运行时编译；tinygrad：全程序 UOp 图 → 重写 → renderer → PTX/LLIR；共同点 = "把 eager 的隐式优化显式化成图变换" |
| 编译型框架的调试？ | 失败点远离用户代码（本例：tensor.py 之下 6 层，hcq2.py:533）；诚实记录 traceback + 定位管线阶段比"绕过"有价值 |
