# Lecture 014 笔记 — A Practitioner's Guide to Triton

> 讲义目录：`lecture_014/`（notebook + `triton_util.py` + `Qs.md` FAQ 链接）
> 作者 Umer H. Adil。主线：为什么/何时用 Triton → 编程模型（块级）→
> 从 vector add 一路写到 grouped/swizzled matmul → benchmark → autotune。

## 0. `triton_util.py` —— 讲义自带的调试工具箱（先读这个）

| 工具 | 作用 |
| --- | --- |
| `check_tensors_gpu_ready` | 断言 contiguous；非模拟模式下断言 is_cuda |
| `test_pid_conds` / `breakpoint_if` / `print_if` | 按 pid 条件下断点/打印，如 `'>1,=0'` 表示 `pid_0>1 and pid_1==0` |
| `cdiv` | `(a+b-1)//b`，Triton 也有内置 `triton.cdiv` |
| `get_1d_offest` / `get_2d_offset` | 块内偏移 + 块基址（`n_prev_chunks*size + arange`；2D 用 `expand_dims` 外积出 stride 网格） |
| `get_1d_mask` / `get_2d_mask` | `offs < max` 的边界掩码 |

四个 jit 小函数是所有 kernel 的积木——**offset 和 mask 的构造被抽象出来复用**，
这是本讲代码组织的核心风格。

## 1. Why & when（决策树，值得背下来）

- **Triton 是什么**：Python 风格代码 → Triton 编译器做指令重排/向量化/共享内存管理 →
  PTX。与 CUDA 编译目标相同，但把「块内调度」交给编译器。
- **vs CUDA**：CUDA 全手动、上限最高、难写难调；Triton 放弃部分控制权换取
  「容易写出还不错性能」的代码。
- **vs torch.compile**：`torch.compile` 生成的新 kernel 本身就是 Triton kernel——
  它生成的代码可以作为手写 kernel 的起点。
- **使用顺序**：model 不够快 → ① `torch.compile` → ② 改写代码让 compile 更好使 →
  ③ 找热点写自定义 Triton kernel → ④ 仍不够再写 CUDA。（确知要极限性能可直接 CUDA。）
- 讲义还标注了 Triton 的 rough edges（作者以 `# Weirdness:` 注释记录），预期快速打磨中。

## 2. 编程模型：只有「块」一层

- CUDA：grid → block → thread，线程算**标量**，手工管理 shared memory。
- Triton：grid → **program（块）**，一个 program 对一整个块做**向量化**操作，
  load/计算/store/造 mask 全是块级向量；shared memory 由编译器代管。
- 术语：kernel 实例 = "program"，`program_id` ≡ CUDA 的 block id（pid）。
- 例子（size 8、块 4）：CUDA 8 线程各算 `z[i]`；Triton 2 个 program 各算 `z[0:3]=x[0:3]+y[0:3]`。

**与 GPU-Puzzles solutions 的映射**：numba 版一个线程一个标量（CUDA 思维），
Triton 版一个 program 一行/一块（块思维）——同一 puzzle 的两种分解，对照着看收益极大。
显式题号对应：块级 `tl.load`+mask ↔ **Puzzle 9（Pooling）** 的整块 shared 载入与
**Puzzle 11（1D Conv）** 的 halo 补载（`other=` 即越界补 0）；`tl.dot` 块级乘加 ↔
**Puzzle 14（Matmul）** 的分块迭代部分点积；autotune 枚举 BLOCK_SIZE ↔
Puzzle 8~14 里手工选 TPB 的自动化版本。

## 3. 调试方法论（讲义反复强调）

- `TRITON_INTERPRET=1`：CPU 模拟 GPU 执行，kernel 可像普通 Python 一样断点/单步。
  （与 numba 的「CPU 通道」、L003 的 Python 模拟器同一哲学：**先正确，再快**。）
- 推荐流程：模拟器里写对 → tiny 例子验证 → 再上 GPU 提速。
- 讲义特意埋了两个教学 bug：
  1. offsets 没随 pid 平移（永远 `[0,1]`）；
  2. 平移系数错了（`pid*n` 应为 `pid*bs`）。
  「GPU 编程 = 大量索引计算，索引极易错」→ 小例子先行。

## 4. 2D 数据：offset/mask 网格

4×7 矩阵、块 2×2 的例子：`offs_0[:,None]*stride_0 + offs_1[None,:]*stride_1`
外积出 2D 偏移网格，mask 同理 `(offs_0<max_0) & (offs_1<max_1)`。
对应 util 里的 `get_2d_offset/get_2d_mask`。stride 作为参数传入，
所以同一 kernel 兼容 row-major/列-major/转置等任意布局——**布局自由度来自 stride 参数化**。

## 5. Matmul 三部曲

### 5.1 naive matmul

- 分解：m 轴 → pid 维 0；n 轴 → pid 维 1；**k 轴不分块维度**——在单个 program 内循环
  累加（`acc += tl.dot(a, b)`，`tl.dot` 是块级矩阵乘）。
- 与 L003/L004 的 naive matmul 完全同构，只是「线程算一格」变成「program 算一块」。

### 5.2 grouped ordering（swizzling）—— L2 优化

- Triton 管**块内**访存顺序，不管**块间**顺序——块间顺序是留给用户的性能旋钮。
- 原理：相邻执行的 program 们读的数据越重叠，L2 命中率越高。
  行主序下「连续」算一行输出：90 次块读；**group ordering** 让 3×3 的输出块
  「连续」算：54 次块读（group_size=3，可调）。
- 实现：**不改 kernel，改 pid 解释**——`pid_0, pid_1 = swizzle(pid_0, pid_1)`，
  把重排函数作用于 pid 再当普通索引用。讲义先用最小例子演示「改 pid = 改处理顺序」，
  再验证 `tl.swizzle2d` 对 5×4（元素 0..19）行主序做 grouped 重排的效果。
- 内置 `triton.language.swizzle2d` 直接可用，不必手写。

### 5.3 autotune 版

- `@triton.autotune(configs=[...], key=[M, N, K])`：枚举 meta 参数组合
  （BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages），每个候选都实测一遍，留下最快的；
  `key` 变化（如矩阵尺寸变）会触发重新调优。
- 讲义引用 Mark Saroufim 的《CUDA Performance Checklist》作为选 config 的启发式来源。
- 记录的一个未解怪现象：autotune 行偶尔返回错误结果、无法稳定复现（作者公开求复现）。

## 6. Benchmark

- `triton.testing.do_bench` / `perf_report`：Triton 自带基准工具，
  与 torch native / torch.compile 输出对比表。
- 讲义观察：带宽 GB/s 未随矩阵变大而上升（作者存疑：怀疑 shared memory 打满后
  反复重载）——**「讲义留下的开放问题」也是笔记的一部分**。
- 更大 BLOCK_SIZE 更好；调参后与 PyTorch 差距缩小但未反超（该教学 kernel 未用
  tensor core/流水线等全部手段）。
- profiling：`ncu --target-processes all your_python_file.py`（Nsight Compute）。

## 7. Qs.md（FAQ 增补）

讲义作者另写了一篇 gist：**Memory Safety in Triton**
（https://gist.github.com/UmerHA/eb1c623fd71a49b0965079926750faaf）——
讲 mask/`other` 值与越界读写的关系，正是 `tl.load(ptr, mask=…, other=…)` 的语义细节。

## 8. sm_75（Quadro RTX 5000）适配标注

- Triton 支持 sm_75，本讲 notebook（vector add → matmul → swizzle → autotune）
  不依赖新架构特性，可直接跑：`pip install triton`（Linux/WSL2）或
  Windows 用 `triton-windows` 社区包。
  **勘误（2026-10-05 实测）**：此前版本声称"solutions venv 已装 triton 3.4.0"不实——
  venv 实测无 triton，Windows 全局亦未安装；因此本机 torch.compile（inductor 后端
  依赖 triton）对 CUDA 目标**不可用**（实测报错），本讲示例在本机只能停留在
  CPU 模拟（`TRITON_INTERPRET=1`）或阅读层面。AR002 gemm-lab 的手写 GEMM 实验
  因此改用 numba CUDA 真机路线（见 `02-handwritten-kernels/notes/gemm-lab-notes.md`）。
- autotune 的候选空间注意 sm_75 限制：每 block ≤1024 线程 → BLOCK_SIZE×dtype
  折算的 `num_warps` 上限；BLOCK_M=128 一档在大 K 下可能因 shared memory
  （64KB/SM）受限，Turing 上 autotune 会自动淘汰溢出配置。
- `tl.dot` 在 Turing 上走 mma.sync（FP16/INT8）或 FP32 模拟路径——
  讲义示例为 FP32 教学版，速度参考意义有限。

## 9. 一句话总结

Triton 的全部心智模型：**一个 program = 一块向量计算；offset/mask 是积木；
pid 是你可以随意重解释的调度自由度（swizzle）；BLOCK_SIZE/num_warps 交给 autotune**——
而正确性永远先在 `TRITON_INTERPRET=1` 的 CPU 模拟器里挣到手。

## 10. 面试深潜（Interview Deep-Dive）

### 10.1 高频问法与口述骨架

**Q1：「Triton 和 CUDA 的本质区别？」**

> 「抽象层级不同。CUDA 是两级分解：grid→block→thread，线程算标量，shared memory
   和同步手工管。Triton 只有一级：**program（块）**，一个 program 对一整块做向量化
   load/计算/store，块内怎么分 warp、shared memory 怎么用、怎么 bank 分配——全部
   由编译器自动完成。代价是控制粒度粗一点，收益是**容易写出 decent 性能**：
   Python 语法、`tl.arange` 向量原语、`tl.dot` 直接映射 tensor core。」
> 加分：「编译产物同样是 PTX——Triton 编译器做的是自动 tiling + 向量化 +
> shared memory 管理，等于把 CUDA 里最易错的部分自动化了。」

**Q2：「`tl.load` 的 mask 和 other 是什么语义？」**

> 「块大小取 2 的幂（`next_power_of_2`），但数据边界任意——mask 用
   `offs < n` 逐元素标出有效位，越界位置读 `other`（如 `-inf`），store 端同样
   masked。这是 Triton 的边界安全机制：**块是固定形状的向量，mask 是块与世界
   之间的契约**。」（关联 L014 §7 的 Memory Safety FAQ）

**Q3：「什么是 swizzling？为什么能加速？」**

> 「Triton 管**块内**访存顺序，不管**块间**调度顺序——这是留给用户的旋钮。
   相邻执行的 program 读的数据重叠越多，L2 命中率越高。行主序下『连续』处理一行
   输出要 90 次块读；group ordering 把 3×3 的输出块聚在一起『连续』处理，只要
   54 次。实现上不碰 kernel——**把 pid 过一个重排函数（`tl.swizzle2d`），再当
   普通索引用**。」

**Q4：「autotune 怎么工作？」**

> 「`@triton.autotune` 枚举配置空间（BLOCK_M/N/K、num_warps、num_stages），
   每个候选实测一遍，留下最快的编译；`key` 参数（如 M/N/K）变化会触发按新形状
   重新调优。本质是把『块形状选多大』这个经验问题变成离线搜索。」
> 关联 torch.compile：「**torch.compile 生成的自定义 kernel 就是 Triton kernel**，
> 它的产物可以直接当手写起点。」

### 10.2 追问链

1. 「num_warps 是什么？」→ 一个 program 内部切成几个 warp 执行；块越大需要的
   warp 越多（BLOCK≥2048 用 8、≥4096 用 16），它影响占用和调度。
2. 「怎么调试 Triton？」→ `TRITON_INTERPRET=1` 让 kernel 在 CPU 上解释执行，
   可以断点/单步，像普通 Python；讲义还提供了按 pid 条件下断点的工具
   （`breakpoint_if('>1,=0')`）——「先在解释器里挣到正确性，再上 GPU 提速」。
3. 「一个 program 对应 CUDA 里什么？」→ 一个 block；`program_id(0/1/2)` ≡
   blockIdx.x/y/z，只是它被刻意抽象成『块级程序的编号』。
4. 「Triton 的劣势？」→ 控制粒度粗（warp 级原语、手动流水线、异步拷贝受限），
   极限性能场景仍要 CUDA/PTX；讲义也记录了若干 rough edges（偶发不可复现的
   autotune 错误结果）。
5. 「BLOCK_SIZE 为什么取 2 的幂？」→ `tl.arange` 要求端点是 2 的幂；掩码把
   实际数据边界的任意性吸收掉。

### 10.3 数字卡片

- group ordering（group_size=3，9×9 输出块）：**90 → 54 次块读**（9+81 → 27+27）
- 决策顺序口诀：**compile → rewrite → Triton → CUDA**（四级下探）
- `tl.arange(0, BLOCK)` 的 BLOCK 必须是 **2 的幂**；`next_power_of_2(n)` 配 mask
- 调试环境变量：**`TRITON_INTERPRET=1`**（面试里说出这个 = 真写过）

### 10.4 手写题得分骨架（Triton 逐元素/按行 kernel）

默写五要素：① `tl.program_id(0)` 定位 ② `tl.arange(0, BLOCK)` 块内偏移
③ `tl.load(ptr + offs, mask=offs < n, other=…)` ④ 向量化计算 ⑤ masked store。
得分细节：`BLOCK = next_power_of_2(n_cols)`、按块大小调 `num_warps`、
`perf_report`/`do_bench` 与 torch 基线对比。

### 10.5 红线清单

- ❌ 把 Triton program 说成「一个线程」——是**一个块**
- ❌ 忘 mask 直接 `tl.load`（越界未定义行为）
- ❌ 认为 Triton 一定比 CUDA 快（是「容易达到 decent」，上限仍看编译器）
- ❌ 说不出 torch.compile 与 Triton 的关系
- ❌ autotune 只写 configs 不写 `key`（形状变化不重调）

### 10.6 60 秒电梯陈述

> 「Triton 的心智模型是块级编程：一个 program 拿一块向量，offset/mask 是积木，
> 编译器替我管 shared memory 和 warp 划分，产物还是 PTX。性能上我保留两个旋钮：
> 块间调度用 swizzle 提高 L2 命中，块形状和 num_warps 交给 autotune 扫。选型上
> 我遵循 compile→rewrite→Triton→CUDA 的下探顺序——而且 torch.compile 生成的
> kernel 本身就是 Triton，可以直接接手继续优化。正确性我永远先在
> TRITON_INTERPRET=1 里验证，索引 bug 在 CPU 上最便宜。」
