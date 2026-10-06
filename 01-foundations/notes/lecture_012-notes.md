# Lecture 012 笔记 — Flash Attention 前向：Torch → Numba → CUDA

> 讲义目录：`lecture_012/`（flash_attention.ipynb + 3 个 .cu + main.cu）
> 实现 FlashAttention-2 前向（arXiv 2307.08691），d=128，对比三种写法，
> 并用「寄存器溢出对照组」讲透 register spilling。

## 0. 算法本身：online softmax（整讲的地基）

标准 attention 要先算完整 S=QK^T 再 softmax，S 无法装进 SRAM。
Flash 的做法是**分块 + 在线重缩放**，任何时刻只持有当前块的 running 统计：

```
对每个 Q 块 i（m_i=-inf, l_i=0, O_i=0）:
  对每个 K/V 块 j:
    S_ij = scaling * Q_i @ K_j^T
    m_new = max(m_old, rowmax(S_ij))
    P_ij = exp(S_ij - m_new)
    l_i  = exp(m_old - m_new) * l_i + rowsum(P_ij)
    O_i  = exp(m_old - m_new) * O_i + P_ij @ V_j      # ← 老贡献按比例缩放
  O_i = O_i / l_i ;  L_i = m_i + log(l_i)
```

三处易错点（写 kernel 时的检查表）：
1. **先更新 m 再算 P**，P 用的是新 m；
2. **l 和 O 的老值都要乘 `exp(last_m - m)`**——两个缩放系数必须同源；
3. 最终 `L = m + log(l)`（不是 `log(l)`），`O` 归一化在最后一步。

讲义的 `check_close` 同时对拍 O 和 L（L 与 `torch.logsumexp(S,dim=-1)` 比），非常值得抄——
只对拍 O 会漏掉 m/l 路径上的部分 bug。

## 1. Torch 版（可读性基准，CPU 上跑）

`flash_attention_torch`（B_r=B_c=16）：就是上面的伪码直译，
外层 i 循环、内层 j 循环、`last_m_i` 显式保存上一轮的 m。
作用是**定义「正确」**：后面 numba/CUDA 版全部与它和 `F.scaled_dot_product_attention` 双重对拍。

## 2. Numba 版两步走——本讲的性能教学主线

### 2.1 all-smem 版：一切都放 shared memory

`flash_attention_numba_all_smem`：Q_i/K_j/V_j/S 放 shared，**m_i/l_i/O_i 也放 shared**，
单 block 串行处理整个外层循环（`grid=(1,)`）。正确但慢。

### 2.2 讲义的算术题（核心教学内容）

```
Shar = Q_i + K_j + V_j + S ≈ 25 KB
Loc  = m_i + l_i + O_i     ≈ 8 KB   (B_r=16: 4*(16+16+16*128) = 8320 B)
合计 ≈ 33 KB —— 1 block/SM 勉强放下
```

**想法**：把 m_i/l_i/O_i 从 shared 挪到**每线程局部（寄存器）**。
**问题**：若每线程持有全尺寸数组，8320B × 512 线程 ≈ 4MB ≫ 256KB 寄存器堆 → 溢出。
**解法**：按线程分片——`tpb=(32,16)` 时每线程只负责
`d//blockDim.x=4` 列 × `B_r//blockDim.y=1` 行，局部数组缩成
`l_i[1]+m_i[1]+O_i[1][4]` = 24B/线程，全 block 12KB，稳稳落在寄存器里。

这就是「**数据布局跟着线程分片走**」的通用手法：shared 放需要广播/复用的（Q/K/V/S），
寄存器放每线程私有的累加器（m/l/O）。

### 2.3 最终版 numba kernel 的另两个结构变化

- **外层循环并行化**：`for i in range(block_id_x, block_id_x+1)` ——每个 block 只处理
  一个 i，`grid=(T_r,)`。block 数随 N_out 扩展，这是能吃满 GPU 的关键一步
  （all-smem 版永远只有 1 个 block 在干活）。
- S 的计算分两级 `syncthreads()`：装载 K/V 后同步一次，写完 S 再同步一次才能被
  各线程按行读取——与 L004 tiled matmul 的双屏障同构。

CUDA 版 `flash_attention.cu`（B_r=8, B_c=32, blockDim 128×8, d=128）就是它的 C++ 翻译：
`float O_i[B_r_over_bdy][d_over_bdx]` 即按线程分片的寄存器累加器；
S 存 shared 后由**持有该行的线程**独自做 max/exp/累加（`S[ii*bdy+tid_y][jj]`），
P·V 则沿 dd 用 `V_j[jj][dd*bdx+tid_x]` 广播读——行私有、列广播的典型寄存器复用模式。

## 3. `flash_attention_spilling_from_registers.cu` —— 反面教材

故意把全尺寸 `l_i[B_r]; m_i[B_r]; O_i[B_r][d]` 声明为线程局部数组：
8320B/线程不可能住进寄存器 → 编译器把多余部分**spill 到 local memory**
（local memory 实际驻留 DRAM，走 L1/L2 缓存）。

讲义给出两层证据：
1. **PTX 层**：`.local .align 16 .b8 __local_depot0[8320];` —— 8320B 正是那三个数组的
   尺寸，`.local` depot 出现 = 溢出实锤。`nvcc -ptx` 或 godbolt 都能看。
2. **ncu 层**：Duration 13.02ms vs 2.10ms（**~6× 慢**）、L2 吞吐 1.71% vs 0.56%
   （溢出数据在 L2 进进出出）、SM 计算吞吐反而低。

工程口诀：**local array 若不能被编译器完全常量索引展开，就会溢出**；
写小循环界（`B_r_over_bdy`、`d_over_bdx` 是编译期 constexpr）是让数组留在寄存器的必要条件。

## 4. 性能结论（讲义的诚实汇报）

- 小尺寸（32×32）下自定义 kernel 与 `sdpa` 相当；尺寸变大后**落后**——
  因为该实现是标量/朴素分块版，没有 tensor core、没有 warp 级流水。
- spilling 版永远远慢于 sdpa。
- 这讲的定位是「教学正确性与资源布局」，不是打榜 FA2。

## 5. 两个进阶集成方式（拓宽视野）

1. **cuda-python（nvrtc 运行时编译）**：读 .cu → `nvrtcCreateProgram` →
   `--gpu-architecture=compute_{min}{maj}` 用 `torch.cuda.get_device_capability()` 动态拼
   （这正是本仓库 solutions 在 sm_75 上跑通的同款思路）→ `cuModuleLoadData(ptx)` →
   `cuLaunchKernel(grid…, torch.cuda.current_stream().stream_id, args.data_ptr(), 0)`。
   讲义原话「The following code example is not intuitive」——参数打包成
   连续的 int/float/指针对数组再传 `data_ptr()`。
2. **Thunder 注册自定义 sdpa 后端**：`OperatorExecutor` + `register_operator`
   （meta 函数声明输出 shape）+ `register_implementation(thunder.torch.scaled_dot_product_attention,
   checker=…, execution_transform=…)`——checker 决定何时接管，
   transform 里调 `cuLaunchKernel`。让普通 `F.sdpa` 调用自动路由到自己的 kernel。

## 6. main.cu —— 独立 profiling 驱动

不依赖 torch：host 端 `new float[]` + `cudaMalloc/Memcpy`，`CUDA_CHECK` 宏包所有 API，
依次 launch spilling 版与正常版，专供 `nvcc -O3 main.cu flash_attention.cu … &&
ncu ./test_attention` 做对照剖析。**给 profiler 用的最小可复现工程**值得模仿。
讲义目录另有 `torch_extension_template.cu`：`torch::Tensor` 绑定的公共模板，
notebook 的 `get_loaded_cuda_module` 把它与各 kernel 源拼接后经 `load_inline` 编译。

## 7. 与本仓库 solutions / GPU-Puzzles 的关联

- online softmax ↔ **Puzzle 10/12（Dot / Prefix Sum）**：rowmax→exp→rowsum→归一的增量
  合并与 Dot 树状归约、Prefix Sum 块内合并同属一族流水，Flash 把它变成「跨块增量式」
  （全套 14 题没有 softmax 题，这里是算法形态类比而非题号对应）。
- 协作装载 + 双 `__syncthreads()` ↔ Puzzle 11（1D conv）、Puzzle 14（matmul）的结构。
- sm_75 适配：`compute_75` 正是讲义 cuda-python 段动态拼 arch 的用例；
  33KB shared 需求 ≤ Turing 64KB 上限，可跑；`nvcc` 编译需 CUDA Toolkit（本机待装）。

## 8. 一句话总结

L012 用同一算法的三次实现教你 Flash 的真实工程结构：
**shared 放需要广播的块，寄存器放按线程分片的 running 状态，外层循环交给 blockIdx**——
并用一个故意写坏的版本告诉你 `.local depot` 就是寄存器溢出的墓碑。

## 9. 面试深潜（Interview Deep-Dive）

### 9.1 高频问法与口述骨架

**Q1：「FlashAttention 解决什么问题？」**

> 「两个问题。显存：标准 attention 要物化 S=QK^T，O(N²) 空间，长序列直接爆显存；
> 带宽：S 写回 HBM 再读回来做 softmax，HBM 往返主导耗时。Flash 把 Q/K/V 分块，
> 在 SRAM 里完成 S 的计算和 softmax 的**增量归一化**，任何时刻只保留 running
> 统计（m/l/O），从不物化完整 S——显存 O(N)，HBM 读写也大幅下降。」
> 加分：「它不是近似算法，输出与标准 attention **数学等价**，这是和 sparse/linear
> attention 的本质区别。」

**Q2：「online softmax 的递推式写一下 / 口述。」（算法核心，必须滚瓜烂熟）**

```
m_new = max(m_old, rowmax(S_ij))            # 先更新 m
P_ij  = exp(S_ij - m_new)
l_new = exp(m_old - m_new) * l_old + rowsum(P_ij)
O_new = exp(m_old - m_new) * O_old + P_ij @ V_j
# 收尾：O = O / l；L = m + log(l)
```

三个必中要点：① **先更新 m 再算 P**；② **l 和 O 的旧值乘同一个缩放
`exp(m_old - m_new)`**——两处系数必须同源，这是最容易写错的地方；③ 收尾
`L = m + log(l)` 而非 `log(l)`。

**Q3：「为什么它数值稳定？」**

> 「减去行最大值 m 后再做 exp——这就是经典 max-subtraction trick，防止 exp 上溢；
> l 累加的是已缩放的量，log-sum-exp 形式保证下溢也不至于除零（工程上 m 初值取
> -inf 或极小值）。」

**Q4：「kernel 里 shared memory 和寄存器怎么分工？」（工程核心）**

> 「**需要广播/复用的进 shared，每线程私有的累加器进寄存器**。Q/K/V 块和 S 矩阵
> 要被全 block 读——shared；m/l/O 是每行/每列分片的 running 状态——按线程分片放
> 寄存器。L012 的算术：全放 shared 要 33KB 挤 1 block/SM；按 (32,16) 分片后每线程
> 只需 l[1]+m[1]+O[1][4] = 24B，全 block 12KB——同时解决了占用率和带宽。」
> 这就是「**数据布局跟着线程分片走**」的通用手法。

**Q5：「什么是 register spilling？怎么发现？」**

> 「线程局部数组放不进寄存器时，编译器把溢出部分存到 local memory——名字叫
> local，实际驻留 DRAM，走 L1/L2 缓存。三步诊断：① PTX 里搜 `.local` depot
> （`__local_depot0[8320]`，8320B 正好是溢出数组的尺寸）；② `--ptxas-options=-v`
> 看编译期寄存器用量；③ ncu 看 L2 吞吐异常升高 + Duration 增大。L012 实测溢出版
> 慢 **~6×**（13.02ms vs 2.10ms），L2 吞吐 1.71% vs 0.56%——溢出数据在 L2 进出。」
> 预防：「局部数组的循环界必须是编译期常量，才能完全展开留在寄存器。」

### 9.2 追问链

1. 「FA1 和 FA2 的区别？」→ FA2 把外层循环并行化（每 block 处理一个 Q 块，
   `grid=(T_r,)`），减少非 matmul FLOPs、优化了循环顺序——L012 kernel 里
   `for i in range(block_id_x, block_id_x+1)` 就是这个设计。
2. 「为什么要把 L（logsumexp）也写出来？」→ 反向传播重算 softmax 时需要每行的
   归一化常数；对拍时它也是比 `logsumexp(S)` 更强的正确性证据。
3. 「m 初值为什么是 -inf / 1e-30？」→ 保证第一轮 `exp(m_old-m_new)=0` 干净地
   覆盖初始累加器；数值实现用极小值避开 NaN。
4. 「不做 Flash、直接分块 softmax 行不行？」→ 不行——softmax 分母依赖整行，
   分块必须带 running max/sum 重缩放，这正是 Flash 的核心贡献。
5. 「S 矩阵为什么还要进 shared？」→ 行归约（rowmax/rowsum）需要按行读，
   而计算 S 时线程按二维分片写——shared 承担写读布局转接（write-read swizzle）。

### 9.3 数字卡片

- spilling 数组 8320B（`4*(B_r+B_r+B_r*d)`，B_r=16/d=128）→ PTX `.local` depot 实锤
- 溢出 vs 正常：**13.02ms vs 2.10ms（~6×）**，L2 吞吐 **1.71% vs 0.56%**
- shared 用量：全 shared **33KB**（1 block/SM 勉强）→ 分片后 **12KB/块**
- 分片算术：`d/blockDim.x` 列 × `B_r/blockDim.y` 行 / 线程 → 局部数组降两个数量级

### 9.4 手写题得分骨架（online softmax 内循环）

默写四行递推（Q2），外层 `for j in range(T_c)`。**得分细节**：`last_m` 显式保存、
两个 `exp` 系数同源、收尾 `O/l` 与 `m+log(l)`。验证时对拍两样东西：O 对
`sdpa`，L 对 `torch.logsumexp`——「只对拍 O 会漏掉 m/l 路径的 bug」是高级陈述。

### 9.5 红线清单

- ❌ 说 FlashAttention 是「近似/有损加速」——精确等价
- ❌ 递推里忘记缩放 O（只缩放 l）或两处系数不同源
- ❌ 先算 P 再更新 m（顺序颠倒）
- ❌ 「local memory 就是 shared memory」——local 在 DRAM
- ❌ 讲不清为什么 softmax 不能朴素分块

### 9.6 60 秒电梯陈述

> 「Flash 的本质是把 softmax 改造成增量可重缩放的形式：维护行最大 m 和行和 l，
> 每来一块 K/V 就用 exp(m_old-m_new) 把旧累加折算到新基准，再吸收新块的贡献——
> 数学上与整体 softmax 严格等价，但从不物化 N² 的 S 矩阵。kernel 落地时我的布局
> 原则是：要广播的块进 shared、每线程私有的 m/l/O 按分片进寄存器；我会盯 PTX 的
> `.local` depot 和 ncu 的 L2 吞吐来确认没有寄存器溢出——溢出的代价我实测过，
> 6 倍。」
