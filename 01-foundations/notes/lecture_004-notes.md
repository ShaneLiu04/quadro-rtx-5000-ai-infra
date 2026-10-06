# Lecture 004 笔记 — CUDA 性能初探：launch 开销、融合与 tiling

> 讲义文件：`lecture_004/cuda-mode-session-4.ipynb`（Thomas Viehmann，基于 Jeremy L003 notebook；对应 PMPP 书 ch4+ch5）

## 0. 本讲三个实验

1. **GELU 融合**：Python 组合算子 vs 单个 CUDA kernel（kernel fusion 的意义）
2. **空 kernel 计时**：量化 launch 开销的下限
3. **Matmul tiling**：naive → shared memory 分块（PMPP ch5 的核心案例）

## 1. GELU 融合——为什么要写自定义 kernel

### 1.1 基线：Python 手写 tanh 近似 GELU

```python
def gelu(x):
    return 0.5 * x * (1+ torch.tanh((2/torch.pi)**0.5 * (x+0.044715 * x**3)))
```

x = `randn(1024,1024, device='cuda')`。与 `F.gelu(x, approximate='tanh')` 数值一致，
但 Python 版明显更慢——因为它发起**多个 kernel**（mul、add、tanh、mul…），
每个都要读一遍、写一遍全局内存；而 `F.gelu` 是一个融合 kernel。

### 1.2 CUDA 版：一个 kernel 算完整公式

```cpp
__global__ void my_gelu_kernel(float* out, float* inp, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = inp[i];
    out[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f/3.141592653589793f) * (x + 0.044715f*(x*x*x))));
}
```

要点：
- **每个元素只读一次、只写一次**——中间量全在寄存器里，这就是 fusion 的本质。
- C++ 侧要写 `0.044715f`（float 字面量）：讲义脚注提到 1D 灰度 kernel 里
  `0.2989*in[i]` 的 double 提升 bug 由 Andreas 发现修正为 `0.2989f`——
  **double 常量会让整条计算走 FP64 路径**，消费级卡 FP64 极慢（sm_75 上 FP64 吞吐是 FP32 的 1/32）。
- `--ptxas-options=-v`：编译时打印寄存器/共享内存用量，是后续调 occupancy 的基本工具。
- host 封装同时提供 `my_gelu`（自己分配输出）与 `my_gelu_out`（复用调用方 buffer）两种签名。

结论：`%timeit` 下 CUDA 版与 `F.gelu` 同量级，都远快于 Python 组合——
**融合省的不是计算，是访存与 launch**。

## 2. 空 kernel——launch 开销的下限测量

```cpp
__global__ void my_empty_kernel(float* out, float* inp, int n) { }   // 什么都不做
```

- `%timeit my_empty_out(x, x)`：得到**单次 kernel launch + 同步**的固定成本
  （典型量级 5~10 µs；讲义机器约 4-6 µs）。
- 再用 `torch.profiler` 跑 10000 次看 table：确认时间花在 launch/同步而非执行。
- 由此得出两条工程经验：
  1. **kernel 太小不值得 launch**——元素级小操作宁可合并进大 kernel（回到 fusion）；
  2. 任何 benchmark 都要和 launch 开销对照，快过它的优化都是幻觉。

## 3. Matmul：naive → tiled（本讲核心）

### 3.1 naive 版（与 L003 相同）

一个线程算 C 的一个格，k 循环里每次 `m[r*k+i]`、`n[i*w+c]` 都直接读全局内存。
1024×1024×1024 时远慢于 cuBLAS。问题：**A 的行被同 block 列线程重复读，
B 的列被同 block 行线程重复读**，全局内存带宽被浪费。

### 3.2 tiled 版：shared memory 分块

```cpp
constexpr int TILE_SIZE = 16;
__global__ void tiled_matmul_kernel(float* out, float* M, float* N, int h, int w, int k) {
  __shared__ float M_tile[16][16], N_tile[16][16];
  int ir = threadIdx.y, ic = threadIdx.x;                 // 块内坐标
  int r  = blockIdx.y*blockDim.y + ir;
  int c  = blockIdx.x*blockDim.x + ic;
  float res = 0.0f;
  for (int K_tileidx = 0; K_tileidx < cdiv(k, TILE_SIZE); K_tileidx++) {
    M_tile[ir][ic] = (r < h && K_tileidx*16+ic < k) ? M[r*k + K_tileidx*16+ic] : 0.f;  // 越界补 0
    N_tile[ir][ic] = (K_tileidx*16+ir < k && c < w) ? N[(K_tileidx*16+ir)*w + c] : 0.f;
    __syncthreads();
    for (int idx = 0; idx < 16; idx++) res += M_tile[ir][idx] * N_tile[idx][ic];
    __syncthreads();   // 必须！防止快的线程在别人还在读上一轮 tile 时就覆写
  }
  if (r < h && c < w) out[r*w+c] = res;
}
```

逐行拆解五个要点：
1. **`__shared__`**：block 内可见的片上高速缓存（sm_75 每 SM 64KB），
   生命周期与 block 相同。16×16×4B×2 = 2KB/block。
2. **协作加载（cooperative loading）**：block 的 256 个线程各搬 1 个元素进 tile，
   之后 16 次乘加全部命中 shared——**全局读次数从 16×16×k 降到 2×k/16×256**。
3. **边界用 0 填充（zero-padding）**：讲义特意注释「cannot just exit if we want to do padding!」——
   越界线程不能提前 return，否则它们不参与 tile 装载，`__syncthreads()` 会死锁/漏数据。
   这是与 naive 版 guard 逻辑的关键差异。
4. **两次 `__syncthreads()`**：装载后同步（保证 tile 就绪）；累加后同步
   （保证所有线程读完旧 tile 才允许下一轮覆写）。
5. **coalescing 顺带正确**：`threadIdx.x` 是最快变化的位，块内相邻线程读连续地址
   （`M[r*k + ...+ic]` 沿列连续；`N[...*w + c]` 同行相邻）——硬件合并成宽事务。

### 3.3 正确性验证

用**非整除尺寸** `500×200 @ 200×1000` 验证 padding 路径（`(tiled - a@b).abs().max()`）——
整除尺寸测不出边界 bug，这是讲义埋的验证方法论。

### 3.4 Occupancy 分析（讲义结尾的思考题）

> shared memory: 64k/2k → 32 blocks；threads: 1536/256 → 6 blocks
> ⟹ **我们可以负担更大的 tile**

即 shared memory 不是瓶颈（能驻留 32 个 block），线程数才是（每 SM 1536 线程上限 ÷
256 = 6 block）。结论方向：加大 TILE_SIZE 或调整 block 形状让 SM 吃满。
（讲义机器是 A100；sm_75 每 SM 1024 线程上限，同样结论适用，比值变为 1024/256=4。）

## 4. 与本仓库 solutions 的关联

- Tiled matmul ↔ **GPU-Puzzles Puzzle 14**：solutions 里先用协作装载 + `cuda.syncthreads()`
  + zero-padding 的同款结构，验证方式同样是「非整除尺寸必须过」。
- 双 `__syncthreads()` ↔ **Puzzle 11/12（1D conv、Prefix Sum）** 中 shared memory 的
  读写屏障模式：装载后同步一次、使用后视情况再同步。
- 空 kernel 开销测量 ↔ verify/benchmark 里「小尺寸下 launch 开销主导，别看绝对时间」的注意项。

## 5. sm_75 适配标注

- 全部 kernel 无架构特定依赖，sm_75 可跑（需完整 CUDA Toolkit + MSVC 的 load_inline 环境）。
- Occupancy 数字换算到 Turing：64KB shared/SM（可全配 shared），1024 线程/SM，
  16×16 tile 时 shared 限制 = 64KB/2KB = 32，线程限制 = 1024/256 = 4 → 线程仍是瓶颈，
  讲义「值得试更大 tile」的结论在 sm_75 上同样成立。
- GELU kernel 若忘写 `f` 后缀会走 FP64：Turing FP64:FP32 = 1:32，惩罚比 A100(1:2) 更狠——
  讲义那条脚注在本机尤其重要。

## 6. 一句话总结

L004 给出三条性能直觉：**融合（少读少写少 launch）、launch 有固定成本（小 kernel 不值得）、
复用数据上 shared memory（tiling 用协作装载换带宽）**——并用 occupancy 算术
（shared/线程上限相除）告诉你下一个优化旋钮在哪。

## 7. 面试深潜（Interview Deep-Dive）

### 7.1 高频问法与口述骨架

**Q1：「什么是 kernel fusion？为什么有效？」**

> 「不是把计算合并——是把**访存**合并。未融合的算子链每个中间结果都要写回 HBM
> 再读回来；融合后中间量留在寄存器里，每个元素全程只读一次、只写一次。
> 收益 = 消掉的中间读写 + 消掉的 launch 开销。GELU 例子里 Python 组合版发起
> 多个 kernel（mul/add/tanh…），融合版一个 kernel 一个公式。」
> 加分：「fusion 省的是**带宽**，不是 FLOPs——所以对访存 bound 算子收益最大。」

**Q2：「launch 开销多大？怎么量？」**

> 「用空 kernel 测：kernel 体什么都不做，端到端时间就是 launch + 同步的固定成本，
> 实测个位数 µs。两个推论：① 元素级小操作不值得单独 launch，宁可融进大 kernel；
> ② 任何 benchmark 都要和这个下限对照——比它小的『优化』都是噪声。」
> （L004 的空 kernel 实验 + 10000 次 profiler 验证）

**Q3：「讲一下 tiled matmul 为什么比 naive 快。」（性能题之王，备好算术）**

| | naive | tiled (16×16) |
| --- | --- | --- |
| 每线程全局读 | 2×k 次 | 2×(k/16) 次 |
| 每 block 全局读 | — | 2×16×16 次 / tile 轮 |
| 数据复用 | 无 | tile 内 16 次乘加全命中 shared |

> 「naive 里同行线程重复读 A 的同一行、同列线程重复读 B 的同一列，全局带宽浪费。
> tiling 把一个 16×16 的 tile 协作装载进 shared memory——256 个线程各搬 1 个元素，
> 之后 16 次乘加全部命中片上存储；k 维每前进 16 才需要一次全局读。
> 全局读次数从 O(k) 降到 O(k/16)，本质是**用协作装载把带宽需求除以 tile 边长**。」

**Q4：「tiled matmul 里两次 `__syncthreads()` 分别防什么？」**

> 「第一次：装载后同步，保证所有线程把 tile 写完才开始读——否则读到脏数据。
> 第二次：累加后同步，保证所有线程把**上一轮** tile 读完，才允许任何线程开始写
> **下一轮** tile——否则快的线程覆写慢的线程还要读的数据。」
> 追问「能不能只写一次？」→ 「不能，这是生产者-消费者两个方向各需一道栅栏。」

**Q5：「occupancy 怎么算？」（现场口算题）**

> 「一个 SM 能同时驻留几个 block = 三类资源取最小：
> 线程数上限÷block 线程数、shared memory 总量÷每 block shared 用量、寄存器总量÷每 block 寄存器用量。
> L004 例子：64KB/2KB = 32，1536/256 = 6 → shared 不是瓶颈、线程数是，
> 结论『可以加大 tile』。**occupancy 高不等于快**，但 occupancy 太低一定喂不饱
> 延迟隐藏所需的并行度。」

### 7.2 追问链

1. 「越界线程为什么不能 early return？」→ tile 协作装载需要**全部**线程参与搬运；
   early return 的线程不搬数据、也不到栅栏——轻则数据缺口，重则 `__syncthreads()` 死锁。
   正确做法：留在 kernel 里，用条件把越界装载替换成 0（zero-padding）。
2. 「`0.2989` 和 `0.2989f` 有区别吗？」→ 有——不带 f 的 double 字面量会把整条
   表达式提升成 FP64 计算；Turing FP64 吞吐是 FP32 的 **1/32**，性能直接崩。
   「我在讲义里见过这个真实 bug」是绝佳的细节展示。
3. 「tile 是不是越大越好？」→ 受三种资源约束（见 Q5）；tile 过大 → 驻留 block 数
   下降 → 延迟隐藏变差；最优 tile 通常靠 benchmark/autotune 扫出来。
4. 「什么是 coalescing？」→ 同一 warp 的 32 线程访问 32 个连续 4B 地址，硬件合并成
   一个 128B 事务；不连续则拆成多个事务、带宽利用率骤降。tile 装载天然 coalesce：
   `threadIdx.x` 是最快变化位（L004 代码注释原话）。
5. 「shared memory 有什么坑？」→（延伸，标明是下一讲范围）bank conflict：
   shared 分 32 个 4B bank，同一 warp 内两线程落同一 bank 不同地址则串行化；
   常见解法是 padding 一列打破冲突。

### 7.3 数字卡片

- 16×16 tile × 4B × 2 矩阵 = **2 KB** shared / block
- naive→tiled 全局读：**k → k/16**（tile 边长就是带宽除数）
- Turing FP64:FP32 = **1:32**（本机）；A100 是 1:2——「消费卡别碰 double」
- sm_75：1024 线程/SM、64 KB shared/SM → 16×16 tile 时 shared 允许 32 block、
  线程只允许 4 block → **线程数才是瓶颈**（现场口算素材）
- 空 kernel launch ≈ **4~6 µs**（讲义实测）

### 7.4 手写题得分骨架（tiled matmul）

默写五要素：① `__shared__` 两个 tile ② 协作装载 + 越界补 0 ③ 装载后 `__syncthreads()`
④ 块内乘加累加 ⑤ 累加后 `__syncthreads()`，循环外写回。**验证时用非整除尺寸**
（如 500×200）——「我会特意跑非整除尺寸验证 padding 路径」是高质量的主动陈述。

### 7.5 红线清单

- ❌「fusion 是为了减少计算量」——是减少**访存**
- ❌ tiled matmul 只写一次 syncthreads
- ❌ 越界线程 early return（暴露对协作装载理解不足）
- ❌ 说 occupancy 越高性能越好（必要非充分，讲延迟隐藏的权衡）
- ❌ 忘 f 后缀 / 说「无所谓」

### 7.6 60 秒电梯陈述

> 「我的性能直觉是三条：融合省带宽、launch 有 µs 级下限、复用数据上 shared memory。
> 写 tiled kernel 时我有固定纪律——全员参与装载、越界补零不早退、生产和消费两道
> 栅栏、非整除尺寸验证。调优时先口算 occupancy 的三个除法找瓶颈，再决定是加大
> tile 还是调整 block 形状，而不是盲改。」
