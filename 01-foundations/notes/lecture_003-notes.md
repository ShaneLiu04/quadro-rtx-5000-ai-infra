# Lecture 003 笔记 — Getting started with CUDA（Python 模拟 → CUDA C++）

> 讲义文件：`lecture_003/pmpp.ipynb`（Jeremy Howard，基于 PMPP 书）
> 主线：先用**纯 Python 逐步模拟** GPU 的执行模型（kernel → block kernel → 2D block kernel），
> 再把同一逻辑翻译成 CUDA C++。这是本仓库 GPU-Puzzles solutions 的思想源头。

## 1. RGB→灰度：四级递进

数据：puppy.jpg，resize 到 150（CHW 布局，`ch,h,w = x.shape`）。

### 1.1 Basic Python（串行循环）

```python
def rgb2grey_py(x):
    x = x.flatten()            # CHW 摊平后：R 在 [0,n)，G 在 [n,2n)，B 在 [2n,3n)
    for i in range(n): res[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n]
```

注意 `flatten` 之后三个通道靠 `+n`、`+2*n` 偏移访问——这是后面 CUDA 版的索引来源。

### 1.2 Python Kernel（模拟「每个线程一个元素」）

```python
def run_kernel(f, times, *args):
    for i in range(times): f(i, *args)
```

**关键约束：kernel 不能有返回值，只能修改传入参数的内容**——与 CUDA 语义一致
（`__global__` 函数返回 void，结果写回指针）。

### 1.3 Python Block Kernel（模拟 grid/block 分解）

```python
def blk_kernel(f, blocks, threads, *args):
    for i in range(blocks):
        for j in range(threads): f(i, j, threads, *args)

def rgb2grey_bk(blockidx, threadidx, blockdim, x, out, n):
    i = blockidx*blockdim + threadidx          # 全局索引 = block*blockDim + thread
    if i < n: out[i] = ...
```

host 侧配置与真 CUDA 完全同构：`threads=256; blocks=ceil(h*w/threads)`。

### 1.4 硬件背景（讲义给出，换算到本机 sm_75）

- SM 是基本执行单元；一个 block 的所有线程跑在同一个 SM 上，可共享
  **shared memory** 并互相同步。
- 讲义数字（RTX 3090, GA102）：82 SM × 128 CUDA cores，128KB L1/shared，256KB 寄存器堆。
- **本机 Quadro RTX 5000（sm_75/TU104）**：48 SM × 64 FP32 cores，64KB L1/shared 每 SM。
  换算方法即本表做法——读架构白皮书，把 SM 数、核数、shared 容量对齐到自己的卡。

## 2. CUDA C++ 版（`load_inline` JIT 编译）

### 2.1 公共前缀 `cuda_begin`（值得整段抄走）

```cpp
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
```

### 2.2 1D 版 kernel + host 封装

```cpp
__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) out[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n];
}
// host: threads=256, blocks=cdiv(w*h,threads), C10_CUDA_KERNEL_LAUNCH_CHECK()
```

要点：
- grid 限制：dim0 最多 2^31 个 block，dim1/2 最多 2^16；每 block 最多 1024 线程（**取 32 的倍数**，对齐 warp）。
- `wurlitzer` 扩展把 CUDA C 层的 `printf` 输出拉回 notebook。
- `os.environ['CUDA_LAUNCH_BLOCKING']='1'`：让 launch 同步化，方便定位是哪一次 launch 出错。

### 2.3 2D 版 kernel（与 Lecture 002 对照）

同一问题改成 `int c = blockIdx.x*blockDim.x+threadIdx.x; int r = blockIdx.y*blockDim.y+threadIdx.y;`
+ `if (c<w && r<h)`，`dim3 tpb(16,16)`。**1D 摊平版与 2D 版是同一计算的两种分解**，
讲义两版都写了，帮助建立「grid 形状是自由选择」的直觉。

## 3. Matmul：同样四级递进

MNIST 数据（50000×784 float）× 权重（784×10）。

### 3.1 纯 Python 三重循环 → `ar*bc*ac` 次标量乘加（50000×10×784 ≈ 3.9 亿次）。

### 3.2 2D Python block kernel

```python
def matmul_bk(blockidx, threadidx, blockdim, m, n, out, h, w, k):
    r = blockidx.y*blockdim.y + threadidx.y     # 行由 y 维负责
    c = blockidx.x*blockdim.x + threadidx.x     # 列由 x 维负责
    if (r>=h or c>=w): return
    o = 0.
    for i in range(k): o += m[r*k+i] * n[i*w+c]
    out[r*w+c] = o
```

- 模拟器 `blk_kernel2d` 用 `SimpleNamespace` 模拟 `dim3`（x/y 两个 for 嵌套）。
- **一个线程算输出一个格**，k 维留在循环里——这是 naive matmul 的定义。
- 与 GPU-Puzzles Puzzle 14（Matmul）完全同构；本仓库 solutions 的 Puzzle 14 就是这个分解 + shared memory 优化。

### 3.3 Broadcasting 优化（CPU 侧对比）

`a[i,:,None] * b).sum(0)`：把最内层 k 循环交给 PyTorch 向量化——**CPU 上提速三个数量级**。
讲义借此说明：先确认瓶颈真的在「无法向量化的循环」上，再上 GPU。

### 3.4 CUDA matmul（16×16 tpb）

与 3.2 一一对应的 C++ 翻译；`torch.isclose(..., atol=1e-5)` 对拍，最后与
`m1c@m2c`（cuBLAS）对比耗时——naive 版慢 1~2 个数量级，为后续讲义的
shared memory / tiling / tensor core 优化埋下伏笔。

## 4. 本讲方法论（最重要的一条）

**「Python 模拟器先行」**：`run_kernel` / `blk_kernel` / `blk_kernel2d` 三个纯 Python
函数就是极简 GPU 语义模拟器。先在 CPU 上验证索引逻辑，再翻译成 CUDA。
本仓库 GPU-Puzzles 的 CPU 通道（`verify.py` 用 numba `cuda.grid` 语义逐 kernel 对拍）
正是这套方法的工程化版本。

## 5. sm_75 适配标注

- notebook 全部 kernel（1D/2D 灰度、naive matmul）不依赖新架构特性，sm_75 可跑；
  需 `load_inline` 环境（CUDA Toolkit + MSVC），当前机器未装完整 toolkit，可先跑 Python 模拟部分。
- 讲义硬件数字按 RTX 3090 给出；对照本机：48 SM、64 FP32 core/SM、
  64KB shared/SM、1024 线程/SM 上限、每 block 上限 1024 线程。

## 6. 一句话总结

L003 的本质是把 CUDA 执行模型「降维」成三个 Python 循环——
kernel=函数、block=外层循环、thread=内层循环、全局索引=blockidx*blockdim+threadidx——
让你在写第一行 C++ 之前就已经会写 CUDA。

## 7. 面试深潜（Interview Deep-Dive）

### 7.1 高频问法与口述骨架

**Q1：「二维矩阵每个线程处理一个元素，索引怎么写？哪个维度给 x？」（高频手写/口述）**

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;   // x = 列（连续方向）
int row = blockIdx.y * blockDim.y + threadIdx.y;   // y = 行
if (row < H && col < W) out[row * W + col] = in[row * W + col] + 10;
```

关键陈述：「**x 对应内存连续的维度**——`threadIdx.x` 是块内变化最快的位，
相邻线程访问相邻地址，硬件才能合并成一个宽事务（coalescing）。如果行列对调，
每次访存被拆成 32 个独立事务，带宽掉一个数量级。」（对照 L002 §1.2、L004 tile 装载）

**Q2：「SM / block / warp / thread 是什么关系？」**

> 「SM 是物理执行单元（本机 48 个），block 是逻辑调度单位——一个 block 整体驻留在
> 一个 SM，直到跑完；warp 是 SM 的实际调度单位，32 线程锁步执行，同一 warp 内
> 分叉（if 走不同支）会两支都执行、串行化；thread 是编程抽象，一个线程一个标量
> 数据流。shared memory 和 syncthreads 是 block 级语义，寄存器是线程私有。」

**Q3：「写 CUDA 前你怎么验证索引逻辑？」（展示方法论的送分题）**

> 「我习惯先在 CPU 上跑一个执行模型模拟器——把 kernel 写成普通函数，
> 用两层循环枚举 block 和 thread，传入伪造的 blockIdx/threadIdx/blockDim。
> 索引算错在模拟器里立刻暴露，不用等 GPU 上的诡异越界。我在 GPU-Puzzles
> 练习里把这套做法工程化成了带 Barrier 栅栏的完整模拟器。」
> （→ 直接衔接项目陈述，见 INTERVIEW-INDEX）

### 7.2 追问链

1. 「为什么同一 block 必须在同一 SM？」→ shared memory 是 SM 内的物理存储，
   `__syncthreads()` 是 SM 内的硬件栅栏；跨 SM 无法廉价同步。
2. 「block 之间要同步怎么办？」→ 标准答：拆成两个 kernel（kernel 边界是全局同步点）；
   进阶答：cooperative groups / grid sync（有前提，谨慎提）。
3. 「grid 一样大为什么 block 形状影响性能？」→ 影响占用率与尾部效应：
   block 太大→驻留 block 数少、负载不均；太小→launch/调度开销占比高。
4. 「CHW 和 HWC 布局对 kernel 意味着什么？」→ 只是 stride 参数化不同：
   把 stride 当参数传，同一 kernel 兼容两种布局（L003 灰度 1D 版直接靠 `+n` 偏移访问通道）。

### 7.3 数字卡片（本机 sm_75）

- **48 SM × 64 FP32 core = 3072 CUDA core**；384 个第二代 Tensor Core
- 每 SM：**64 KB** L1/shared（可全配 shared）、最多 **1024** 线程、256 KB 寄存器堆
- 1 个 block ≤ 1024 线程 = 32 warp；「一个 SM 至少驻留 2 个 512 线程 block 才不浪费调度槽」
- 记忆口诀：**「4864」= 48 SM × 64 core**（RTX 5000）

### 7.4 红线清单

- ❌ 行/列与 x/y 映射说反还讲不出 coalescing 理由
- ❌ 把 warp 说成「编译器自动并行化的循环」——它是硬件锁步调度单位
- ❌ 「block 内线程同时执行」——是**并发**（concurrently scheduled），同步要显式 syncthreads
- ❌ 认为 shared memory 跨 block 可见

### 7.5 60 秒电梯陈述

> 「CUDA 编程模型的全部就是一层索引代数：全局索引 = blockIdx×blockDim+threadIdx，
> 高维每维独立做一遍，配上 guard 就不会越界。我在写任何 kernel 前都会先在 CPU
> 模拟器里把索引跑对——因为我见过太多 bug 不在算法而在 off-by-one 的索引上。
> 性能上我第一个检查的是维度映射：连续内存必须对应 threadIdx.x，否则 coalescing
> 全毁。」
