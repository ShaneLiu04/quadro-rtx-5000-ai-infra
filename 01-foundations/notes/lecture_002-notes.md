# Lecture 002 笔记 — 三个经典 CUDA C++ 入门例子

> 讲义目录：`01-foundations/gpu-mode-lectures/lecture_002/`
> 上游 README：需要 PyTorch（2.1.2）与 CUDA Toolkit（nvcc）。三个例子：`vector_addition`（纯 CUDA C++，`make` 编译）、`rgb_to_grayscale` 与 `mean_filter`（`load_inline` 编译，输出 `output.png`）。

## 1. `vector_addition` — CUDA C++ 的 "Hello World"

文件：`vector_addition/vector_addition.cu`（含 Makefile）

### 1.1 kernel 本体

```cpp
__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;   // 全局线程索引
  if (i < n) { C[i] = A[i] + B[i]; }               // guard：线程数 > n 时保护
}
```

### 1.2 host 侧五步流程（所有 CUDA 程序的骨架）

1. `cudaMalloc` 分配 device 内存（A_d/B_d/C_d）
2. `cudaMemcpy(…, cudaMemcpyHostToDevice)` 上传输入
3. 配置并启动：`vecAddKernel<<<numBlocks, numThreads>>>(…)`，
   其中 `numThreads=256`，`numBlocks = cdiv(n, numThreads)`（向上取整除法）
4. `gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());`
5. `cudaMemcpy(…, cudaMemcpyDeviceToHost)` 取回结果，`cudaFree` 释放

### 1.3 值得抄走的部分：错误检查宏

```cpp
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
```

CUDA API 调用不会抛异常，**错误是粘滞的**（sticky error）：不检查的话错误会
在很久之后的下一个 CUDA 调用中才冒出来，极难定位。生产代码每次 API 调用后都应检查。

**与 GPU-Puzzles 的关联**：这就是 Puzzle 2（Zip）+ Puzzle 6（Blocks）的 C++ 原型。
Puzzle 3 的 guard 对应这里 `if (i < n)`；256 线程/block 是实践常用值。

## 2. `rgb_to_grayscale` — 2D 图像 kernel + PyTorch 张量接口

文件：`rgb_to_grayscale/grayscale_kernel.cu` + `rgb_to_grayscale.py`

### 2.1 kernel 本体（2D 版）

```cpp
__global__ void rgb_to_grayscale_kernel(unsigned char* output, unsigned char* input, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int outputOffset = row * width + col;             // 输出是单通道 2D
        int inputOffset = (row * width + col) * channels; // 输入是 CHW 3 通道
        unsigned char r = input[inputOffset + 0];
        unsigned char g = input[inputOffset + 1];
        unsigned char b = input[inputOffset + 2];
        output[outputOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}
```

要点：
- **一个线程算一个像素**，2D grid/block（`dim3 threads_per_block(16, 16)`）。
- 输入张量是 PyTorch 的 CHW 布局（channel-major），像素寻址要乘 `channels`。
- 加权系数 0.21/0.71/0.07（红/绿/蓝，讲义取的近似值）。
- 与 Lecture 003 notebook 里的 1D 版（把图像摊平、`i = blockIdx.x*blockDim.x+threadIdx.x`
  一次性索引 `w*h` 个像素，三个通道靠 `+n`、`+2n` 偏移）互为对照——**同一问题的两种分解**。

### 2.2 与 PyTorch 集成的细节

- 在 `torch::Tensor` 上直接取 `data_ptr<unsigned char>()`；断言 device/dtype。
- launch 用 `torch::cuda::getCurrentCUDAStream()`——**不要用默认流**，否则与
  PyTorch 异步执行模型脱节，出现竞态。
- `C10_CUDA_KERNEL_LAUNCH_CHECK()` 等 PyTorch 封装的错误检查。

**与 GPU-Puzzles 的关联**：2D 索引 + 双向 guard ↔ Puzzle 4（Map 2D）/ Puzzle 7（Blocks 2D）。

## 3. `mean_filter` — 3D block（x, y, channel）+ 邻域窗口

文件：`mean_filter/mean_filter_kernel.cu` + `mean_filter.py`

### 3.1 kernel 本体

```cpp
__global__ void mean_filter_kernel(unsigned char* output, unsigned char* input, int width, int height, int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = threadIdx.z;                    // 第三维线程直接对应通道！
    int baseOffset = channel * height * width;
    if (col < width && row < height) {
        int pixVal = 0, pixels = 0;
        for (int blurRow=-radius; blurRow <= radius; blurRow += 1)
            for (int blurCol=-radius; blurCol <= radius; blurCol += 1) {
                int curRow = row + blurRow, curCol = col + blurCol;
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    pixVal += input[baseOffset + curRow * width + curCol];
                    pixels += 1;
                }
            }
        output[baseOffset + row * width + col] = (unsigned char)(pixVal / pixels);
    }
}
```

要点：
- **`threadIdx.z` 当通道索引用**：block 是 `(16, 16, channels)` 三维——线程的第三维
  不必是空间维度，任何「并行轴」都可以映射到线程维度。
- 邻域累加的**边界处理是逐像素条件判断**（`curRow/curCol` 越界跳过），
  `pixels` 计实际参与平均的个数——边界像素的窗口更小。
- 这是**每个线程多次读全局内存**的朴素版本（(2r+1)² 次读/像素）；
  优化方向正是 shared memory 分块共享邻域（本讲未展开，GPU-Puzzles Puzzle 9 的思路）。

**与 GPU-Puzzles 的关联**：邻域窗口求和 ↔ Puzzle 9（Pooling：窗口 3 的局部和）。
Puzzle 9 要求每线程只 1 次全局读——手段就是先把块数据载入 shared memory，
这正是本例朴素实现的优化下一步。

## 4. sm_75（Quadro RTX 5000）适配标注

- 三个例子只用基础 CUDA 特性，sm_75 完全可跑；但**本机当前无 nvcc**
  （CUDA 目录只剩 extras），要跑需装 CUDA Toolkit 12.5 或改用 pip 组件
  （本仓库 solutions/.venv 已验证 numba + NVVM 路径；CUDA C++ `load_inline` 路径
  需要完整 toolkit + MSVC）。
- `vector_addition` 是纯 C++ 程序，不依赖 PyTorch，适合做 nvcc 装好后的第一个编译目标。
- 16×16=256 线程/block：sm_75 每 SM 最多 1024 线程、64KB shared memory，
  该配置在 Turing 上同样合理。

## 5. 一句话总结

第二讲把「host 五步流程 + guard + cdiv + 错误检查」焊成一个肌肉记忆，
并展示线程维度可以映射任意并行轴（空间 x/y、颜色通道 z）——
这是从「能跑」到「会分解问题」的第一步。

## 6. 面试深潜（Interview Deep-Dive）

### 6.1 高频问法与口述骨架

**Q1：「讲一下 CUDA 的执行模型。」（开场必问，60 秒版）**

> 「GPU 由若干 SM 组成；host 启动 kernel 时给定 grid（block 的三维网格）和 block
>（线程的三维块）。所有线程执行同一份 kernel 代码，靠内建索引
> `blockIdx*blockDim+threadIdx` 区分各自处理的数据。同一 block 的线程保证落在
> 同一 SM，可以共享 shared memory 并用 `__syncthreads()` 互相同步；不同 block
> 之间不保证执行顺序、不共享内存。block 是调度单元，SM 上驻留多少 block 取决于
> 线程数/shared memory/寄存器三类资源。」

**Q2：手写 vector add（最常见手写题，默写级熟练）**

```cpp
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 全局索引
    if (i < n) c[i] = a[i] + b[i];                   // guard
}
// host: cudaMalloc ×3 → cudaMemcpy H2D ×2 →
//       vecAdd<<<(n+255)/256, 256>>>(...) → cudaDeviceSynchronize → cudaMemcpy D2H → cudaFree
```

得分点 checklist：① 索引公式 ② guard ③ **cdiv 向上取整** ④ 拷贝方向
⑤ 错误检查（下面 Q3）。漏 guard 是现场最常见扣分。

**Q3：「CUDA 的错误怎么处理？」**

> 「CUDA API 不抛异常，返回错误码，而且错误是**粘滞的**（sticky）——不检查的话，
> 真正的错误会潜伏到之后某个无关调用才冒出来。所以生产代码每次 API 调用后都包
> `gpuErrchk(...)` 宏；kernel launch 后查 `cudaPeekAtLastError()`，同步点查
> `cudaDeviceSynchronize()`。」

### 6.2 追问链

1. 「为什么 block 常取 256 线程？」→ 32 的倍数（整 warp 不浪费）+ 256 允许一 SM 驻留
   多个 block，占用与调度灵活性平衡；16×16 的 2D block 也是 256。
2. 「grid/block 的上限？」→ grid.x ≤ 2³¹，grid.y/z ≤ 2¹⁶；每 block ≤ **1024** 线程
   （三个维度乘积），超出即 launch 失败。
3. 「2D 数据哪个维度对应 x？」→ **连续内存的维度给 x**（threadIdx.x 变化最快），
   保 coalescing——这题答对直接进入 coalescing 话题，是 L003/L004 的钩子。
4. 「线程第三维（z）能干嘛？」→ L002 的 mean_filter 用 `threadIdx.z` 当颜色通道——
   「线程维度可以映射任何并行轴，不必是空间轴」是展示理解深度的好例子。

### 6.3 数字卡片

- warp = **32** 线程（调度与分支的基本单位）
- 每 block 上限 **1024** 线程；实践取 **32 的倍数**，常用 128/256
- `cdiv(a,b) = (a+b-1)/b`——口算：9 数据/4 线程 → 3 block，第 3 个 block 只有 1 个有效线程

### 6.4 红线清单

- ❌ 手写 vector add 忘 guard / 忘 cdiv
- ❌ 「错误打印一下就行」——不知 sticky error 特性
- ❌ `cudaMemcpy` 方向参数写反（H2D/D2H）
- ❌ 把 shared memory 说成「所有 block 共享」（是**block 内**共享）

### 6.5 60 秒电梯陈述

> 「我能从零写对带边界处理的 CUDA kernel：全局索引公式、guard、cdiv 配 grid 是
> 肌肉记忆；我知道同一 block 落同一 SM 才有 shared memory 和 syncthreads 的语义；
> 生产代码我会用宏包住每个 CUDA API——因为 CUDA 错误是粘滞的，不查会把错误
> 拖到离病因很远的地方。」
