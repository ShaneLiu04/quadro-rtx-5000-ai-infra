# Lecture 001 笔记 — 如何 authoring & profiling 一个 GPU kernel

> 讲义目录：`01-foundations/gpu-mode-lectures/lecture_001/`
> 上游 README：本讲是 CUDA MODE 系列第一讲的杂项笔记与脚本，主题是「用各种 GPU 编程语言写 kernel + 做 profiling」。

## 0. 本讲文件地图（逐文件）

| 文件 | 内容 | 关联知识点 |
| --- | --- | --- |
| `pytorch_square.py` | PyTorch 三种写法做逐元素平方 + `torch.profiler` 用法 | GPU 计时正确姿势、profiler |
| `numba_square.py` | Numba CUDA 写 2D 逐元素平方 | `cuda.grid(2)`、grid/block 配置 |
| `triton_square.py` | Triton 写按行平方（改自官方 fused-softmax 教程） | `tl.arange/load/store/mask`、`num_warps`、`perf_report` |
| `load_inline.py` / `hello_load_inline.py` / `load_inline_cuda/` | `torch.utils.cpp_extension.load_inline` JIT 编译 CUDA C++ | CUDA C++ 扩展、`<<<blocks, threads>>>` |
| `nsys_square.py` | Nsight Systems / torch profiler 时间线 | 系统级 profiling |
| `pt_profiler.py` | torch profiler 的 schedule（wait/warmup/active） | 迭代级 profiling |
| `square_kernel.ptx` | 编译产物 PTX | 编译管线：source → PTX → SASS |
| `ncu_logs` / `triton_profile` | Nsight Compute / Triton profiling 输出样例 | kernel 级 roofline |

## 1. `pytorch_square.py` — 先学会「正确地计时」

**关键点：CUDA 是异步的，不能用 Python `time` 模块直接测。**

```python
start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
# warmup 若干次后：
start.record(); func(input); end.record(); torch.cuda.synchronize()
return start.elapsed_time(end)
```

- Warmup 必不可少：首次调用含 JIT/缓存/频率爬升，不代表稳态。
- `torch.cuda.Event` 在 GPU 时间线上打点，`synchronize()` 等流干。
- 三种平方写法 `torch.square(a)` / `a**2` / `a*a` 性能不同，用 `torch.profiler` 的
  `key_averages().table(sort_by="cuda_time_total")` 看真实 kernel 名与耗时。

**与 GPU-Puzzles 的关联**：Puzzle 1（Map）就是 `out = a + 10` 的逐元素操作；本讲的
square 即同一模式。RTX 5000（sm_75）上跑此脚本需 torch cu121（本机已装 2.5.1+cu121，可直接跑）。

## 2. `numba_square.py` — Numba CUDA 的标准模板

```python
@cuda.jit
def square_matrix_kernel(matrix, result):
    row, col = cuda.grid(2)          # 2D 全局线程坐标（一行顶 blockIdx*blockDim+threadIdx）
    if row < matrix.shape[0] and col < matrix.shape[1]:   # guard
        result[row, col] = matrix[row, col] ** 2
```

- `cuda.grid(2)` = `(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y)` 的语法糖。
- host 侧四步：`cuda.to_device` 上传 → `cuda.device_array` 分配输出 →
  `kernel[blocks_per_grid, threads_per_block](...)` 启动 → `copy_to_host` 回收。
- grid 尺寸用 `ceil(行/TPB)` 计算（即 `cdiv`）。

**与 GPU-Puzzles 的关联**：对应 Puzzle 3/4/6/7（guard、2D、block 索引）。
本仓库 solutions 的 CPU 模拟器验证的正是同一套语义。

## 3. `triton_square.py` — Triton 的思维模型（块级编程）

```python
@triton.jit
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)                       # 一个 program 处理一行
    col_offsets = tl.arange(0, BLOCK_SIZE)           # 块内向量偏移
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))  # 带掩码整行加载
    square_output = row * row                        # 向量化运算
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)
```

- `BLOCK_SIZE = triton.next_power_of_2(n_cols)`：块大小取 ≥ 列数的 2 的幂，掩码兜住越界。
- `num_warps` 按 BLOCK_SIZE 加大（≥2048 用 8，≥4096 用 16）——编译期并行度旋钮。
- `triton.testing.perf_report` + `do_bench`：内置基准工具，与 torch-native / torch.compile 对比。

**与 GPU-Puzzles 的关联**：Puzzle 8（Shared）在 CUDA 里手工做的事（分块载入共享内存），
Triton 的 `tl.load` 到「块寄存器」+ 编译器自动 shared memory 管理，对应同一思想。

## 4. `load_inline.py` — CUDA C++ 与 PyTorch 的最短路径

- `torch.utils.cpp_extension.load_inline(name, cpp_sources, cuda_sources, functions, with_cuda=True)`
  运行时 JIT 编译出可 `import` 的模块；`build_directory` 缓存构建产物（`build.ninja`、`cuda.cuda.o`）。
- kernel 本体就是经典 CUDA C++：

```cpp
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) { ... }
}
```

- 讲义记录的坑：`ncu python load_inline.py` 下 CUDA 初始化失败（Error 36）——
  profiling 工具与 CUDA 运行时的兼容性问题，需 `--target-processes all` 等选项。

## 5. profiling 工具链小结（本讲核心收获）

| 工具 | 层级 | 看什么 |
| --- | --- | --- |
| `torch.profiler` | PyTorch 算子级 | 算子名、CPU/CUDA 时间、trace 导出 chrome json |
| Nsight Systems (`nsys`) | 系统级时间线 | kernel 间隙、拷贝与计算重叠、launch 开销 |
| Nsight Compute (`ncu`) | 单 kernel | roofline：算力 vs 带宽 vs 显存占用 |

`pt_profiler.py` 的 schedule 模式值得记住：`wait=1, warmup=1, active=2, repeat=1` +
`on_trace_ready` 回调，适合训练循环里「跳过第一步、预热一步、记录两步」。

## 6. sm_75（Quadro RTX 5000）适配标注

- 本讲所有脚本不依赖 Hopper/Ampere 特性，sm_75 可直接跑（torch 路径已就绪；
  numba 路径即本仓库 solutions/.venv 的装配方式）。
- Triton 支持 sm_75；`triton_square.py` 可直接运行。
- 讲义默认 GPU 是 RTX 3090/4090 一档，读性能数字时注意换算到本卡：
  FP32 ≈ 11.2 TFLOPS，显存带宽 448 GB/s，48 SM。

## 7. 一句话总结

第一讲的主线不是某个算法，而是**「写 kernel 的 4 条路（PyTorch / Numba / Triton / CUDA C++）
+ 验证正确性的对拍 + 三层 profiling 工具」**——这套工作流是后面所有讲义的地基。

## 8. 面试深潜（Interview Deep-Dive）

### 8.1 高频问法与口述骨架

**Q1：「你怎么测量一个 CUDA kernel 的耗时？」（几乎必问）**

口述骨架（30 秒版）：
> CUDA 是异步执行模型——launch 立刻返回，kernel 在 device 队列里跑。所以 host 侧
> `time.time()` 测的是 launch 时间不是执行时间。正确做法三种：
> ① `torch.cuda.Event(enable_timing=True)` 打点 + `torch.cuda.synchronize()`，测稳态；
> ② 先 warmup 若干轮再测（首含 JIT/缓存/频率爬升，不代表稳态）；
> ③ 工具级：`triton.testing.do_bench` / `%timeit` + synchronize。

**Q2：「torch.profiler / Nsight Systems / Nsight Compute 各看什么？」**

| 层级 | 工具 | 回答一句话 |
| --- | --- | --- |
| 算子级 | torch.profiler | 「对上框架算子名，看 CPU/CUDA 时间分布、谁在 launch、导出 trace」 |
| 系统级 | nsys | 「看时间线：kernel 间隙、H2D/D2H 重叠、launch 开销、多流」 |
| 单 kernel | ncu | 「看 roofline：算力利用率 vs 带宽利用率 vs shared/寄存器占用」 |

加分句：「三层是从粗到细的漏斗——先 nsys 找到慢的 kernel 和空隙，再 ncu 对单个 kernel
问『是算力 bound 还是带宽 bound』，决定优化方向完全不同。」

**Q3：「什么时候写 Triton/CUDA，什么时候 torch.compile？」**

决策树（L001/L014 一致口径）：模型不够快 → ① `torch.compile` → ② 改写代码让 compile
更好使 → ③ 热点手写 Triton → ④ 仍不够才 CUDA。加分：**「torch.compile 生成的自定义
kernel 本身就是 Triton，可以把它的产物当手写起点」**——这句话能明显拉开档次。

### 8.2 追问链（面试官的下一问）

1. 「为什么 warmup？」→ 首 launch 含模块加载/JIT 编译/clock 爬升；还有 NVJIT 缓存。
2. 「Event 和 synchronize 分别干什么？」→ Event 在 GPU 时间线打点（device 侧时基），
   synchronize 让 host 等流干（否则 elapsed_time 读到未完成的时间）。
3. 「`CUDA_LAUNCH_BLOCKING=1` 是什么？」→ 强制 launch 同步化，**调试定位用**
   （让错误出现在它发生的那次调用），会拖垮性能，生产禁用。
4. 「测出来的时间要不要含 H2D 拷贝？」→ 看问题定义：端到端延迟含，kernel 峰值性能不含；
   主动说「我会分开报三个数：H2D、kernel、E2D」是强加分。

### 8.3 数字卡片（随口报出 = 硬实力信号）

- kernel launch 固定开销：**个位数 µs**（L004 空 kernel 实测约 4~6µs）
- 本机（RTX 5000/sm_75）：带宽 **448 GB/s**，FP32 **11.2 TFLOPS**，FP16 TC **89.2 TFLOPS**，
  PCIe 3.0 H2D **≈12 GB/s**，L2 **4 MB**，48 SM
- 换算感：1 KB 数据走 HBM ≈ 几 ns 量级；同样数据走 PCIe ≈ 数百 ns——「拷贝比算贵」

### 8.4 红线清单（说了就扣分）

- ❌「用 `time.time()` 包一下就行」——忽略异步
- ❌ 不 warmup 直接报首轮时间
- ❌ 把 ncu 当第一入口（应先 nsys 缩小范围）
- ❌ 报优化数字不说测量方法（warmup 几轮、几次取均值、含不含拷贝）

### 8.5 60 秒电梯陈述（本讲版）

> 「我测量 GPU 代码有一套固定纪律：Event 打点、warmup、synchronize、分开报拷贝和
> kernel 时间；定位问题走 nsys→ncu 的漏斗；选型上 torch.compile 优先、热点下探
> Triton、极限才 CUDA——因为我清楚 launch 开销在 µs 级，小 kernel 融合进大 kernel
> 比优化它本身收益更大。」
