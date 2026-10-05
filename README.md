# AI Infra 资料库

面向这台机器整理的一套代码，从 GPU 编程入门到可以改的训练、推理和编译器。全部是 2026-10-05 从 GitHub 浅克隆的快照（`--depth 1`，未拉取子模块），合计约 1.9 GB。

在这台 Windows 环境的 PATH 里没有 `nvidia-smi`。阅读可以在这里进行；编译、跑 kernel、记时，需要在已经装好 Quadro RTX 5000 驱动和 CUDA Toolkit 的系统上做。

个人版 Gitee 的单仓库上限是 500MB，单文件上限是 50MB。推送到 [quadro-rtx-5000-ai-infra](https://gitee.com/liu-xingyan04/quadro-rtx-5000-ai-infra) 时，仓库正文包含源码、笔记、图片和 llama.cpp 自带的词表。PDF 与 PPTX 合计约 570MB，放在同一仓库的 Release 附件里，压缩包内的相对路径与本目录一致。把附件解压到仓库根目录后，笔记里的幻灯片链接会重新对上。附件清单见 `SLIDES-MANIFEST.txt`。

## 这台机器的边界

| 部件 | 规格 | 做研究时意味着什么 |
| --- | --- | --- |
| CPU | Xeon Gold 6234，8 核 16 线程，基频 3.3 GHz | 数据预处理和 CPU 侧参考实现够用。核数少，训练时的数据加载要先放进内存 |
| 内存 | 128 GB | 数据集和 tokenizer 缓存可以常驻内存。权重大到放不进显存时，也可以从内存分段送到 GPU |
| GPU | Quadro RTX 5000，Turing TU104，sm_75 | 48 个 SM，3072 个 CUDA Core，384 个第二代 Tensor Core |
| 显存 | 16 GB GDDR6，带宽 448 GB/s | 工作集要以这 16 GB 为准 |
| 算力 | FP32 约 11.2 TFLOPS；FP16 Tensor Core 约 89.2 TFLOPS | FP32 的算术强度拐点约 25 FLOP/byte，FP16 Tensor Core 约 199 FLOP/byte。大 GEMM 才吃得满 Tensor Core，softmax、归一化、访存型算子先看带宽 |
| 片上存储 | 每 SM 的 shared memory 上限 64 KB，L2 4 MB | kernel 设计按 64 KB shared memory 分块，不要按 Hopper 的 228 KB 来写 |
| 互联 | 到 GPU 是 PCIe 3.0 | 主机到设备实际大约 12 GB/s。一层一层把权重从内存换进显存，解码会被这条总线卡住 |

Turing 的 Tensor Core 做 FP16、INT8、INT4。BF16、FP8、FP4、TMA、WGMMA 属于 Ampere 之后的硬件。官方 FlashAttention 2/3、DeepGEMM、ThunderKittens、当前主线的 vLLM kernel，都默认 sm_80 或更高。这些仓库没有放进本目录。

16 GB 上比较现实的模型尺度：

- 从头训练：GPT-2 small / medium 这一档（约 1.2 亿到 3.5 亿参数）。7B 的 FP16 权重本身约 14 GB，再加梯度和优化器状态放不下。
- 微调：4-bit QLoRA 可以放下 7B，13B 需要短序列、小 batch 和 gradient checkpointing。
- 推理：7B FP16 或 13B 的 4-bit 比较从容。32B 的 4-bit 权重大约 18 GB，已经超过显存。

## 目录

每类 1 到 2 个仓库。提交号是浅克隆时的 `HEAD`。

### 01 入门：建立 GPU 编程手感

| 目录 | 上游 | 提交 | 在这台机器上怎么用 |
| --- | --- | --- | --- |
| `01-foundations/GPU-Puzzles` | [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | `b3c4b23` | Sasha Rush 的交互式练习。用 Numba 写 CUDA，几个小时内从线程索引走到分块 GEMM。任何 sm_75 都能跑 |
| `01-foundations/gpu-mode-lectures` | [gpu-mode/lectures](https://github.com/gpu-mode/lectures) | `77a8df4` | GPU MODE 的讲义。先看 `lecture_001` 到 `lecture_004`，再看 `lecture_014`（Triton）和 `lecture_012`（FlashAttention 的算法）。后面大量讲义针对 Hopper，当作下一档硬件的预习 |

### 02 手写算子：从朴素实现走到 Tensor Core

| 目录 | 上游 | 提交 | 在这台机器上怎么用 |
| --- | --- | --- | --- |
| `02-handwritten-kernels/LeetCUDA` | [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA) | `c5e985f` | 200 多个 kernel，按难度排好。从 `kernels/` 里的 FP32 SGEMM 和 FP16 WMMA HGEMM 做起。仓库里的 FP8、BF16 示例在这张卡上没有对应硬件 |
| `02-handwritten-kernels/how-to-optim-algorithm-in-cuda` | [BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) | `5c85a89` | 同一条路线的中文笔记：CUDA、CuTe、Triton、PTX、PyTorch、推理。有 11 个文件名含冒号，Windows 无法原样检出，已写成冒号替换成 ` -` 的副本，说明在该目录的 `WINDOWS-CHECKOUT.txt` |

### 03 GEMM：工业级线性代数

| 目录 | 上游 | 提交 | 在这台机器上怎么用 |
| --- | --- | --- | --- |
| `03-gemm/cutlass` | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | `0b55a2f` | 本卡对应的例子是 `examples/08_turing_tensorop_gemm` 和 `examples/09_turing_tensorop_conv2dfprop`。单测在 `test/unit/gemm/device/` 下文件名带 `sm75` 的那一组。`examples/` 里带 hopper、blackwell 的目录是给更新架构看的源码 |

这一类只放了 CUTLASS。它同时覆盖 SIMT、Turing WMMA 和后面几代 Tensor Core，没有第二个仓库能替换这个位置。

### 04 Kernel 语言与编译器

| 目录 | 上游 | 提交 | 在这台机器上怎么用 |
| --- | --- | --- | --- |
| `04-kernel-dsl/triton` | [triton-lang/triton](https://github.com/triton-lang/triton) | `fa8415b` | Triton 支持 sm_75。先跑 `python/tutorials` 里的向量加、softmax、matmul。教程里使用 TMA 的新示例需要 Hopper。这个克隆没有带 LLVM 子模块，读代码和跑官方教程够用；要改编译器本身，再单独拉取子模块 |
| `04-kernel-dsl/tvm` | [apache/tvm](https://github.com/apache/tvm) | `a771de8` | 经典编译器栈，调度和代码生成可以指定 sm_75。`3rdparty/` 是空的子模块指针，当前副本用于阅读 Relax、MetaSchedule 和 codegen，不能直接编译 |

### 05 推理系统：16 GB 里把模型跑起来

| 目录 | 上游 | 提交 | 在这台机器上怎么用 |
| --- | --- | --- | --- |
| `05-inference/llama.cpp` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `d89651a` | GGUF 推理，CUDA 后端包含面向 Turing 64 KB shared memory 的 FlashAttention。看 `ggml/src/ggml-cuda/` 里的量化 GEMM 和 attention。7B、13B 量化模型是这张卡的主场 |
| `05-inference/exllamav2` | [turboderp-org/exllamav2](https://github.com/turboderp-org/exllamav2) | `7dc12af` | 面向消费级显卡的 GPTQ / EXL2 推理，Turing 用 xformers 路径。仓库后加的 paged attention 依赖 FlashAttention 2，那条路径要求 sm_80。EXL3 在另一个仓库，不支持 Turing |

两个代码库代表两种权重布局：GGUF 和 EXL2。比较它们的 KV cache、量化分组和 batch 方式，就是这张卡上的推理系统研究。

### 06 训练算法：单卡能做完的实验

| 目录 | 上游 | 提交 | 在这台机器上怎么用 |
| --- | --- | --- | --- |
| `06-training/nanoGPT` | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | `3adf61e` | 一份短的 GPT 训练代码。改深度、注意力、优化器、数据顺序，几个小时能看到 loss。用 FP16，先跑 small 和 medium |
| `06-training/llm.c` | [karpathy/llm.c](https://github.com/karpathy/llm.c) | `f1e2ace` | 同一类模型的纯 C / CUDA 训练。`dev/cuda/` 把每一步访存写开，方便对照 roofline，确认时间花在算力上还是带宽上 |

### 07 框架与显存：改运行时，而不是只调参

| 目录 | 上游 | 提交 | 在这台机器上怎么用 |
| --- | --- | --- | --- |
| `07-frameworks/tinygrad` | [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) | `246ca9a` | 一个能读完的深度学习框架：调度、内存规划、kernel 融合、CUDA 后端都在同一棵树里。改一处就能在 sm_75 上测量 |
| `07-frameworks/bitsandbytes` | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | `8336490` | 8-bit 优化器、LLM.int8()、4-bit QLoRA。官方硬件表把 sm_75 列为推荐档。这是 16 GB 上微调 7B 的主要手段 |

### 08 算子研究的题目和记分板

| 目录 | 上游 | 提交 | 在这台机器上怎么用 |
| --- | --- | --- | --- |
| `08-kernel-research/KernelBench` | [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) | `423217d` | 用 PyTorch 算子当题目，比对自己的 CUDA / Triton 和 `torch.compile`。评测跑在本机 GPU 上，不依赖 Hopper |
| `08-kernel-research/flash-attention-sm75` | [JohnScheuer/flash-attention-sm75](https://github.com/JohnScheuer/flash-attention-sm75) | `e2d4fb2` | 专门给 sm_75 写的 FlashAttention 前向，用 WMMA，支持 head dim 64 和 128。作者给出的测量慢于 PyTorch SDPA，且没有反向。把它做到接近 SDPA，再补上反向，是这张卡上一条完整的算子研究线 |

## 建议顺序

1. 做完 `GPU-Puzzles`，接着看 `gpu-mode-lectures` 的 `lecture_001` 到 `lecture_004`。
2. 在 `LeetCUDA` 里把 FP32 GEMM 从朴素版写到分块、向量化、双缓冲，用 cuBLAS 做对照。再写 FP16 WMMA，对照 `cutlass/examples/08_turing_tensorop_gemm`。
3. 用 Triton 把同一个 matmul 和 softmax 重写一遍，看和手写 CUDA 的差距。
4. 用 `nanoGPT` 跑一个 small GPT，确认 16 GB 上的 batch、序列长度和 FP16 占用。需要看带宽时切到 `llm.c`。
5. 用 `llama.cpp` 跑一个 7B 或 13B 的 GGUF，记下 prefill 和 decode 的 token/s，以及 KV cache 随上下文的增长。
6. 选 `KernelBench` 的一道题，或者以 `flash-attention-sm75` 为起点，用 Nsight Compute 看是算力、带宽还是 shared memory 容量卡住了。

计时工具随 CUDA Toolkit 安装，不在这个目录里：Nsight Systems 看时间线，Nsight Compute 看单个 kernel 的 roofline。PyTorch 自带的 profiler 用来对上框架里的算子名。

## 三条可以同时做的研究

算子。目标是这张卡上的 cuBLAS 和 PyTorch SDPA。FP16 GEMM 的天花板是约 89.2 TFLOPS，访存型算子的天花板是 448 GB/s。`flash-attention-sm75` 目前两项都没碰到，而且只有前向。

算法。在 `nanoGPT` 里改结构或优化器，用 small / medium 把实验做完。要上 7B，走 `bitsandbytes` 的 4-bit QLoRA，基座权重量化，可训练的是 LoRA。优化器状态因此小一个数量级。

架构。推理侧对照 `llama.cpp` 和 `exllamav2` 的权重格式与 KV cache。编译器侧改 `tinygrad` 的调度或内存规划，在同一张卡上 A/B。`tvm` 用来对照工业界的编译器是怎么划分这些层的。

## 没有放进来的项目

这些项目质量很高，和这张卡的硬件对不上，克隆下来也无法作为实验底座。

| 项目 | 原因 |
| --- | --- |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | 当前开发转向 CuTe DSL，作者明确不为 Turing 维护 kernel |
| [vllm-project/vllm](https://github.com/vllm-project/vllm)、[sgl-project/sglang](https://github.com/sgl-project/sglang)、TensorRT-LLM | 服务系统的调度思想值得读论文和文档；仓库里的融合 kernel 按 Ampere 及更新的架构编译 |
| DeepGEMM、FlashMLA、ThunderKittens | FP8、TMA、WGMMA，需要 Hopper 或更新 |
| TorchTitan、Megatron-LM、DeepSpeed 的多卡路径 | 这台机器是单卡，没有 NVLink |
| PyTorch 本体 | 体积大，而且应当作为已经安装好的基线去对比，不需要把源码树放在这里 |
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | 官方 API 示例。入门手感由 GPU-Puzzles 和 GPU MODE 覆盖，需要查某个 CUDA API 时再单独克隆 |

## 更新快照

这些目录是浅克隆。在能访问 GitHub 的机器上，进入对应目录执行 `git pull`。公司网络克隆时使用了临时代理 `http://proxy.ai.yinwang.com:8080`，没有写入 git 配置。

`how-to-optim-algorithm-in-cuda` 再次检出时，Windows 仍会拒绝那 11 个带冒号的文件名。按 `WINDOWS-CHECKOUT.txt` 里的对应关系，用 `git cat-file` 取出 blob 后写入替换过冒号的文件名即可。

Triton 和 TVM 都没有拉取子模块。阅读和跑 Triton 自带教程不需要它们。要在本地编译这两棵源码树时，再按各自 README 初始化子模块，LLVM 那一份会占用几十 GB。
