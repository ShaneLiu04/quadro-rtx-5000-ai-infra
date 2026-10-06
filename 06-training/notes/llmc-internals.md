# llm.c 训练内部机制拆解（AR006 F4）

> 证据分级：**[源码]** = llm.c 源码（本仓库 06-training/llm.c 克隆，附文件:行号）；
> **[本机]** = 本 AR nanogpt-lab 真机实测（results/*.json，Quadro RTX 5000，sm_75）。
> 本 AR 对上游零改动、未编译（无 MSVC/nvcc）——所有 [源码] 结论来自读码，
> 所有 PyTorch eager 行为数据来自我们自己的 gpt_lab 实验（E1-E7）。
> 本文的主线问题：**llm.c 相对 PyTorch eager 到底省了什么？**
> 答案的证据一半在源码里，一半在我们的 E5 profiler 数据里。

## 0. 一句话总览

- **[源码]** llm.c 的哲学与 llama.cpp 相反：llama.cpp 连 GEMM 都自研
  （量化 GEMM 是其核心资产），而 llm.c 的 GEMM 完全交给 **cuBLASLt**
  （matmul.cuh L109-228 就是一个 cuBLASLt wrapper），自己手写的全部是
  **GEMM 之外的东西**——loss/optimizer/layernorm/encoder/residual。
- **[本机]** E5 profiler 恰好给出了这么做的量化理由：eager 一步训练里
  linear/matmul 只占 **27%**（其中 mm 25.7%），其余 **~73%** 的时间
  在 elementwise/copy 11.3%、attention 9.9%、layernorm 9.5%、
  eager 调度开销 9.1%（20 步内 4900 次 transpose 调用）、
  activation 5.9%、optimizer 2.4% ……——**llm.c 手写的正是这 73%**，
  至于 GEMM 本身，cuBLASLt 已经足够好，不重造。

## 1. matmul.cuh：GEMM 全部走 cuBLASLt，但把「邻居」塞进 epilogue

- **[源码]** `matmul_cublaslt`（matmul.cuh L109-228）是全文件核心：
  创建 `cublasLtMatmulDesc_t` → 布局 → epilogue →
  `cublasLtMatmulAlgoGetHeuristic` 选算法（L205-206，注释说明结果
  会被内部缓存，CPU 开销可忽略）→ `cublasLtMatmul`（L216-218）。
- **[源码]** 融合点全在 **epilogue**（L174-198）：
  - `CUBLASLT_EPILOGUE_BIAS` / `CUBLASLT_EPILOGUE_BGRADB`（L187）——
    bias 前向与 bias 梯度都由 GEMM 尾部完成；
  - `CUBLASLT_EPILOGUE_GELU_AUX_BIAS` / `DGELU`（L182-185）——GELU
    也可融进 GEMM，但 **默认关掉**：train_gpt2.cu L350
    `model->gelu_fusion = 0`（注释：仅 H100+ 考虑开，Ada/Ampere 上
    cuBLASLt 融合 GELU 反而慢）。L235-241 的 wrapper 按这个开关
    决定「GEMM 后单独跑 gelu_forward」还是「epilogue 一体」。
- **[源码]** 梯度累加语义是 llm.c 的反复出现的主题：**backward 里
  dweight 用 +=**（L287-289 注释，`accumulate=true` 即 beta=1.0），
  dinp 用 =（L278-280）。这样 grad accumulation 与单步 backward
  共享同一条代码路径，无需额外加法 kernel。
- **[源码]** bias 梯度（dout 沿 B*T 求和）不走 cuBLASLt 而是自研
  `matmul_backward_bias_kernel9`（L16-81），并有两条路径（L263-274）：
  若 OC 足够大使得 grid 的 y 维为 1，直接 `+=` 写回 dbias；
  否则各 block 写入 `dbias_buffer` 临时区，再由
  `reduce_add_sum_kernel`（L83-102）归约——**「跨 block 归约 vs
  原子 vs 临时缓冲」的标准三选一**，这里选择了确定性的临时缓冲。
- **[本机]** 对照 E5：eager 里 bias/elementwise 归类共 11.3%，这些
  在 llm.c 里要么消失在 epilogue 里，要么被 kernel9 一次性吃掉。

## 2. fused_classifier.cuh：softmax logits 永不物化 + 反向就地启动

llm.c 最有代表性的训练态 kernel（文件头注释 L2-5 自述）：
「forwards the cross entropy loss；never materializes the full
normalized logits；（fusion）also kicks off the backward pass」。

- **[源码]** 每个线程块处理**一行 logits**（一个 token），只归约出
  softmax 的两个标量参数 `SoftmaxParams{1/sum, max}`（L14-63，
  `prepare_softmax_blockwide3`）——**全概率分布从不写回显存**。
- **[源码]** loss 由 0 号线程单点计算（L83-86）：
  `prob = exp(logits[idx,ix] - max) * scale`，`losses[idx] -= log(prob)`。
- **[源码]** 教科书级注释 L88-92：如果不加 `__syncthreads()`，
  用于算 loss 的 logits 会被并发地（race）改写成梯度
  ——梯度值域 [-1,1] 而 softmax offset 可能 < -90，
  `exp^(90+)` 直接出 inf。**「融合前后向」引入的竞态与其修法**，
  这是单独读 PyTorch 代码永远看不到的一类工程问题。
- **[源码]** 反向不产生新缓冲：**logits 原数组被就地改写成 dlogits**
  （L96-117），`store128cs`（streaming store，L110-112）刻意降低
  L2 持久性，把 cache 留给刚读过的原始 logits；block 顺序反向
  （L76 `gridDim.x - (blockIdx.x+1)`）以提高 matmul 数据的 cache 命中。
- **[源码]** 调用点（train_gpt2.cu）：
  - 训练 L822：`fused_classifier(..., True, main_stream)`，且
    `dloss = 1/(B*T*grad_accum_steps)`（L819）——**梯度累积的
    1/grad_accum 缩放直接融进 dloss 标量**，不需要额外 unscale；
  - 验证 L778：`WriteDLogits=False`，只算 loss 不写梯度。
- **[本机]** 对照：PyTorch eager 的 `F.cross_entropy` 物化完整
  logits  （我们的实验 vocab=124 太小看不出代价，但 GPT-2 124M
  口径：B=4、T=1024、V=50257、fp16，logits 一份 ≈ 412 MB
  （4×1024×50257×2B），前向读 + 反向读改写至少两次过显存）。fused_classifier 把
  这两趟显存交通全部变成 L2 内的第二次读（L98-100 注释自己
  就在数这是「2nd read of logits」）。

## 3. adamw.cuh：一个 kernel 更新全模型（含 GradScaler 与随机舍入）

- **[源码]** `adamw_update` 设备函数（adamw.cuh L18-47）单线程处理
  单参数：读 grad/m/v → 更新一二阶矩（lerp，L14-16 用两次 FMA
  实现，注释引 NVIDIA blog）→ bias correction → 参数更新 →
  **stochastic rounding 写回低精度参数**（L43）。
- **[源码]** fp16 训练的完整方案全在这一个 kernel 里：
  - `master_params_memory` fp32 主副本（L38，有则从它读旧参数，
    L46 写回）——对应 PyTorch 的 fp32 master weights；
  - `grad_scale`（L26 乘进梯度）——对应 GradScaler 的
    `scale/unscale_`，只是 llm.c 把它变成了 kernel 的一个标量参数；
  - stochastic rounding + **每张量确定性种子**
    （train_gpt2.cu L1064 `seed = random_u32(&model->rng_state)`，
    rng_state 初始化于 L346 `13371337 + process_rank`）——
    低精度参数更新**可复现**。
- **[源码]** weight decay 只作用于 2D 张量（train_gpt2.cu L1080，
    注释 L1076-1079：embedding 也 decay 是 OK 的，因为 token
    embedding 与输出投影 weight-tied、position embedding 每步都
    参与前向）——与 nanoGPT 的 `configure_optimizers` 按
    dim>=2 过滤同一思想，但放在调用侧而非优化器里。
- **[源码]** launch 结构：`adamw_kernel3<<<dim3(num_blocks,
  num_slices), 512>>>`（adamw.cuh L83），**num_slices = 层数**——
  所有层的同种参数张量用 grid.y 打包，一次 launch 更新完整个
  模型（L1062 循环 14 种参数张量 → 共 14 次 launch，而非
  数百个小 kernel）。最后 `cudaDeviceSynchronize`（L1123）。
- **[本机]** 对照 E5：我们用的 PyTorch fused AdamW 也只占 2.4%
  （fused 已经把逐元素循环压掉了），但 eager 仍要为每个参数
  张量过一遍 Python 级参数遍历；llm.c 的 14-launch + 层维 grid
  是同一方向的更激进版本。**真正的收益不在 optimizer 本身**
  （2.4% 的天花板很低），而在于它是「dloss 融合 + 随机舍入 +
  master weights」整套 fp16 训练方案的落点。

## 4. layernorm.cuh：residual 融合 + 「backward 一律 +=」

- **[源码]** 文件头注释（L1-9）定调：LayerNorm 和 Residual 有时
  融在一起；**所有参数的 backward 都用 `+=`**，因此
  gradient accumulation（微步间累加）与单步 backward 在 kernel
  层面不可区分——只需要在微步 0 时清零
  （train_gpt2.cu L796-802：`micro_step == 0` 时 memset
  losses 和 grads_memory）。
- **[源码]** 融合路径：`fused_residual_forward_kernel5`（L142 起，
  launcher L467-487）一个 kernel 完成 residual 加法 + LayerNorm，
  并缓存 mean/rstd 给 backward（L214）；不满足条件时退回
  `residual_forward` + `layernorm_forward` 两步（L486-487）。
- **[源码]** backward `layernorm_backward_kernel10`（L234 起）用
  shared memory + atomic flag 做跨 block 部分和归约（L363 注释：
  原子递增一个 kernel 启动前清零的 flag），L355 用
  cache-hint 写 dweight 让下一个 kernel（optimizer）命中 L2。
- **[本机]** 对照 E5：eager 里 layernorm 9.5% + 大部分
  elementwise/copy 11.3%（残差加法、copy）正是这两个融合
  对象；4900 次 transpose/20 步的调度开销里也有相当部分
  来自 LN 前后的形状搬运。

## 5. encoder.cuh：训练侧的「确定性」专题

embedding 看似最简单（查表 + 相加），backward 却是**非确定性
的重灾区**：同一 token 在 batch 里出现 k 次，其梯度要累加 k 份。

- **[源码]** forward 无惊喜：`encoder_forward_kernel3`（L19-44）
  逐元素 `wte[ix] + wpe[t]`，x128 向量化 load/store。
- **[源码]** wpe backward 的注释（L125-126）直接承认历史：
  旧版用 `atomicAdd` 聚合 batch 维，**非确定**；现版把划分改成
  「每个 (t,c) 元素由一个线程独占，循环累加全部 B」——每元素
  只写一次，**完全确定**（L119-152）。
- **[源码]** wte backward 更精巧（L46-117 + launcher L169-233）：
  CPU 侧把 batch 内 tokens 按词元分桶并**按桶大小降序排序**
  （L199-206 注释：大桶先跑，否则它们会拖到全 GPU 空转时），
  GPU 侧每桶一个 block 聚合，stochastic rounding 写回低精度
  dwte，种子对每参数唯一（L114 注释：determinism AND 规避
  SquirrelNoise5 参数溢出的 UB 风险）。
- 教学要点：**「训练可复现」不只是设个种子**——低精度参数写回
  用随机舍入时，必须保证随机数流本身确定（每张量唯一 seed）；
  而 atomicAdd 的硬件调度顺序是非确定的，必须用划分/归约
  结构消除。这与推理侧的确定性（AR005 的 KV cache 一致性门）
  是同一主题的两个难度档位。

## 6. mfu.h + gpt2_estimate_mfu：MFU 的两套实现对照

- **[源码]** 公式（train_gpt2.cu L1126-1152）：
  `flops_per_token = 6*N + 6*L*C*T`（L1143），注释（L1136-1138）
  说明第一项是全部权重 matmul（含 weight-tied 的 embedding，
  L1130-1135），第二项是 attention matmul，「通常是小项」。
- **[本机]** 我们的 gpt_lab.py L55-60 用 PaLM/nanoGPT 口径
  `6N + 12*L*H*Q*T = 6N + 12*L*C*T`——**attention 项恰为
  llm.c 的 2×**（差值即 causal mask 的 1/2 因子算不算进平均
  序列长度；llm.c 源码注释未言明，此处按公式差异如实记录）。
  对 GPT-2 124M 口径该差异约 ~13%（113M vs 57M flops/token 的
  attention 项），属于 MFU 这类估计量的固有粗糙度内。
- **[源码]** 峰值 FLOPs 的求法（mfu.h）：白皮书 PerfData 表
  （L42-46：VOLTA/AMPERE/HOPPER/ADA）+ **线性缩放模型**
  （L146 `adjusted = value * (new_cores/CORES) * (new_mhz/CLOCK)`，
  L111-112 用 4080 验证线性假设成立）；GPU 按 `deviceProp.name`
  字符串查 `gpu_db`（L56-95）。
- **[源码 + 本机]** 关键事实：`gpu_db` 里**没有任何 Turing 卡**
  （只有 V100、A 系列、RTX 30/40 系、H100）——本机 Quadro
  RTX 5000 上 `get_flops_promised` 返回 -1，`gpt2_estimate_mfu`
  走 L1148-1150 返回 -1「don't know」。这正是我们 lab 不抄
  上游 estimate_mfu（nanoGPT 那边还硬编码 A100 312 TF）、
  改为**用 CUDA device 属性自算峰值**（fp16 TC 89.2 TF /
  fp32 11.2 TF，按精度选口径）的原因——不是风格偏好，是
  上游实现覆盖不到 sm_75。
- **[本机]** E3 的 MFU 数据（7.5%→17.1%→23.7%→29.9% 随参数量
  单调升）同时是对这套公式的一次实测：小模型在 16GB Turing
  上是 launch-bound，MFU 天花板远低于大模型在数据中心卡上的
  经验值（A100 上 llm.c README 宣称 ~35%+），量纲一致、
  绝对值按硬件缩小。

## 7. llm.c vs PyTorch eager：融合清单（对照 E5 profiler）

**[本机]** E5（10.67M 模型、fp16、20 步 profiler，fresh run
301.7k tok/s 与 E1 的 300.2k 在 ±10% 内）给出 eager 一步的
时间构成；右两列是 llm.c 的对应手法：

| E5 类别 | 占比 | llm.c 对应手法 | 源码依据 |
|---|---|---|---|
| linear/matmul（mm 25.7%，75 次 mm/步） | 27% | 保留 cuBLASLt，但 bias/GELU/bias-grad 融进 epilogue | matmul.cuh L174-198 |
| elementwise/copy（残差加、copy、bias） | 11.3% | residual+LN 融合；bias 进 epilogue；dinp/dweight 的 += 累加 | layernorm.cuh L142；matmul.cuh L187、L287-289 |
| attention（softmax/permute/SDPA） | 9.9% | cuDNN attention（fp32 WIP 路径在手写 attention.cuh，含 online softmax + causal 只算下三角） | attention.cuh L85-99 |
| layernorm | 9.5% | fused_residual_forward_kernel5 一体化，mean/rstd 缓存复用 | layernorm.cuh L142-214 |
| eager 开销（20 步 4900 次 transpose 等） | 9.1% | 无 Python/autograd 层，C++ 主循环直接 launch；输出缓冲复用为 scratch（train_gpt2.cu L830-832） | train_gpt2.cu L827-832 |
| activation（GELU 前后向） | 5.9% | epilogue GELU（H100+ 默认开）；否则独立 kernel | matmul.cuh L182-185、L235-241 |
| loss/softmax（CE） | ~2% | fused_classifier：不物化 probs、dlogits 就地、反向就地启动 | fused_classifier.cuh 全文件 |
| optimizer（fused AdamW） | 2.4% | 单 kernel + 层维 grid，14 张量 = 14 launch；grad_scale/master/随机舍入一体 | adamw.cuh L18-88 |

（E5 归类占比合计 ~77%，其余为 dropout 采样已关、embedding 等
长尾；「~73% 非 GEMM」按 1−27% 口径。）

**结论**：llm.c 的性能故事不是「我写了更快的 GEMM」（GEMM 根本
是 cuBLASLt），而是**消灭 kernel 间的一切中间物化与调度空隙**
——把 CE 反向启动融进 loss kernel、把 unscale 融进 dloss、
把 grad accumulation 融进 +=、把 bias 融进 epilogue、把层维
融进 grid。E5 里那 9.1% 的纯调度开销和 11.3% 的 elementwise
是 eager 架构固有的，只能靠这种「整步手写」拿回来。

## 8. 边界声明（我们没做什么）

- 未编译、未运行任何 llm.c 代码（无 MSVC/nvcc）——所有 [源码]
  结论为读码所得，无本机行为验证；
- cudnn_att.cpp（cuDNN fused attention 的 C++ 侧）与 zero.cuh
  （multi-GPU ZeRO）只在总览层面提及，未逐行拆解；
- attention.cuh 的手写路径（fp32-only WIP，文件头 L13 自注）
  与生产 SDPA 路径的关系未实测；
- 我们 lab 的 SDPA 实验用的是 PyTorch 原生 `F.scaled_dot_product_attention`
  （E7：本机 sm_75 实际命中 mem-efficient backend，flash 需
  sm_80+），不是 llm.c 的任一路径——两者的对照是架构性的，
  不是同 kernel 的 A/B。
