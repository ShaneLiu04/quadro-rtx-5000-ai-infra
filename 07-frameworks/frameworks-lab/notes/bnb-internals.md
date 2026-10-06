# bitsandbytes 内部机制拆解（AR007 F4 笔记）

> 分级标注：**[源码]** = 读克隆源码得出（附 文件:行号）；**[实测]** = 本机真机实验数据
> （Quadro RTX 5000 sm_75 / torch 2.5.1+cu121 / bnb 0.50.3.dev0 克隆，降级导入）。
> 上游零改动；本笔记所有行号对应当前 main 分支克隆。

## 0. 一句话总结

bnb 的本质是「**在 torch 的 op 分发层做精度/硬件双维度特化**」：每个量化算子用
`torch.library` 注册多套实现（CUDA 原生 / 纯 torch 兜底 / triton / mps / xpu / hpu），
权重格式与 kernel 布局共同设计（4bit 打包 nibble、int8 TN 布局），优化器状态用
十进制 decade 码本抗下溢。面试一句话：**它是一个 dispatch + layout + codebook
三件套的框架，不是一堆 kernel 的集合**。

## 1. 加载与降级链（cextension.py）

- [源码] `cextension.py:359` `get_native_library()` 按环境变量 `BNB_BACKEND`
  （默认 cuda）找 `libbitsandbytes_cuda121.dll`；找不到则 `:393` 抛
  RuntimeError，被 `ErrorHandlerMockBNBNativeLibrary` 捕获替换 [源码] `:295-304`
  ——**import 不炸，进入降级态**。
- [实测] 本机克隆导入：stderr 打印 load error，`bnb.__version__ = 0.50.3.dev0`，
  `type(lib).__name__ == 'ErrorHandlerMockBNBNativeLibrary'`，`degraded=True`
  （results/e0_env.json）。
- [实测] **早期探针曾观察到 CPU 调用返回 (None,None,None)**——复跑不可复现，
  归因为首次导入时 `__pycache__` 字节码编译期的部分初始化瞬态；受控 E0 运行
  的稳定结论是：**纯 torch default 兜底在 CPU 上正确服务**（见 §2）。

## 2. torch.library 多后端注册（_ops.py + backends/）

- [源码] `_ops.py:6-7`：`register_fake = torch.library.register_fake`、
  `register_kernel = torch.library.register_kernel`——bnb 完全建立在 torch
  官方自定义 op 机制上；`:10-12` `torch.library.define("bitsandbytes::int8_mixed_scaled_mm", ...)`
  定义 schema。
- [源码] `backends/default/ops.py`：**纯 torch 参考实现**——
  - `:142-177` `int8_vectorwise_quant`（行 absmax + 离群列置零）
  - `:180-196` `quantize_blockwise`（中点界 bucketize，"same assumption as CUDA kernel"）
  - `:233+` `quantize_4bit`；`:38-61` `int8_mm_dequant`（**:57 的
    `6.200124e-05` = 1/127² = 1/16129**——int8 双缩放反量化的全部秘密）
  - `:122-139` `int8_linear_matmul` 默认实现 = fp32 模拟
- [实测] dispatch dump（E0）：`register_kernel(op, "default")` 落在
  **DispatchKey::Undefined**，向所有非 CUDA 后端键广播（"default backend kernel
  registered at /dev/null:142"——行号 142 正是 default/ops.py 的
  int8_vectorwise_quant）。**结论：default 注册 = 全后端兜底**，这就是降级态
  CPU 能跑的原因。
- [源码] `backends/cpu/ops.py:36`：cpu kernel 注册被
  `if not isinstance(lib, ErrorHandlerMockBNBNativeLibrary)` 守卫——native 缺失时
  整组 CPU kernel 不注册，只剩 default 兜底；`:19-33` `torch._int_mm` 的 CPU
  路径要求 **torch ≥ 2.6 且 AVX512**（防 VNNI 缺失时 int32 溢出，注释引用
  pytorch/pytorch#136942）。
- [实测] 本机 `torch._int_mm` CUDA **bit-exact**（K=512 逐元素等于 float64 参考，
  int32 精确）；CPU 路径 256³ 也 bit-exact（E0 JSON）。

### 2.5 框架 dispatch 决策树（整合视图，srs §3.5）

一次 `Linear4bit(x)` 的完整分派路径（[源码] 行号链）：

```
Linear4bit.forward                          nn/modules.py:637
└─ bnb.matmul_4bit                          autograd/_functions.py:407
   ├─ CPU 且 packing_format_for_cpu（AVX512BF16 专用打包，eval 时置位）
   │  └─ F.gemv_4bit                        functional.py:1300
   │     └─ torch.ops.bitsandbytes.gemv_4bit
   │        [源码注释 _functions.py:421 "kernel supports any M via tiled
   │         GEMM despite the gemv name"——名字是历史遗留]
   ├─ 需要 grad / 权重 [K,N] 反向放置(:437 警告) / nested 且 state2.blocksize≠256(:485)
   │  └─ dequantize_4bit + F.linear         安全回退路径
   └─ 标准 4bit 前向（无 grad）
      └─ torch.ops.bitsandbytes.gemm_4bit.default
         ├─ torch.library 层：按设备 DispatchKey 分派（§2 的 dump：CUDA
         │  kernel at /dev/null:384；非 CUDA 键 → Undefined 广播 default 兜底）
         └─ csrc/gemm_4bit.cu:41-131 中央 dispatcher（进入 CUDA 后二级分派）
            ├─ bf16 + sm75（无 bf16 TC）→ SIMT   gemm_4bit_simt.cu   [:118]
            ├─ fp16 + sm75                 → mma   gemm_4bit_sm75.cu  (m16n8k8)
            └─ fp16/bf16 + sm80+           → mma   gemm_4bit_sm80.cu  (per-arch tile 表 :490-684)
```

**gemv/gemm 分工**（[源码]）：
- 4bit：主干统一走 `gemm_4bit`（csrc dispatcher 按 arch/dtype 选 kernel）；
  `gemv_4bit` 仅剩 CPU AVX512BF16 packing 一个分支，且注释明说它实际是
  tiled GEMM——"gemv"是初版仅支持 M=1 时的历史命名。
- int8（Linear8bitLt）：`int8_linear_matmul`（functional.py:1536）→
  `igemmlt`（csrc/ops.cu:283）→ `cublasGemmEx`（:225，CUDA_R_8I 输入 /
  CUDA_R_32I 累加 / `CUBLAS_GEMM_DEFAULT_TENSOR_OP`）——**bnb 不写 int8
  GEMM kernel，只钉布局调 cuBLAS**；行向量输入另有 `A.shape[0]==1`
  转置捷径（functional.py:1075）。这与 E2 实测互证：int8 的收益全在
  布局（TN），bnb 把布局做成数据格式而不是用户责任。
- Embedding4bit：`embedding_dim % blocksize == 0` → 按行部分反量化路径
  （modules.py:969 `_forward_with_partial_dequantize`），否则全量反量化
  + `F.embedding`——又一处"形状整除才走快路径"的分派。

## 3. 四大算法地标（functional.py）

| 算法 | 地标 | 关键语义 |
|---|---|---|
| NF4 码本 | `functional.py:170-196`（256 元素表，前 16 = NF4 级）；`:772-806` `get_4bit_type` | **NF4 不是浮点编码**——4bit 索引查表；官方表端点钉 ±1（QLoRA 论文拟合值），**≠ 裸 ndtri 分位数** |
| 4bit 量化入口 | `:884` `quantize_4bit` | blocksize 默认 **64**，合法 {32,64,128,256,512,1024,2048,4096}；compress_statistics → 嵌套双重量化 |
| blockwise 8bit | `:613` `quantize_blockwise` | 通用 blocksize **4096**（注意：优化器状态路径用 256，见 §4） |
| LLM.int8 | `:1590` `int8_double_quant`（threshold 参数 → 稀疏离群分解）；`:1655` `int8_vectorwise_quant`；`:1641` `int8_vectorwise_dequant`；`:1536` `int8_linear_matmul` | threshold>0 时 |x|≥τ 的元素剔除，**含离群的整列**走 fp16 旁路 |
| dynamic 8-bit | `:296-348` `create_dynamic_map` | 十进制 decade 码本：每 decade 10^-7..10^-1 内线性均值，**255 非零级跨 7 个数量级**——优化器状态专用（见 §4） |

- [实测×源码交叉] E1：我们的 `nf4_levels` 与 `get_4bit_type('nf4')` 前 16 值
  **逐值一致**（官方表，e0_env.json）；E3 审查后补验：`dynamic_map(signed)`
  **与** `create_dynamic_map(signed)` **双变体逐值一致**（signed/unsigned
  各 255 非零级——审查曾抓到我们 unsigned 公式 135 级的偏差，已按
  functional.py:320-323 精确公式修正；证据 = verify_numbers.py 运行时
  交叉比对，落 results/doc_check.json）。

## 4. 8-bit 优化器（bnb 最大的工程智慧）

- [源码] `optim/optimizer.py:134-159`：`name2qmap["dynamic"]`（m 用，signed）、
  `["udynamic"]`（v 用，unsigned——v 非负）；`:519-522` 把 qmap 挂进 param state。
- [源码] `backends/cpu/ops.py:469-580` `_optimizer_update_8bit_blockwise_cpu`
  （纯 Python 参考实现，黄金语义源）：
  - `:490` **blocksize = 256**（不是 4096！——块内动态范围紧 16×）
  - `:498/502` 状态经 qmap 查表反量化（非线性 decade 码本）
  - `:507-517` Adam 更新：`denom = (v.sqrt()/c2).add_(eps)`；
    `p -= lr/c1 · m/denom`；`:575-577` 重新量化
- [实测] **为什么必须是 decade 码本**：我们先用线性 int8 复刻（127 级线性 +
  block 4096）→ 训练发散（loss 29→89，w 爆到 ~50）：块内 v 小于 absmax/254
  的坐标 round 到 0 → √v̂=0 → 步长 = lr·m̂/ε ≈ 10⁵ 级 → 爆炸。换成 bnb 语义
  （decade 码本 + block 256）后：与 fp32 Adam 同种子双训 800 步，**终值 loss 差
  三次同种子运行 0.0007/0.0253/0.0207（训练噪声内，门 <0.05 三过）**，
  优化器状态显存 26.09MB→6.62MB（**0.254×**，理论 2.03/8 命中）
  （results/e3_adam8bit.json）。
- [面试讲法] "8-bit Adam 能工作的三个前提：**指数分布码本**（跨数量级表示小
  v）、**小块**（256，块内同质）、**每步 dequant→update→requant 全链**——
  单独把状态 dtype 改成 int8 是不行的，会爆。"

## 5. csrc：4-bit GEMM 的硬件特化（本机 sm_75 主场）

- [源码] `gemm_4bit.cu:41-131` 中央 dispatcher：**SIMT（sm60+）vs MMA
  （sm75 仅 fp16 → sm75 kernel；sm80+ bf16/fp16 → sm80 kernel）**；
  `:118` **bf16 在 sm75 无 TC → 落 SIMT**（与本机 AR006 "Turing 无 bf16 TC"
  完全互证）。
- [源码] `gemm_4bit_sm75.cu`（上游为本机架构写的专用 kernel）：
  - `:1` `mma.sync.aligned.m16n8k8`，**fp16 only**；`:115` `#if __CUDA_ARCH__ == 750`
  - `:26-73` 内联 PTX：mma.sync + `ldmatrix.x2/x1`（smem → 寄存器矩阵装载）
  - `:98-168` 布局：A [M,K] fp16 / **B [N,K/2] packed uint8（2 nibble/byte）** / C fp16；
    K_CHUNK=64 双缓冲 smem（27-54KB/tile）
  - `:152-153` **码本进寄存器**：每 lane 持 16 个 NF4/FP4 LUT 值之一；
    `:219-225` 反量化用 `__shfl_sync` 按 nibble 索引取 LUT 值——**零内存查表**；
    `:218` 注释 "Centroid × scale in fp32 avoids double rounding to half"
  - `:104-107` **嵌套 8-bit absmax + 256 码本 + offset** = QLoRA 双重量化在
    kernel 内原位解压
  - `:338-413` launcher 按占用率选 tile（`:384` "2x more blocks wins at normal
    occupancy on sm75"）
- [源码] `gemm_4bit_sm80.cu:490-684`：per-arch tile 校准表（sm86 检测、sm90
  仅 64x64-64、sm100/103 B200/B300、sm120 按 num_sms≥150 分档、HBM crossover
  启发式）——**框架为每代 GPU 维护经验分派表**，这就是"kernel 之上的决策层"。
- [实测×源码互证] E2 发现 cuBLAS int8 TC 快路径要求 **TN 布局**（rowA×colB
  0.253ms vs row×row 0.464ms，1.83×）；bnb 的 int8 权重布局
  （`int8_vectorwise_quant` 行存 + igemmlt 自定义格式）与 sm75 kernel 的
  B [N,K/2] row-major（= mma B operand col-major ⇔ TN）**都是为了让数据
  落在 TC 快路径上**——权重格式与 kernel 布局共同设计。

## 6. 面试高频问答卡

| 问题 | 回答要点（全部有本机数据支撑） |
|---|---|
| NF4 为什么比 INT4 好？ | 分位匹配 N(0,1) + absmax 归一化把权重压进 [-1,1]，NF4 级在此区间中心密集；实测 rel-RMSE 0.092 vs 0.100（E1）；**官方表不是裸 ndtri**——裸分位数在 absmax 归一化下反而不如 INT4（0.127，E1） |
| QLoRA 双重量化省多少？ | 4bit 权重 0.5 B/param + fp16 块缩放 2/64 = 0.53 B/param（实测 0.5315，E5）；缩放再量化省 ~0.03 B/param；7.5B 底座 4.0GB（实测外推，E5） |
| LLM.int8 为什么需要离群分解？ | 行 absmax 被离群列摧毁：重尾输入实测误差 4.9e-2（per-row 也救不了）；τ=8 分离 8/2048 离群列（0.4%）后误差回到 int8 本底 1.1e-2（4.2×）——分解消掉离群污染，但不会比 int8 更准；τ 要卡在正常 max（~4.5σ）与离群量级（16σ）之间（E2） |
| 8-bit Adam 为什么不炸？ | decade 码本（7 数量级动态范围）+ block 256 + 每步全链 dequant/requant；线性 int8 会因 v 下溢归零爆步长（实测发散 vs bnb 语义收敛差在训练噪声内：三次同种子运行 0.0007/0.0253/0.0207）（E3） |
| int8 推理为什么可能比 fp16 慢？ | 布局决定 TC 路径：cuBLAS int8 默认 NT 落慢路径（实测 0.59×）；TN 布局 1.29×（配对 1.83×）；框架（bnb igemmlt）用自定义权重布局保 TN（E2 + csrc 拆解） |
| bnb 在没有 CUDA 二进制的机器上还能干嘛？ | 降级导入 + default 纯 torch 兜底 CPU 可跑（实测）；torch.library 的 Undefined 键注册向所有后端广播（dispatch dump 实测） |
