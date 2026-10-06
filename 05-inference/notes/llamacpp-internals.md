# llama.cpp 内部机制拆解（AR005 F4）

> 证据分级：**[source]** = llama.cpp 源码（本仓库 05-inference/llama.cpp 克隆，附文件:行号）；
> **[local machine]** = 本 AR gguf-lab 真机实测（results/*.json，RTX 5000 sm_75）。
> 本 AR 对上游零改动、未编译（无 MSVC/nvcc）——所有 [source] 结论来自读码，所有行为数据来自我们自己的复刻实现。
> 我们复刻了什么/没复刻什么见 §7。

## 1. ggml 计算图模型

- **[source]** ggml 是张量图 IR：`ggml_tensor`（src/ggml.h，含 type/ne[]/nb[]/op/src[]/backend），
  `ggml_graph_build`→`ggml_graph_compute` 驱动；每个 op 有 `GGML_OP_*` 分派
  （ggml.c L7104 `case GGML_OP_ROPE` 所在的巨型 switch）。
- **[source]** backend 化：CUDA 后端实现 `ggml_backend_cuda`，对每个 op 检查
  `supports_op` 后在 device buffer 上执行；下文的 mul_mat 五级阶梯就是
  `ggml_cuda_mul_mat`（ggml-cuda.cu L1908-1927）内的 if-return 链。
- **[local machine]** 我们的复刻用 PyTorch eager 前向替代 graph executor——语义
  等价（同样的算子序列），执行模型不同（torch 自己的调度）。差异清单见 §7。
- 面试要点：**graph executor 让「算子融合/选路」成为可能**——ggml-cuda.cu 里
  能看到 mul_mat 与后续 GLU 节点的融合机会检测
  （`ggml_cuda_should_fuse_mul_mat_vec_q`，L1797-1823；阶梯主体在 L1875 起
  的 `ggml_cuda_mul_mat`，本文引 L1908-1927 为核心段），这是图执行器相对
  eager 的核心收益。

## 2. 量化 GEMM 的路线选择：mmvq vs mmq（+ 完整阶梯）

**[source]** `ggml_cuda_mul_mat`（ggml-cuda.cu L1908-1927）的选路阶梯：

| 顺序 | 路线 | 条件（源码语义） | 适配场景 |
|------|------|------------------|----------|
| 1 | `mul_mat_vec_f` | batch=1 且非量化 | 单步 decode，fp16/f32 权重 |
| 2 | `mul_mat_f`（mmf） | `ggml_cuda_should_use_mmf(...)`：fp16/bf16 类型的分块 GEMM | prefill/大 batch，非量化 |
| 3 | **mmvq** | `src0 量化 && src1/dst 为 f32 && ne11 ≤ MMVQ_MAX_BATCH_SIZE`（L1806-1807）；按类型/架构还有更细的 cap 表 | **小 batch 量化**（decode） |
| 4 | **mmq** | `ggml_cuda_should_use_mmq(...)`：量化 && batch 较大 | **prefill/大 batch 量化** |
| 5 | cuBLAS | 兜底 | 其余 |

- **[source]** `MMVQ_MAX_BATCH_SIZE = 8`（mmvq.cuh L3）；MUL_MAT_ID 场景按
  quant 类型×架构有更严的 cap（mmvq.cu L150-175
  `get_mmvq_mmid_max_batch_pascal_older`、L177-188
  `get_mmvq_mmid_max_batch_turing_plus`）：Pascal 及以前 Q4_K 只到 5；
  Turing+ 表里 Q4_K 走 default=8（该表中只有 Q3_K 收紧到 5）
  ——**[local machine]** 本机正是 Turing(sm_75)，这张表直接适用于本机。
- **[source]** 两条路线的本质区别：
  - **mmvq**（mmvq.cu）：不转换格式，每个 thread block 直接从**packed 量化块
    现场反量化并与激活向量点积**——权重只读一遍、无需中间缓冲，但每 block
    只服务 ≤8 个激活行，算术强度低，只在小 batch 下划算。
  - **mmq**（mmq.cu / mmq-instance-q4_k.cu）：先把**激活也量化**成 Q8_1 tile
    写入共享内存/全局中间缓冲，然后做 tile 化的整数点积，块 scale 乘加在
    体外完成——为大 batch 摊销打包成本，本质是「W4A8 风格」的 tile GEMM。
- **[local machine]** AR005 E3 实测完整解释了为什么必须分两条路线：
  「每步现反量化成 fp16 再 GEMM」的朴素路线比融合慢 9.2×（123.3→13.4
  t/s，dequant 66.9 ms/步主导）；即便用 triton 单 kernel 融合的 dequant
  （1.9 ms/30 张量）仍掉 16%（104.1 t/s）——**正确解法是像 mmvq 一样把反量化内联进
  点积循环，永远不物化 fp16 权重**。
- **[local machine]** E4 的 batch 曲线（时间恒定、吞吐线性 ×63.6）从另一个
  方向印证：小 batch 时瓶颈是权重读取/launch 次数而非算力，所以 mmvq 的
  「省一遍中间缓冲」在小 batch 是净赢；batch 大了算力开始值钱，mmq 的
  tile 化整数路径才反超。

## 3. k-quants 块布局（与 E1 自实现互证）

**[source]** `block_q4_K`（ggml-common.h L327-338）：

```
struct block_q4_K {       // 144 bytes = 4.5 bits/weight, 256 weights
    ggml_half d, dmin;    // super-block scale（各 2B）
    uint8_t scales[12];   // 8 个子块的 (scale, min) 各 6-bit 打包
    uint8_t qs[128];      // 256 个 4-bit nibble
};
```

- 256 元素 super-block → 8 个 32 元素子块；子块是仿射量化
  `x ≈ d·sc·q − dmin·m`，q∈[0,15]（ggml-quants.c L1530-1552
  `dequantize_row_q4_K`）。
- **[source]** 6-bit 打包位序（`get_scale_min_k4`，L880-888）：
  j<4 时 `sc = byte[j]&63`、`m = byte[j+4]&63`；j≥4 时 sc 的低 4 位在
  byte[j+4]、高 2 位在 byte[j−4] 的顶 2 位；m 的低 4 位在 byte[j+4] 高半、
  高 2 位在 byte[j] 顶 2 位。
- **[source]** nibble 交错：byte = 偶子块(低 4 位) | 奇子块(高 4 位)，
  每 32 字节一组对应一对子块（L1542-1549）。
- **[source]** 量化器不是简单 min-max：`make_qkx2_quants`（L799-878）在
  21 个 (rmin + 0.1·is) 的 scale 假设下做加权最小二乘搜索 scale/min，
  权重 `w = av_x + |x|`（L1473-1476）；`nearest_int`（L621-626）用
  2^23+2^22 魔数加法实现 round-half-even——与 `torch.round` 语义一致
  **[local machine]**（E1 移植时验证）。
- **[local machine]** E1 自实现逐行移植了上述全部细节并过门：
  6-bit 打包/解包**精确往返**；块大小 144B/34B 对账精确；Q4_K rel-err
  2.52e-2（高斯权重，~4.5 bit 有效精度的典型误差）；Q8_0 的 d 存 fp16
  会引入额外 2^-11 相对误差（dequant 用 fp16(d)）。
- **[source]** norm 类张量（attn_norm/ffn_norm/output_norm）从不量化——
  `quantize_tensor` 类工具与 convert 脚本对 1D 张量跳过 **[local machine]**
  （我们的 GGUF 导出遵循同一惯例）。
- 为什么 k-quants 比 Q4_0 好：子块独立 scale/min 捕捉**块内分布异质性**；
  6-bit 二级量化让 scale 自身的误差远小于 4-bit 数据的量化步。
  **[local machine]** E3：Q4_K Δppl +0.98 vs 理想均匀 4-bit 的预期损失更小
  （无 Q4_0 对照实现，此对比来自文献共识，见 results.md §E3 谨慎表述）。

## 4. KV cache 与 GQA

**[source]** `llama_kv_cache`（llama-kv-cache.h L20+，llama-kv-cache.cpp）：

- KV cell 是 **ring buffer**：`kv_size` 个 cell，`v_heads` 记录「从哪里开始
  找空闲 slot」（L301-303 注释明说这只是加速 find_slot 的提示、不属于 KV
  状态）；`find_slot(ubatch, cont)`（L898）为每个 ubatch 找一段连续或
  分散的 cell 区间（`slot_info` 含 ssrc/sdst 映射）。
- 每 cell 记录 `pos` 与所属 `seq_id` 集合，支持多序列共享 cache 与
  状态导出/导入（state_read_meta/state_read_data，L353-356）。
- **[source]** GQA 的实现位置不在 cache 而在图构建：K/V 投影输出
  `n_kv_head × head_dim`，cache 只存这么多；attention 构建时对 K/V 做
  `rep = n_head/n_kv_head` 的 repeat（llama-graph.cpp 中
  `build_attn` 的 cq/ck 处理）。
- **[local machine]** E4 实测对账公式：
  `bytes = 2(K和V) × layers × kv_heads × seq × head_dim × dtype_bytes × batch`
  ——GQA 的折扣因子就是 kv_heads/head_count（本模型 2/4）。8 个
  (batch, ctx) 组合实测 vs 公式最大误差 **0.00%**。
- **[source]** 现代 llama.cpp 还有 paged/分层的变体文件
  （llama-kv-cache-iswa/msa/dsa/dsv4.cpp——滑动窗口/多流注意力的
  cache 布局变体），本 AR 未深入，只做存在性标注。

## 5. RoPE

**[source]** `ggml_rope_ext`（ggml.c L4398-4416）：参数
`(ctx, a, b(pos), c(freq_factors), n_dims, mode, n_ctx_orig, freq_base,
freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)`——
b 是每 token 的位置向量；YaRN/mscale 等扩展经 ggml_rope_impl 的
v_vec/sections 机制进入（rope_ext 自身不带 sections 参数）。
- 频率 `1/(theta^(2i/d))`、q/k 按旋转矩阵逐维旋转——标准实现
  在 ggml-cpu/ops.cpp / CUDA 的 rope kernel 中。
- **[local machine]** 我们的复刻用标准 RoPE（theta=10000，cos/sin 作用于
  偶/奇维对）；KV-cache 一致性门（增量解码 vs 全量重算 logits
  maxdiff：prefill 0.0、decode 步 1.72e-5，cache_check.json）证明位置编码与 cache 的交互实现正确——这是
  RoPE 实现最常见的 bug 来源（位置偏移/维度配对错误），门能抓住它们。

## 6. 采样链（llama-sampler.cpp）

- **[source]** 新版采样器是**链式组合**（`llama_sampler_chain_add`），
  top-k/top-p/min_p/typical/xtc…各是独立 backend；且每个 backend 有
  **CPU 实现**（`apply`）和 **GGML 图实现**（`backend_init`/`backend_apply`）
  两套（L1440-1737）——GPU 图路径用 ggml 算子拼出 top-p：
  sort → softmax → cdf → `mask = cdf < p` → **`log(mask)` 作为 bias 加回
  logits**（L1666-1699，log(0)=−inf 实现截断）。
- **[source]** top-p CPU 实现的工程细节（L1549-1602）：p≥1 直接跳过；
  候选 >1024 时**先只 partial sort 前 256 个**（L1564-1567），累加不够再
  全排序（L1588-1592）——对 32k vocab 省 sort 是实打实的收益。
- **[source]** 前缀语义：`cum_sum ≥ p` 时**保留当前 token**（`last_idx = i+1`，
  L1582-1584）+ `min_keep` 保底——与「最小前缀集使 cumsum ≥ p」一致。
- **[local machine]** E5 自实现同一语义并过构造分布门（p=0.6→2 token、
  p=0.9→4 token）；χ² 分布一致性 p=0.406（vs torch.multinomial 的
  p=0.750，N=10k）。**[local machine]** E5 还实测了 llama.cpp 那个
  「先 top-256 再全排」启发式的动机：32k vocab 下 top-p（含 sort）比
  greedy 贵 62.5×（2.130 vs 0.034 ms，batch 32）。
- **[source]** 温度在 `llama_sampler_init_temp`（logits/T）与 dist 实现；
  **[local machine]** E5：温度把 top-p 0.9 的平均截断集从 1.7（T=0.5）
  膨胀到 9.2（T=1.5）——分布越平，「装下 90% 概率质量」需要的 token 越多。

## 7. 复刻范围声明（诚实清单）

| 组件 | llama.cpp | 本 AR 复刻 | 差异 |
|------|-----------|-----------|------|
| 量化格式 Q8_0/Q4_K | C 参考（ggml-quants.c） | Python/torch 逐行向量化移植 + triton fused kernel | 浮点求和顺序不同 → 量化输出可能差 ±1 级（罕见）；块布局/位序/反量化公式精确一致；无法跑 C 版对拍（无编译器），一致性靠逐行读码 + 数学验证 |
| GGUF 容器 | gguf-py（Python） | 直接使用（sys.path 导入，零改动） | 无 |
| 前向执行 | ggml graph executor + CUDA/CPU backend | PyTorch eager | 无图融合、无自定义 kernel 选路 |
| 量化 GEMM | mmvq/mmq/mmf/cuBLAS 五级阶梯 | torch matmul（fp16 权重）+ 单独的 triton dequant kernel | 未实现 fused 量化 GEMM；用 E3 的分解实验证明其必要性 |
| KV cache | ring buffer + slot 管理 + 状态序列化 | 预分配整段 (B, kvh, ctx_max, hd) | 无 slot 复用/多序列；E4 只验证显存公式 |
| 采样 | 链式 sampler、CPU+GPU 图双实现 | torch sort/cumsum 单实现 | 无 chain 抽象；语义与 CPU 版对齐并验证 |
| RoPE | ggml_rope_ext（含 YaRN 等扩展） | 标准 RoPE | 无扩展支持 |

**未验证行为**（只有 [source] 读码结论、无 [本机] 实测）：graph executor 的
实际融合收益、mmvq/mmq 的真实切换点、ring buffer 碎片化行为——需要编译
运行 llama.cpp（本机无 MSVC/nvcc，AR 范围外）。
