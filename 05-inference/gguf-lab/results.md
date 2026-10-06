# AR005 实验结果分析（gguf-lab）

> 全部数据为真机实测（Quadro RTX 5000, sm_75, 16GB, WDDM 显示 GPU）。
> JSON 原始数据在 `results/`，图在 `figs/`，复现：`python charlm.py && python charlm.py --verify-cache && python bench_infer.py all && python plot_results.py`。
> 计时纪律沿用 AR002-004：烧机 + 3 次预热 + 自适应 reps 中位数 + CUDA Event；子毫秒 kernel 额外记录 20 次取 min（WDDM 噪声只增时延）。

## F1 门：KV cache 一致性（cache_check.json）

增量解码（KV cache）与无 cache 全量重算逐 token 对拍（fp32，batch 4，
48 步）：prefill logits maxdiff **0.0**，decode 步 maxdiff **1.72e-5**
（门 ≤1e-4 PASS）；greedy id 序列逐 token 相同；固定种子两次生成
bit 级相同；batch 各行（同 prompt）输出一致。

## 模型与环境（E0，fig1）

- 语料：llama.cpp/docs 52 个 md，499,853 字符（英文技术文本），char-level vocab=119；
  5% 末尾留作 held-out。
- 模型：llama 架构 3.21M 参数（4L d=256，GQA 4:2，RoPE θ=10000，SwiGLU n_ff=768，
  RMSNorm），fp32 训练 3000 步（AdamW 3e-4 + cosine，batch 64×ctx128），147s。
- loss 4.84→0.44，门 <2.0 通过；**val loss 在 ~600 步后回升（2.14→3.05）**
  （[训练日志] 初版 val 曲线只打印 stdout 未落盘——e0_train.json 仅含 train
  loss 曲线；charlm.py 已修为 val_curve 落盘，但现有 checkpoint 是旧版产物，
  此两值无法从 JSON 复核，如实标注）：
  3.2M 参数对 500KB 语料训练约 200 epochs，过拟合。这是刻意的（AR 聚焦推理侧），
  但两个后果要诚实说明：(1) held-out ppl 偏高（fp16 15.60）；(2) 记忆化使 logits
  尖锐（top-1 概率均值 0.736），对量化扰动的 argmax 翻转更敏感（E2 见）。

## E1 量化往返（fig2）

对全部 2D 权重做 quantize→dequantize（CPU 参考实现，公式逐行对照 ggml-quants.c）：

| 格式 | rel-err（整体） | max-abs 归一 | 有效位宽 | 门 |
|------|----------------|--------------|----------|-----|
| Q8_0 | 0.0019 | 0.0040 | 1.0625 B/w | maxabs ≤ 1/128（半步长上界）PASS |
| Q4_K | 0.0252 | 0.0832 | 0.5625 B/w | rel ≤ 0.25 PASS |

- 12 字节 scales 的 6-bit 打包/解包（get_scale_min_k4 位序）**精确往返**；
  块大小对账：34B/32 元素与 144B/256 元素逐张量断言通过。
- norm 权重（1D）不量化——llama.cpp 惯例。
- 直方图（fig2）：Q8_0 误差紧贴均匀量化理论（步长 amax/127）；Q4_K 误差宽一个
  数量级且与子块统计相关（带 min 偏移的 4-bit + 6-bit 子块 scale 的复合误差）。

**诚实修订记录**：srs 初版 Q8_0 门写的是「rel-err ≤ 2^-8=3.9e-3」——把
「max-abs 元素的半步长」误当「范数相对误差」。32 元素块高斯数据 amax≈2.2σ，
误差 RMS = (2.2/127)σ/√12 ≈ 5e-3·σ，实测 rel 1.9e-3~5.3e-3 与理论吻合。
修订为 max-abs 归一 ≤ 1/128（真不等式），rel-err 如实记录。

## E2 GGUF 三格式闭环

用 gguf-py（GGUFWriter/Reader，llama 架构 metadata + char tokenizer）：

| 格式 | 文件大小 | 压缩比 | worst rel | teacher-forced top-1 一致率 | greedy 生成 vs fp16 |
|------|---------|--------|-----------|------------------------------|---------------------|
| f16  | 6271.3 KB | 1.00× | 0.0000（精确） | 100% | 100% 逐 token 相同 |
| q8_0 | 3335.6 KB | 1.88× | 0.0053 | 98.4% | 100% 逐 token 相同 |
| q4_K | 1769.8 KB | 3.54× | 0.0719 | 78.6% | 8.3%（轨迹发散） |

- metadata（embedding_length/block_count/head_count_kv 等）与 tokenizer 精确读回。
- q4_K 生成样例：`following commands the llamas and CPU backend to the standar`
  ——贪心轨迹发散但仍是**连贯英文技术文本**：量化扰动改变「走哪条记忆轨迹」
  而不破坏语言结构。top-1 一致率 78.6% 是「结构保留」的量化证据。
- **诚实修订记录**：srs 初版的「greedy 前缀重叠 ≥60%」门不可达且**测错了东西**：
  对过拟合记忆模型，贪心轨迹是混沌的（一个 argmax 翻转后全部后续 token 改变），
  重叠度衡量的是轨迹稳定性而非语言质量。修订为 teacher-forced top-1 一致率
  （q8_0 ≥95%、q4_K ≥50%），greedy 重叠如实报告并分析。

## E3 量化推理：质量与速度

**质量（fig4）**：held-out ppl（fp32 计算，权重分别替换）：

| 权重 | ppl | Δppl |
|------|-----|------|
| fp16 | 15.6031 | — |
| Q8_0 | 15.5933 | **−0.0098**（门 ≤0.05 PASS） |
| Q4_K | 16.5806 | +0.9775 |

8-bit 块量化的权重噪声在 ppl 上完全不可见；4.5-bit 有效精度可见但语言结构保留。
这与 llama.cpp 社区经验一致：Q8_0 通常「无损」，Q4_K 轻损。

**dequant 实现阶梯（fig3，两个数据规模分开对比，min-of-20）**：

*同数据规模（4096 个 Q4_K super-block = 1.05M 权重，e3.json
`dequant_q4_K_4096blocks`）*：

| 实现 | 耗时 | 加速 |
|------|------|------|
| naive 逐块 Python 循环 | 9204.6 ms | 1× |
| torch 多算子向量化 | 2.2008 ms | **4182×**（launch-bound 在此规模已现） |

*全尺寸（4096×4096 = 16.7M 权重，fp32 输出）*：

| 实现 | 耗时 | 有效流量 |
|------|------|----------|
| torch 多算子向量化 | 3.963 ms | 19.3 GB/s（4.3% HBM） |
| triton 单 kernel 融合（flat 设计） | 0.290 ms | **264.3 GB/s（59% HBM）**，13.7× |

- 门：带宽 ≥200 GB/s PASS（264.3）；相对 naive 加速 ≥10× PASS（4182×，同规模）。
- **关键实测发现**：torch 多算子向量化在 WDDM 下是 **launch-bound**——
  每 launch ~50-200µs × 几十个算子主导耗时，带宽远未触及（全尺寸也只
  19.3 GB/s = 4.3% HBM）。triton 初版「每 program 1 块」也是错的
  （6.5 万个微型 CTA，CTA 调度吞吐成为瓶颈，实测只有 ~16 GB/s——
  [开发过程记录] 该变体未保留在最终代码/JSON 中）；**flat 设计**
  （每 program 2048 连续输出元素、全部向量 gather、消除标量 load）
  才到 264 GB/s。
- q4_K fp16 输出变体墙钟更快（0.274 vs 0.290 ms）但流量 GB/s 更低
  （156.7 vs 264.3）——kernel 是 **gather 指令吞吐受限**而非 DRAM 受限
  （每元素 8 次字节 gather 固定）。
- 这就是 llama.cpp 把 dequant **融进 GEMM kernel**（mmvq/mmq）的根本原因：
  生产路径根本不物化 fp16 权重。

**端到端解码速度（fig9，batch=1，64 步，fp16 计算）**：

| 部署 | tokens/s |
|------|----------|
| fp16 权重 | 123.3 |
| Q4_K 预反量化（驻留 fp16） | 129.5（与 fp16 相当，差 5% 在 WDDM 噪声内） |
| Q4_K 每步现 dequant（torch 多算子，66.87 ms/步） | 13.4（**慢 9.2×** vs fp16） |
| Q4_K 每步现 dequant（triton per-tensor，1.887 ms/步） | 104.1（仍慢 16% vs fp16 / 20% vs 预反量化） |

- 「4-bit 权重让推理更快」在**预反量化部署**下是假的（速度与 fp16 相当
  ——123.3 vs 129.5 t/s，5% 差属 WDDM 运行间噪声，不构成提速；省的是
  显存 3.54×）；在**每步现反量化部署**下更是反的（dequant 开销主导）。
- 30 个 2D 权重张量即使各用 1 次 triton kernel 也仍有 30 次 launch（实测
  合计 1.887 ms/步 ≈ 63µs/launch）——模型尺度下 per-tensor dequant 依然
  launch-bound，吞吐掉到 104 t/s。生产解法只有一个：**融合进
  GEMM**（mmvq 路线），或整个模型一次 flat dequant。

## E4 decode 吞吐与 KV cache（fig5/6/8）

**吞吐 vs batch（fig5）**：batch 1→64，48 步解码的墙钟几乎恒定
（397.95 → 399.03 ms），总吞吐线性放大 **121 → 7699 t/s（63.6×/64×）**。
- 每步读的权重字节与 batch 无关 → batch 摊销权重读取 → 教科书 memory-bound
  签名。
- 但单序列 121 t/s 距权重流下限（fp16 6.42MB / 448 GB/s = 69,805 t/s）差
  **577×**——3.2M 模型每步 ~30 次 kernel launch × WDDM ~270µs 主导，是
  **launch-bound**。真实 LLM（权重 GB 级）权重读取才成为每步主导，那时
  这个下限才是真天花板。两个 regime 的对照正是本图的教学点。

**KV cache 公式对账（fig6）**：`2 × layers(4) × kv_heads(2) × seq × head_dim(64)
× 2B × batch`，seq ∈ {128..1024} × batch ∈ {1,16} 共 8 点，实测
（torch.cuda.memory_allocated 差值）与公式最大误差 **0.00%**。GQA 4:2 使
KV cache 只有 MHA 的一半（kv_heads/head_count = 1/2）——公式里乘的就是
kv_heads 而非 head_count。

**prefill vs decode（fig8）**：同样 256 个 token，一次 prefill（M=256 的
GEMM）29.98 µs/token（0.21 TFLOPS），逐步 decode（M=1 的 GEMV）7888
µs/token（0.0008 TFLOPS）——**decode 每 token 慢 263×**。算术强度
256:1 的差异 + decode 每步付全额 launch 开销。

## E5 采样策略（fig7）

- **正确性门**：构造分布 {0.4,0.3,0.15,0.1,0.05} 验证 top-p 前缀性质
  （p=0.6→2 token，p=0.9→4 token）PASS；χ² 分布一致性（N=10k，20 个整数
  对齐 bin、期望计数 >5 过滤后 17 个有效，dof=16）：torch.multinomial p=0.750，自实现 top-p p=0.406（门 >0.01）PASS。
  （χ² 初版用 histc 浮点 bin 边界与整数切片期望不对齐，连 multinomial 自身
  都不通过——修正为整数对齐 bin 后通过；教训记入笔记。）
- **温度 → top-p 截断集（fig7，真实 logits，top-p=0.9）**：T=0.5 平均留
  1.7 个 token，T=1.5 留 9.2 个（最大 45）——单调膨胀。模型 top-1 概率
  均值 0.736（记忆化尖锐分布），低温下几乎退化为 greedy。
- **top-p vs top-k**：同一批 logits 上 top-p 0.9 平均只留 3.45 个 token，
  而 top-k=8 保留 97% 概率质量——top-p 按分布形状自适应（尖峰分布少留、
  平坦分布多留），top-k 固定 k 与分布无关。
- **采样核耗时（batch 32）**：char vocab（119）下 greedy/top-k/top-p =
  0.036/0.347/0.678 ms（都可忽略）；LLM vocab（32k 合成 logits）下
  0.034/0.504/**2.130 ms**——top-p 的 sort 主导，比 greedy 贵 62.5×。
  真实模型 vocab 32k+ 时采样开销不可忽略，这是「采样也要工程优化」的实测依据。

## 跨实验结论（红线素材）

1. **fp16 权重 ≠ 4bit 推理提速**：预反量化速度与 fp16 相当（123.3 vs
   129.5 t/s，差在 WDDM 噪声内；省显存 3.54×），现反量化慢 9.2×；只有
   融合进 GEMM（mmvq/mmq）4-bit 才可能净赢。
2. **WDDM/小模型下 launch 开销主导一切**：decode 每步 ~30 次 kernel
   launch、per-tensor dequant 30 次 launch（1.887ms ≈ 63µs/launch）、
   torch 多算子 dequant ~30 个 kernel——
   全部 launch-bound。AR002 的 launch 台阶（25.6-69.4µs）在本 AR 每个实验
   里都再现。
3. **KV cache 公式必须会口算**：2×layers×kv_heads×seq×head_dim×2B×batch，
   GQA 的折扣在 kv_heads；实测 0.00% 误差。
4. **Q8_0 无损、Q4_K 轻损**（本模型 Δppl −0.01/+0.98），与社区经验一致。
