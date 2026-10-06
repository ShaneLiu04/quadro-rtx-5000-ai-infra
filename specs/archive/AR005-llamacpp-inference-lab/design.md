# [AR005] 详细设计

| 字段 | 内容 |
|------|------|
| AR 编号 | AR005 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-06 |

## 1. 交付物结构

```
05-inference/
├── gguf-lab/
│   ├── charlm.py             # F1：llama 架构微型模型 + GPU 训练 + KV cache 生成
│   ├── quant_gguf.py         # F1：Q8_0/Q4_K 量化/反量化 + GGUF 导出/读回
│   ├── bench_infer.py        # F2：--exp E1..E5/all 调度 + 计时 + 正确性门 + JSON
│   ├── plot_results.py       # F3：读 JSON → figs/*.png（300 DPI）
│   ├── results/*.json        # e1..e5/env + checkpoint/loss 曲线
│   ├── figs/fig1..fig9.png
│   ├── results.md            # 逐图中文分析
│   └── README.md             # 复现步骤
├── notes/
│   ├── llamacpp-internals.md # F4 源码拆解（[源码]/[本机] 分级）
│   └── llamacpp-notes.md     # F5 六段面试笔记
└── llama.cpp/, exllamav2/    # 上游克隆，零改动（mtime 核查）
```

## 2. 统一纪律

### 2.1 计时（沿用 AR002-004 修订版）

1. 会话级 2s FP32 大 GEMM 烧机 → 配置级 3 次 warmup → 自适应 reps（3-10）→ 中位数 time_ms
2. CUDA Event 打点；权重/数据常驻 device（无 H2D 计入）
3. 生成速度口径：tokens/s = batch × tokens / wall time（Event 包裹整段生成）
4. 显存口径：`torch.cuda.memory_allocated()` 前后差（KV cache 对账），非 reserved

### 2.2 正确性门（先于计时）

| 对象 | 参考 | 判据 |
|------|------|------|
| KV cache 增量解码 | 同模型全量重算 | 逐 token logits max abs diff ≤ 1e-4（fp16 累计路径） |
| Q8_0 量化往返 | 原 fp16 权重 | rel-err ≤ 2^-8（8-bit 均匀量化理论界） |
| Q4_K 量化往返 | 原 fp16 权重 | rel-err ≤ 0.25（max-abs 归一口径，4.5 bit 有效精度） |
| GGUF 读回权重 | 导出前 | fp16 精确 equal；量化格式 dequant 后 ≤ 对应门 |
| top-p 采样 | 构造分布（已知概率向量） | 截断集 = 最小前缀集使 cumsum ≥ p（含边界 token）|
| 采样分布一致性 | torch.multinomial | χ² 检验 p > 0.01（N=10k，分桶 20）|
| Q4_K 模型生成 | fp16 模型同种子生成 | 前缀重叠度 ≥60%（量化扰动的语言结构保留）|

### 2.3 JSON schema

`{exp, variant, ...params, time_ms|tokens_per_s|ppl|bytes, reps, [err_max, err_rel, hist]}`；
save() 整实验覆盖语义（AR003 教训）；失败即 gate 异常中止（不留半数据）。

## 3. 实现设计

### 3.1 模型（charlm.py）

- 架构对齐 llama：`emb(V,256) → 4×Block[rmsnorm→qkv(4 heads q, 2 heads kv)→
  RoPE→SDPA→o_proj；rmsnorm→SwiGLU(256→683→256 近似 8/3)] → rmsnorm → lm_head`
- RoPE 标准 theta=10000；GQA 4:2（KV cache 只存 2 组 kv → E4 公式演示 GQA 折扣）
- 参数 ~2.1M；fp32 训练（AdamW lr=3e-4, wd=0.01, 3000 步, batch 64×ctx128）
  → 保存 fp16 state_dict + loss 曲线 JSON；训练 ~25s（smoke 124 steps/s 外推）
- `generate(prompt, n, cache=True, batch=1)`：cache 版逐 token 只算最后一行；
  一致性门与全量重算对拍
- tokenizer：char 级（vocab 119，含 byte-fallback 无需——全 ASCII+常见符号）

### 3.2 量化（quant_gguf.py）

- **Q8_0**（对照 ggml-quants.c `quantize_row_q8_0_ref`）：32 元素/块；
  `d = max|x|/127 (fp16)`，`q = round(x/d)`；块布局 = 2B d + 32B qs = 34B；
  dequant `x = d × q`
- **Q4_K**（对照 `quantize_row_q4_K_ref` + ggml-common.h block_q4_K）：
  - super-block 256 元素 → 8 子块 × 32；每子块求 (max, min) →
    scale `a=(max-min)/15`、min `b=min` → quant 6-bit (`q = round((x-b)/a)` clamp 0-15 是 4bit…
    参考实现的 scale/min 量化：make_scale_min_max_6bit 把 8 组 (a,b) 各 6bit 打包进 12B)
  - super-block 级：`d = max(a_i)/63`（6bit 量化 a）、`dmin = max(-b_i)/63`
    （min 常为负，取绝对值方向按参考实现）→ 逐子块
    `a_i^6 = round(a_i/d)`、`b_i^6 = round(b_i/dmin)`
  - 权重 nibble：`q ∈ [0,15]`，两元素/byte；144B/super-block 对账门
  - dequant：`x = (a_i^6 × d) × q + (b_i^6 × dmin)`（与参考公式逐项一致）
- **向量化 dequant**（E3 性能对象）：torch 视图把 packed bytes 转 uint8 张量 →
  nibble 解包（`& 0xF`、`>> 4`）→ gather scales → 乘加；对照 naive Python 循环
- **GGUF I/O**：GGUFWriter（sys.path 指 llama.cpp/gguf-py）：
  metadata 按 llama 架构键（`llama.embedding_length`、`llama.block_count`、
  `llama.attention.head_count`、`llama.attention.head_count_kv`、
  `llama.rope.dimension_count`、`llama.context_length`、`tokenizer.ggml.tokens` 等）；
  张量命名 `token_embd.weight`/`blk.N.attn_q.weight`/…（对齐 llama.cpp convert 约定）；
  Q8_0/Q4_K 张量以 raw bytes 写入（`GGUFWriter.add_tensor` + quant type 标注）；
  GGUFReader 读回重建 state_dict

### 3.3 实验设计

**E1 量化往返**：对训练后权重逐张量 quantize→dequant；误差统计
（rel-err/max-err/直方图 32 桶）落 JSON；块大小对账（len(bytes)=144×n_blocks）；
预期：Q8_0 rel ~2^-9；Q4_K rel ~0.1（4.5 bit）；两格式误差分布形状差异
（Q8_0 均匀 vs Q4_K 与子块统计相关）是 fig2 素材

**E2 GGUF 闭环**：三份导出（fp16/Q8_0/Q4_K，文件大小对比进 JSON——
真实压缩比 vs 理论 4.5/8 bit）；读回重建 → 逐张量对拍 → 三模型同 prompt
生成 100 token → Q4_K vs fp16 重叠度门（≥60%）；metadata 回读一致性

**E3 量化推理质量与速度**：
- 质量：三方困惑度（held-out 500 字符窗 × 20 窗平均）；门：Q8_0 Δppl ≤ 0.05
- 速度：解码步时间分解 = {dequant W_all → fp16, GEMM 序列}；
  对比「每步现 dequant」vs「启动时预反量化驻留」两种部署的 tokens/s
  ——「fp16 权重 ≠ 4bit 推理提速」红线的量化证据
- dequant kernel：naive（逐元素 Python 循环 1K 元素基线）vs 向量化 torch
  （全量 2.1M 权重）；带宽 = 权重字节/时间；门 ≥200 GB/s、加速 ≥10×

**E4 decode 吞吐与 KV cache**：
- batch ∈ {1,2,4,8,16,32,64}（pad 到同 batch 重算？——用同一 prompt 复制 batch 份，
  独立 KV）增量解码 64 token → tokens/s（总吞吐）与 per-seq tokens/s
- 理论：decode 每步读 ~2.1M 权重（fp16 4.2MB）→ 单序列 tokens/s 上限 =
  448×0.83/4.2MB ≈ 88 t/s；batch 提升 → 权重读摊销 → 吞吐上升直至算力/开销上限
- KV cache：`2×4 层×2 kv_head×64 dim×seq×2B`；seq 扫描 {128, 256, 512, 1024} ×
  batch {1,16} 实测 memory_allocated 增量 vs 公式（门 ≤5%）
- prefill vs decode：ctx=256 prefill 一次 vs 逐步 decode 的时间/token 与
  算术强度（prefill GEMM M=256 vs decode M=1——「decode 是 GEMV」的实测）

**E5 采样策略**：
- 实现：greedy（argmax）、top-k（kth 阈值 mask）、top-p（sort+cumsum 截断）、
  temperature（logits/T 前置）
- 行为实验：真实 logits（模型对 20 个 prompt 的 next-token 分布）上，
  T ∈ {0.5,0.7,1.0,1.5} × top-p 的截断集大小分布；top-k vs top-p 在
  长尾分布上的保留行为对比
- 正确性：构造分布 {0.4,0.3,0.15,0.1,0.05} 验证 top-p={0.6,0.9} 截断集；
  与 multinomial 的 χ² 一致性（N=10k）
- 耗时：sort 采样核 vs greedy argmax（logits 规模 = vocab 119 固定——
  如实记录「小 vocab 下采样开销可忽略，大模型 vocab 32k 时 sort 才显著」的口径）

### 3.4 环境探测（env.json）

torch/cuda 版本、GPU/显存、gguf-py 路径与版本探测、语料统计（字符数/vocab）、
训练超参与最终 loss。

## 4. 图表规格（figs/*.png，300 DPI，英文图面）

| fig | 内容 | 关键标注 |
|-----|------|---------|
| fig1 | E0 训练 loss 曲线 | 步数轴 + 最终 loss 注释 |
| fig2 | E1 量化误差直方图（Q8_0 vs Q4_K 同坐标） | rel-err 中位数标注 |
| fig3 | E3 dequant kernel：naive vs 向量化 GB/s | 448 GB/s 峰值线、45% 线 |
| fig4 | E3 困惑度对比（fp16/Q8_0/Q4_K 柱状） | Δppl 数值标注 |
| fig5 | E4 tokens/s vs batch（总吞吐 + 单序列） | 单序列理论上限线（权重带宽） |
| fig6 | E4 KV cache：公式 vs 实测散点 | y=x 参考线、GQA 折扣注释 |
| fig7 | E5 温度→top-p 截断集大小分布 | T 系列颜色映射 |
| fig8 | E4 prefill vs decode 时间/token + 算术强度标注 | 「decode = GEMV」注释 |
| fig9 | E3 端到端生成速度：现 dequant vs 预反量化 vs fp16 | 「fp16 权重 ≠ 4bit 提速」注释 |

## 5. F4 文档结构（llamacpp-internals.md）

1. ggml 计算图模型：tensor/node/graph build→compute（[源码] ggml.c/llama-graph.cpp）
2. 量化 GEMM 双路线：mmvq（batch-1 反量化点积）vs mmq（tile 中间格式）
   （[源码] mmvq.cu/mmq.cu/mmq-instance-q4_k.cu；选路逻辑按 batch/形状）
3. k-quants 布局拆解：super-block 256/子块 32/6-bit scales（对照 E1 自实现，
   [源码]+[本机] 互证）
4. KV cache：llama-kv-cache.cpp 的槽位/seq 管理 + GQA 折叠
   （对照 E4 公式，[源码]+[本机]）
5. RoPE：ggml_rope 的 theta/旋转实现（[源码]）
6. 采样链：llama-sampling.cpp top-k/top-p 实现与本 AR E5 对照（[源码]+[本机]）
7. 「我们复刻了什么 / 没复刻什么」：graph executor 与 CUDA kernel 未复刻
   （torch 前向替代），差异如实列表

## 6. 风险与预案

| 风险 | 概率 | 预案 |
|------|------|------|
| Q4_K 6-bit 打包与参考实现细节不一致（make_scale_min_max 的位序/符号约定） | 中 | 开发中直接对照 ggml-quants.c 逐行移植；读 C 源确认位序后再写 torch；对拍门失败即修 |
| 2M 模型 3k 步 loss 未达 <2.0 | 低 | smoke 已证 0.43M/300 步 2.6 且在降；必要时加步数/降 lr 阶梯，训练封顶 2 分钟 |
| Q4_K 生成重叠度 <60%（小模型对量化扰动敏感） | 中 | 小模型 logits 边缘分布平坦，量化扰动易改 argmax——如实报告实测值并分析（本身是「小模型更怕量化」的教学点）；门按 srs 若不可达则记录修订 |
| GGUF raw bytes 写量化张量的对齐/shape 元数据问题 | 中 | 先以 fp16 全通，再加 Q8_0（块结构简单），最后 Q4_K；逐格式 gate |
| WDDM 计时噪声 | 高 | 沿用烧机/中位数/复跑抽查协议；短窗口只报方向 |
| batch=64 decode 显存超限 | 低 | KV 64×1024×2×4×2×64×2B ≈ 128MB 充裕 |

## 7. 验收对照（srs §3 → 交付物）

- §3.1 模型/门 → charlm.py + 一致性门输出 + loss 曲线 JSON
- §3.2 E1-E5 → results/e1..e5.json（+env.json）
- §3.3 图表 → figs/fig1..fig9.png
- §3.4 F4 → notes/llamacpp-internals.md
- §3.5 F5 → notes/llamacpp-notes.md + INTERVIEW-INDEX + results.md + README
- §4 NFR → mtime 核查（llama.cpp/exllamav2）+ 无 pip/无网络声明 + 真机 JSON + wall time
