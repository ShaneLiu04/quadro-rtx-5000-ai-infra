# [AR005] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR005 |
| AR 主题 | llamacpp-inference-lab（GGUF 量化格式复刻 + 自训 char-LM 端到端 GPU 推理 + decode/KV cache/采样实验 + llama.cpp 源码拆解） |
| 关联 SR | 仓库 README「建议顺序」第 5 条：05-inference 推理侧——量化、KV cache、采样；llama.cpp + exllamav2 源码 |
| 日期 | 2026-10-06 |
| 状态 | Draft |

## 1. 背景与目标

`05-inference/` 含 llama.cpp 与 exllamav2 完整克隆。本机约束（smoke 已核）：
无外网（代理全拦）、无 MSVC/nvcc（C++/CUDA 扩展不可编译）、无本地模型权重、
transformers/torch 齐备、llama.cpp 自带 gguf-py（纯 Python GGUF 读写）。

因此 AR005 走「**复刻 llama.cpp 推理路径**」路线，不依赖网络与编译器：

- **自训微型 char-LM**（llama 架构：RoPE + GQA + SwiGLU，~2M 参数，
  语料 = llama.cpp docs 500KB 英文技术文本，GPU 训练 ~1 分钟）；
- **自实现 Q8_0 / Q4_K 量化与反量化**——逐行对照 `ggml/src/ggml-quants.c`
  参考实现（含 6-bit scale/min 打包），块大小与静态布局对账
  （Q8_0=34B/32 元素、Q4_K=144B/256 元素）；
- **GGUF 导出/读回闭环**：GGUFWriter 写（含 llama 架构 metadata + 量化张量），
  GGUFReader 读回，逐张量对拍 + 文本生成一致性；
- **GPU 端到端量化推理**：fp16 vs Q8_0 vs Q4_K 的困惑度与速度
  （dequant 开销分解、向量化 dequant kernel 带宽）；
- **decode 吞吐 / KV cache / 采样策略实验**（面试三大推理话题的实测证据）；
- **llama.cpp 源码拆解文档**（mmvq/mmq、k-quants、KV cache、GQA、RoPE、采样）。

**smoke 预实验（2026-10-06，已去风险）：**
- gguf-py 可用（GGUFWriter/Reader、Q4_K=12/Q8_0=8 枚举）
- block_q4_K = {fp16 d, fp16 dmin, uint8 scales[12], uint8 qs[128]} = 144B（QK_K=256、
  QR4_K=2、QI4_K=32、K_SCALE_SIZE=12，均经 ggml-common.h 确认）
- 0.43M 模型 GPU 训练 124 steps/s（300 步 loss 4.95→2.6，2.4s）——2M 模型 3k 步 <1 分钟
- 语料：llama.cpp/docs 52 个 md 共 503KB 英文，vocab 119
- 上游 AGENTS.md 明确禁止自治 agent 提交——本 AR 与其完全兼容：**零改动、只读**

## 2. 需求范围

**In Scope：**
- F1：lab 框架 `05-inference/gguf-lab/`：charlm.py（模型/训练/tokenizer）、
  quant_gguf.py（Q8_0/Q4_K 量化/反量化 + GGUF 导入导出）、bench_infer.py
  （--exp E1..E5、计时纪律沿用 AR002-004：烧机/中位数/Event/JSON）、plot_results.py
- F2：实验组：E1 量化往返正确性（块布局对账 + 误差分布）、E2 GGUF 闭环
  （导出/读回/逐张量对拍/生成一致）、E3 量化推理质量与速度（困惑度 +
  dequant 开销分解 + 向量化 kernel 带宽）、E4 decode 吞吐 vs batch +
  KV cache 显存公式实测 + prefill/decode roofline、E5 采样策略（top-k/top-p/
  temperature 的 GPU 实现与分布行为）
- F3：`plot_results.py` ≥8 张可解释图
- F4：`notes/llamacpp-internals.md` 源码拆解（[源码]/[本机] 分级标注）
- F5：`notes/llamacpp-notes.md` 六段面试笔记 + INTERVIEW-INDEX 增补 + README

**Out of Scope：**
- 不修改 llama.cpp/exllamav2 上游任何文件（零改动，mtime 核查）
- 不编译任何 C++/CUDA 扩展（无 MSVC/nvcc）：llama.cpp 二进制与 exllamav2 均不构建；
  exllamav2 仅作源码浏览，不入本 AR 交付
- 不下载模型权重（无外网）；不用 transformers 加载预训练模型（无权重）
- 不 pip install；不实现 Q2_K/Q3_K/Q5_K/Q6_K/IQ 系列（Q8_0+Q4_K 足够覆盖
  「均匀量化 vs 带子块 scale 的 k-quant」两类范式）
- 不做 llama.cpp 的 graph executor 复刻（只拆解文档，推理用自实现 torch 前向）

## 3. 功能需求

### 3.1 F1 — lab 框架与自训模型

**描述**：charlm.py 定义 llama 架构微型模型（RoPE、GQA 4:2、SwiGLU、RMSNorm，
4 层 d=256 ctx=256，~2M 参数）；`train()` 在 GPU 训练并保存 fp16 checkpoint +
loss 曲线 JSON；`generate()` 支持 KV cache 增量解码与批处理。
bench_infer.py 沿用编译/计时分离协议（AR004 教训：预编译/预热与计时分离 +
烧机 + 中位数）。

**验收标准：**
- Given 训练 3000+ 步，When 保存 checkpoint，Then loss 收敛（<2.0）且生成文本
  呈现英文单词级结构（非均匀乱码）
- Given 生成函数，When batch=1 与 batch=16 采样，Then 输出确定性可复现（固定种子）
- Given KV cache 实现，When 增量解码，Then 输出与无 cache 的全量重算逐 token 一致

### 3.2 F2 — 五组实验

**E1 量化往返正确性**：Q8_0/Q4_K 的 quantize/dequantize 自实现；
块大小对账（34B/144B per block）；量化→反量化误差 vs 权重分布
（直方图数据落 JSON）；与 ggml-quants.c 的算法逐行对照说明（文档）。
- 验收：Q8_0 块内 max-abs scale 公式正确；Q4_K 的 6-bit scale/min 打包
  （make_scale_min_max_6bit 等价实现）与参考一致；误差门：Q8_0 rel-err ≤ 2^-8、
  Q4_K rel-err ≤ 0.25（权重 max-abs 归一）

**E2 GGUF 闭环**：GGUFWriter 导出 fp16/Q8_0/Q4_K 三份模型文件
（含 llama 架构 metadata：embedding_length/block_count/head_count/
 head_count_kv/context_length + tokenizer（char vocab））；
GGUFReader 读回重建模型；逐张量对拍；三份模型生成文本与原模型对比。
- 验收：读回模型与导出前权重逐张量一致（fp16 精确；量化格式按 dequant 后
  ≤E1 门）；metadata 读回一致；Q4_K 模型生成连贯文本（与 fp16 同种子前缀
  生成重叠度 ≥60%，量化扰动不完全破坏语言结构）

**E3 量化推理质量与速度**：困惑度（fp16 vs Q8_0 vs Q4_K，同一 held-out 文本）；
端到端解码速度（dequant→GEMM 流水 vs 预反量化驻留 fp16 权重）；
dequant kernel 本身：naive 逐元素 vs 向量化 torch 视图实现的带宽与加速比。
- 验收：三方困惑度量化（Q8_0 Δppl ≤0.05、Q4_K Δppl 有记录）；
  dequant 开销占解码步时间的比例有分解；向量化 kernel 带宽 ≥200 GB/s
  （≥45% HBM）且相对 naive 加速 ≥10×

**E4 decode 吞吐与 KV cache**：batch ∈ {1,2,4,8,16,32,64} 的 tokens/s 曲线
（同一模型、等长前缀增量解码）；KV cache 显存公式
  `2 × layers × kv_heads × head_dim × seq × 2B` 与实测显存增量对账；
  prefill（一次算 ctx tokens）vs decode（逐 token）的时间/token 对比。
- 验收：batch↑ 单位吞吐↑（带宽受限的批并行证据）曲线有拐点或饱和带解释；
  KV 公式对账误差 ≤5%；prefill vs decode 的算术强度差异给出 roofline 定位

**E5 采样策略**：greedy/top-k/top-p/temperature 的 GPU 实现（torch.sort 为主）；
在真实 logits（自训模型）上对比：不同 temperature 下 top-p 截断集大小分布、
top-k vs top-p 的尾部保留行为；采样核的耗时（sort 开销 vs logits 规模）。
- 验收：top-p 实现正确（累积概率截断的边界处理有门：给定构造分布验证）；
  温度→截断集大小的单调性有图；采样实现与 torch 官方 multinomial 抽样
  分布一致性（KS 检验或 χ²，N=10k 样本级）

### 3.3 F3 — 图表

**描述**：≥8 图：fig1 训练 loss 曲线、fig2 量化误差直方图（Q8 vs Q4_K）、
fig3 dequant kernel 带宽（naive vs 向量化 vs HBM 参考线）、fig4 困惑度对比、
fig5 decode 吞吐 vs batch + roofline 线、fig6 KV cache 公式 vs 实测、
fig7 采样分布行为（温度/截断集）、fig8 prefill vs decode 时间线、
fig9 端到端生成速度（三精度 + dequant 摊销）。英文图面，300 DPI。

**验收标准：** 每图含数据 + 参考线/标注 + 单位；无空图；数字与 JSON 一致。

### 3.4 F4 — llama.cpp 源码拆解文档

**描述**：`notes/llamacpp-internals.md`：(1) ggml 计算图执行模型
（tensor/node/build/compute，[源码]）；(2) mmvq vs mmq 两条量化 GEMM 路线
（batch-1 的 int4 点积 vs tile 化批量反量化，[源码] mmvq.cu/mmq.cu/
mmq-instance-q4_k.cu）；(3) k-quants 布局拆解（super-block 256/子块 32/
6-bit scales，对照本 AR E1 自实现，[源码]+[本机]）；(4) KV cache 管理与
GQA 折叠（llama-kv-cache.cpp，[源码]）；(5) RoPE（ggml_rope 实现，[源码]）；
(6) 采样链（llama-sampling.cpp 的 top-k/top-p 与本 AR E5 实现对照）。
[源码]/[本机] 分级标注；不编造未验证的行为。

**验收标准：**
- 每个源码结论附文件/行号；本 AR 实测部分与 JSON 一致
- 量化格式拆解与 E1 自实现互证（块布局/打包公式一致）

### 3.5 F5 — 面试笔记

**描述**：`notes/llamacpp-notes.md` 六段结构；数字卡全部来自本 AR JSON +
AR001-004 衔接；INTERVIEW-INDEX 增补「推理侧：量化/KV cache/采样」条目；
README 复现步骤。

**验收标准：**
- 笔记数字与 JSON 完全一致；六段齐全；红线含「fp16 权重 ≠ 4bit 推理提速」
  （dequant 开销）、「KV cache 显存公式必须会口算」类条目

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 上游完整性 | 文件 | llama.cpp/exllamav2 克隆零改动（mtime 聚类核查） |
| 环境稳定 | 全局 python | 不 pip install；无网络访问 |
| GPU 真实性 | sm_75 | 全部数据真机实测，无模拟 |
| 可重复性 | 基准 | JSON + 脚本一键复现；长窗口波动 <5%（WDDM 口径） |
| 时间预算 | 全套 | E1-E5 总 wall time ≤ 30 分钟（训练 ≤2 分钟） |
| 面试导向 | 笔记 | 六段结构 + 实测数字卡 |

## 5. 约束与假设

- gguf-py 经 sys.path 导入（`05-inference/llama.cpp/gguf-py`），不复制不改
- torch 前向用 SDPA/手写 attention（flash-attn sm_75 不可用是 AR008 主题，此处不展开）
- 计时纪律沿用 AR004：预热与计时分离、每实验 fresh burn、中位数、时钟遥测
- 假设：Q4_K 自实现与 ggml-quants.c 参考 NRMSE 一致（若发现参考实现细节
  差异，如实记录——证据优先）
- 假设：503KB 语料对 2M 模型足够学到字符级英文结构（smoke：0.43M/300 步已现
  loss 2.6 且下降中；训练不足则降模型规模，不降交付标准）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| Q8_0 | 块量化：32 元素/块，块内 max-abs 定 fp16 scale，权重存 int8 |
| Q4_K | k-quant：256 元素 super-block，8 个 32 元素子块各带 6-bit scale/min，super-block 带 fp16 d/dmin |
| GGUF | llama.cpp 的模型容器格式：metadata + 命名张量（含量化类型标注） |
| mmvq / mmq | ggml CUDA 的两条量化 GEMM：mmvq=batch-1 逐 kernel 反量化点积；mmq=tile 化批量中间格式 |
| GQA | grouped-query attention：多个 query head 共享一组 kv head，KV cache 缩放 kv_heads/head_count |
| RoPE | rotary position embedding：q/k 按旋转矩阵编码位置 |
| prefill / decode | 推理两阶段：prefill 一次处理整个 prompt（算力型），decode 逐 token 增量（带宽型，受权重+KV 读取支配） |
