# [AR006] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR006 |
| AR 主题 | nanogpt-training-lab（nanoGPT 训练线 + llm.c 训练 kernel 源码拆解） |
| 关联 SR | 无（学习型 AR，Infra 面试准备第 6 站） |
| 日期 | 2026-10-06 |
| 状态 | Draft → Active |

## 1. 背景与目标

AR001-005 覆盖了 GPU 执行模型（puzzles）、GEMM（numba/cuBLAS/triton）、
推理侧（量化/GGUF/KV cache/采样）。本 AR 补上**训练侧**：用 nanoGPT
（Karpathy）的 GPT 实现在本机真跑大尺度预训练，横跨精度/规模/超参/
吞吐四条实验线；同时拆解 llm.c 的 CUDA 训练 kernel（fused
classifier/adamw/layernorm），理解「训练侧 kernel fusion 与推理侧的
区别」。全部面向面试：混合精度、MFU、scaling、LR/调度、profiler 归因
是训练岗高频考点。

**本机关键事实（决定实验设计）**：Quadro RTX 5000 = Turing sm_75——
**无 TF32、无 BF16 tensor core**（bf16 走仿真）、FP16 TC 峰值 89.2
TFLOPS、FP32 11.2 TFLOPS、16GB/448GB/s、WDDM。这与 A100/H100 时代
的「默认 bf16」叙事形成宝贵反差：fp16+loss scaling 是本机唯一快速路径。

## 2. 需求范围

**In Scope：**
- 本地语料构建（无网络）：llama.cpp/docs + llm.c doc/llmc + nanoGPT
  源码 ≈ 780KB「docs+code」混合语料，char-level tokenizer
- nanoGPT model.py 经 sys.path 导入（上游零改动），训练驱动自研
  （AMP/GradScaler/cosine/val/断点/计时/JSON）
- E1-E7 实验（见 §3）+ ≥10 图 + results.md + F4/F5 笔记

**Out of Scope：**
- llm.c / train_gpt2.cu 编译运行（无 MSVC/nvcc）——只源码拆解
- openwebtext/shakespeare 真数据（无网络）
- 多 GPU / DDP（单机）
- exllamav2（AR005 已声明 out-of-scope）

## 3. 功能需求

### 3.1 F1 — 数据与训练基线

**描述：** 语料构建 + char tokenizer + GPT 基线训练。
**触发条件：** `python gpt_lab.py prep` / `train`。
**期望行为：** 语料拼接去重统计落 JSON；训练跑通 fp16 AMP + GradScaler +
cosine，train/val loss 曲线落盘，生成样例呈现英文/代码结构。
**验收标准：**
- Given ~780KB 语料，When prep，Then vocab/字符数/entropy 落 JSON
- Given 6L-384 GPT（~10M 参数）训练 ≥3000 步，When 收敛，Then final
  train loss < 2.0 且生成样例为结构化文本（非乱码）
- Given 固定种子，When 两次采样，Then 输出确定

### 3.2 F2 — E2 精度三方对比（fp32 vs AMP-fp16 vs AMP-bf16）

**描述：** 同配置三精度训练对比：吞吐、loss 收敛、GradScaler 行为。
**验收标准：**
- Given 同配置（4L-256，固定步数），When 三精度各跑一遍，Then
  tokens/s、loss 曲线、（fp16 的）scaler scale 轨迹落 JSON
- fp16 吞吐 ≥ 1.3× fp32（TC 路径；诚实修订机制保留）
- bf16 在 Turing 的实测行为（预期仿真/不提速）如实记录——本 AR 的
  教学点而非失败

### 3.3 F3 — E3 模型规模 scaling

**描述：** {2L-128, 4L-256, 6L-384, 8L-512} ≈ {0.5M, 3M, 10M, 25M}，
固定 token 预算短训，val loss vs params、tokens/s vs params、VRAM vs
params 三条曲线 + Chinchilla 视角讨论。
**验收标准：**
- Given 固定 token 预算，When 四个规模短训，Then 三组数字落 JSON 且
  val loss 随规模单调（不单调则如实分析原因）
- FLOPs/token = 6N 公式对账：实测步时 vs 公式 FLOPS 推算 ±25% 内

### 3.4 F4 — E4 超参：LR 扫描 + batch

**描述：** LR {1e-4, 3e-4, 1e-3, 3e-3} ×（constant vs cosine）小配置
扫描；batch {16, 64, 256} 墙钟 vs 收敛。
**验收标准：**
- Given 各 LR 配置短训，Then loss 曲线与最佳 LR 落 JSON；过高 LR 的
  发散/尖峰如实记录（若不发散也如实记录）
- batch 增大 → tokens/s 提升且单步时间记录

### 3.5 F5 — E5 profiler 归因 + MFU roofline

**描述：** torch.profiler 对基线训练步做 kernel 级归因（attention/MLP/
optimizer/其他），MFU = 6N·tokens/s ÷ 89.2 TF 峰值，与 AR002-004 的
GEMM 数据交叉。
**验收标准：**
- Given profiler 表，Then top-k op 时间占比落 JSON（top-5 ≥ 50% 的
  归因有效性检查）
- MFU 数字对每个模型规模计算并与 E3 实测互洽（同一公式）

### 3.6 F6 — E6 llm.c 训练 kernel 源码拆解（不编译）

**描述：** fused_classifier.cuh（LM head+CE 融合，logits 不物化）、
adamw.cuh（fused 优化器）、layernorm.cuh fwd/bwd、mfu.h（FLOPs 公式）、
attention.cuh/encoder.cuh；产出「llm.c vs PyTorch eager 融合清单」。
**验收标准：**
- 每个结论附文件:行号；E5 profiler 数据与融合清单对照（哪些 op 在
  eager 下是热点、llm.c 为何融合它们）
- 不编造未编译验证的行为（与 AR005 同一红线）

### 3.7 F7 — E7 torch.compile + SDPA backend（诚实记录）

**描述：** Windows+inductor 现状尝试；SDPA 在 sm_75 的 backend 选择
（fp16 走 flash？fp32 走 math？）实测记录。
**验收标准：**
- 成败与原因如实落 JSON/文档；SDPA backend 判定用
  torch.backends.cuda.sdp_kernel 上下文或 torch.nn.attention 记录

### 3.8 F8 — 图表与文档

**描述：** ≥10 图（loss 系/精度吞吐/scaling log-log/LR 扫描/batch/
profiler 饼图/MFU vs 峰值/VRAM/scaler 轨迹），results.md 逐图中文分析，
F4 笔记 llmc-internals.md，F5 六段面试笔记 + INTERVIEW-INDEX 增补
（训练侧）+ README。
**验收标准：**
- 每图有参考线/标注/单位，无空图，数字与 JSON 一致
- 笔记数字与 JSON 完全一致（AR005 review 教训：文档数字必须逐个
  对 JSON 核对）

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 上游完整性 | 文件 | nanoGPT/llm.c 克隆零改动（mtime 聚类核查） |
| 环境 | 全局 python | 不 pip install；无网络 |
| GPU 真实性 | sm_75 | 全部数据真机实测，无模拟 |
| 可重复性 | 基准 | JSON + 脚本一键复现；计时沿用 AR002-005 纪律 |
| 时间预算 | 全套 | E1-E7 总 wall time ≤ 40 分钟（基线训练 ≤ 5 分钟） |
| 面试导向 | 笔记 | 六段结构 + 实测数字卡 + [实测]/[源码] 分级 |

## 5. 约束与假设

- nanoGPT model.py 经 sys.path 导入，不复制不改；llm.c 只读
- Turing 无 TF32/bf16 TC：fp16 AMP 是唯一提速路径（这是卖点不是缺陷）
- 假设：780KB 语料对 10M 模型足以呈现结构（过拟合教学点沿用 AR005）
- 假设：WDDM 噪声主要影响小 kernel 计时；训练步计时用多步中位数
- GradScaler 行为（scale 翻倍/回退）在 fp16 路径是可观测的教学数据
- 诚实修订机制：任何门被证明口径错误时，如实记录修订（AR005 惯例）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| MFU | Model FLOPs Utilization = 6N·tokens/s ÷ 峰值 FLOPS |
| 6N 公式 | 训练每 token 的 FLOPs ≈ 6×参数量（2 fwd + 4 bwd） |
| AMP | torch.amp 自动混合精度 |
| GradScaler | fp16 AMP 的动态 loss scaling，防梯度下溢 |
| Chinchilla | ~20 tokens/param 的.compute-optimal 经验律 |
