# frameworks-lab — bitsandbytes / tinygrad 量化框架实验（AR007）

bnb + tinygrad 上游克隆只读解剖 → NF4 / LLM.int8 / 8-bit Adam 三个算法
纯 torch 复刻（码本逐值对齐官方）→ Quadro RTX 5000 真机测量每个"营销
数字"的成立条件 → NF4 packed / 8-bit 优化器 / QLoRA 显存账本外推 7.5B。

详细分析：`results.md`；源码拆解：`notes/bnb-internals.md` +
`notes/tinygrad-internals.md`；面试笔记：`notes/frameworks-notes.md`；
跨 AR 索引：`../../01-foundations/notes/INTERVIEW-INDEX.md`。

## 环境

- Windows + Quadro RTX 5000 (Turing sm_75, INT8 TC 2× fp16 峰值) +
  torch 2.5.1+cu121；无网络、无编译器。
- 上游零改动：`../bitsandbytes`、`../tinygrad` 克隆只读（native DLL 缺失
  → 降级导入，恰好成为 dispatch 机制的活样本）；nanoGPT `model.py` 经
  `sys.path` 导入（AR006 已验证路径）。
- 复刻而非调用：算法全部在 `quant_ops.py` 用纯 torch 实现，逐值对齐
  官方码本/语义（[源码] 标注 file:line），在本机 GPU 上实测。

## 复现

```powershell
# 1. 语料构建（~5s；data/corpus.txt 17.7MB，vocab 142，熵 5.732）
python corpus_prep.py

# 2. E0-E5 全部实验（~10 min；也可逐个 python framework_lab.py e0 ... e5）
python framework_lab.py all

# 3. 出图（figs/fig_e0a..fig_e5b 共 11 张，300 DPI，全部从 results/*.json 取数）
python plot_results.py

# 4. 文档数字一致性断言（results/doc_check.json）
python verify_numbers.py
```

## 实验一览

| 实验 | 问题 | 关键结果 |
|------|------|---------|
| E0 | 框架探测 | bnb 降级导入但 default 注册广播全后端（纯 torch 兜底可跑）；tinygrad 惰性图 UOp 可内省（56→236 节点，PRNG 入图），realize 双栈失败诚实记录；`torch._int_mm` CUDA 位精确 |
| E1 | NF4 码本好在哪 | @block64 **NF4 0.092 < INT4 0.100**；裸 ndtri 0.127 反输 INT4（端点钉 ±1 为 absmax 归一化设计）；真实权重 head 层（峰度 8.4）优势 0.1046 vs 0.1291 |
| E2 | LLM.int8 离群分解 + int8 吞吐 | 重尾 per-row 4.86e-2；**τ=8（0.4% 列 fp16）→ 1.14e-2 回 int8 本底**（τ=2 平凡解全 fp16；τ=16/32 残留污染回升）；**G2 诚实 FAIL**：默认布局 0.59× fp16，TN 1.29×（配对 1.83× 提升）——bnb 自定义布局就是钉 TN |
| E3 | 8-bit Adam | 与 fp32 同种子双训 800 步终值差**在训练噪声内**（三次运行 0.0007/0.0253/0.0207，门 <0.05 三过）；状态显存 **0.254×**；未融合步时 1.84×（bnb 融合 kernel 存在的原因）。前置：线性 int8 版本真实发散过（v 下溢→步长 10⁵） |
| E5 | 显存账本 | bytes/param 实测 16.78/14.58/8.03/**0.5315**（理论 ±5% 内）；7.5B 外推 125.8→**4.0GB QLoRA** |

## 文件

| 文件 | 内容 |
|------|------|
| `corpus_prep.py` | 语料收集（bnb py/csrc/docs + tinygrad + nanoGPT）+ char tokenizer |
| `quant_ops.py` | 纯 torch 复刻：NF4/FP4/INT4 码本、blockwise/vectorwise 量化、int8 mm dequant、LLM.int8 分解、dynamic map、Adam8bitStates（全部带 [源码] file:line 标注） |
| `framework_lab.py` | 驱动（e0-e5 + all；门预注册 + JSON 输出 + 诚实 FAIL 记录） |
| `plot_results.py` | 11 张图（300 DPI，JSON 单一真值源） |
| `verify_numbers.py` | results.md / notes 数字 vs JSON 断言（doc_check.json） |
| `results/` | e0/e1/e2/e3/e5 JSON |
| `figs/` | fig_e0a..fig_e5b |
| `results.md` | 逐图中文分析 + G2 诚实修订记录 + 一致性自查 |
| `notes/` | bnb-internals / tinygrad-internals（源码级拆解）/ frameworks-notes（面试六段） |
