# nanogpt-lab — nanoGPT 训练实验（AR006）

本地语料（llama.cpp docs + llm.c 源码 + nanoGPT 源码，898,703 chars）
→ char tokenizer → 上游 nanoGPT `model.py` 零改动导入（`sys.path`）
→ fp32/fp16/bf16 训练、固定预算 scaling、LR/batch 扫描、profiler 归因。

详细分析：`results.md`；llm.c 源码拆解：`../notes/llmc-internals.md`；
面试笔记：`../notes/nanogpt-training-notes.md`。

## 环境

- Windows + Quadro RTX 5000 (Turing sm_75) + torch 2.5.1+cu121；
  无网络、无编译器（llm.c 只读码不编译）。
- 上游零改动：nanoGPT `model.py` 经 `sys.path` 导入；
  MFU 峰值自算（fp16 TC 89.2 / fp32 11.2 TF，不用上游 estimate_mfu——
  nanoGPT 硬编码 A100，llm.c gpu_db 无 Turing 卡）。

## 复现

```powershell
# 1. 语料构建（~5s；data/corpus.txt + tokens.npy + corpus_stats.json）
python corpus_prep.py

# 2. E1-E7 全部实验（~15 min；也可逐个 python gpt_lab.py e1 ... e7）
python gpt_lab.py all

# 3. 出图（figs/fig1..fig13.png，300 DPI，全部从 results/*.json 取数）
python plot_results.py

# 4. 文档数字一致性断言（92 项 → results/doc_check.json）
python verify_numbers.py
```

## 实验一览

| 实验 | 问题 | 关键结果 |
|------|------|---------|
| E1 | fp16+GradScaler 基线 | 10.67M 模型 3000 步：val U 型（1.468@750→2.348），300k tok/s，MFU 23.9% |
| E2 | 精度阶梯 | fp16=2.67× fp32；**bf16 反慢 0.57×**（sm_75 无 BF16 TC，仿真陷阱） |
| E3 | 固定 25M token 预算 scaling | 最优在 3M（1.576）；25M 欠训练最差（2.063）；MFU 单调 7.5→29.9%（门诚实修订） |
| E4 | LR/batch 扫描 | 3e-3 零发散 final 0.82；等预算 B16 loss 0.706 vs B256 2.411（更新次数 vs 吞吐） |
| E5 | profiler 归因 | eager 一步 GEMM 仅 27%；transpose 245 次/步；73% 非 GEMM = llm.c/compile 的融合对象 |
| E7 | SDPA 后端 / torch.compile | sm_75 无 flash→mem-efficient 回退；compile 因 triton API 不匹配失败（诚实记录） |

## 文件

| 文件 | 内容 |
|------|------|
| `corpus_prep.py` | 语料收集 + char tokenizer + 统计（vocab 124，熵 5.162 bits/char） |
| `gpt_lab.py` | 训练驱动（上游零改动导入、AMP/GradScaler、cosine、fused AdamW、门+JSON） |
| `plot_results.py` | 13 张图（300 DPI，JSON 单一真值源） |
| `verify_numbers.py` | results.md 数字 vs JSON 的 84 项断言（doc_check.json） |
| `results/` | e1-e7 JSON + checkpoint + doc_check.json |
| `figs/` | fig1..fig13 |
| `results.md` | 逐图中文分析 + E3 门诚实修订记录 + 一致性自查 |
