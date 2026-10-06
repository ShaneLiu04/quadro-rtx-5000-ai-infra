# gguf-lab — GGUF 量化格式复刻 + 端到端 GPU 推理实验（AR005）

自训 llama 架构 char-LM（RoPE/GQA/SwiGLU）→ 自实现 Q8_0/Q4_K 量化（对照
`ggml-quants.c` 逐行移植，含 6-bit scale 打包）→ GGUF 导出/读回（gguf-py）
→ GPU 量化推理全链路 → decode/KV cache/采样实验 → triton fused dequant。

详细分析：`results.md`；源码拆解：`../notes/llamacpp-internals.md`；
面试笔记：`../notes/llamacpp-notes.md`。

## 环境

- Windows + RTX 5000 (sm_75) + torch (CUDA) + triton；无网络、无编译器。
- 依赖 llama.cpp 的 `gguf-py`（`sys.path` 导入，上游零改动）。

## 复现

```powershell
# 1. 训练 char-LM（~150s, GPU）
python charlm.py

# 2. 量化自测（正确性门）
python quant_gguf.py
python kernels_dequant.py     # triton kernel vs 参考移植 bit-exact

# 3. E1-E5 全部实验（~15 min；也可逐个 python bench_infer.py e1 ... e5）
python bench_infer.py all

# 4. 出图（figs/fig1..fig9.png）
python plot_results.py
```

## 文件

| 文件 | 内容 |
|------|------|
| `charlm.py` | llama 架构微型模型、GPU 训练、KV cache 增量生成（一致性门） |
| `quant_gguf.py` | Q8_0/Q4_K 量化/反量化（ggml-quants.c 逐行向量化移植）+ GGUF 导出/读回 |
| `kernels_dequant.py` | triton 单 kernel fused dequant（flat 设计，264 GB/s @ sm_75） |
| `bench_infer.py` | E1-E5 实验驱动（门 + JSON；计时纪律沿用 AR002-004） |
| `plot_results.py` | 9 张图（300 DPI） |
| `results/` | JSON 数据 + GGUF 模型（f16/q8_0/q4_K）+ checkpoint |
| `results.md` | 逐图中文分析 + 诚实修订记录 |
