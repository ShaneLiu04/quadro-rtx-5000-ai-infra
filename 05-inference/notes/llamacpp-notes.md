# 推理侧面试笔记：量化 / GGUF / KV cache / 采样（AR005）

> 全部数字来自 gguf-lab 真机实测（results/*.json，RTX 5000 sm_75，WDDM）；
> 交叉引用 AR001-004 的基础数字。六段结构。

## 1. 高频问法

- llama.cpp 的 Q4_K 是什么布局？为什么比 Q4_0 好？（→ 3/4 段）
- 4-bit 量化能让推理更快吗？（→ 5 段红线 1）
- KV cache 显存怎么算？GQA 省多少？（→ 3 段卡 5）
- prefill 和 decode 有什么区别？为什么 decode 是 memory-bound？（→ 3 段卡 6/7）
- top-p 和 top-k 的区别？边界 token 怎么处理？（→ 4 段骨架 3）
- mmvq 和 mmq 什么时候用哪个？（→ 2 段追问链 4）
- W4A16 和 W4A8 的区别？（→ 2 段追问链 5）
- 温度对采样分布的影响？（→ 3 段卡 9）

## 2. 追问链

**Q1: 为什么量化后模型体积小了 3.5×，推理却不会快 3.5×？**
→ 权重占显存小 ≠ 每步计算少。W4A16 部署要么先反量化成 fp16（速度与原 fp16
相当——123.3 vs 129.5 t/s，5% 差在 WDDM 噪声内，只省显存），要么每步现场
反量化（慢 9.2×，dequant 66.9ms/步 vs 整个前向 8.1ms/步 [本机 3.2M 模型]）。
→ 追问：那 llama.cpp 怎么解决的？→ 把反量化**内联进 GEMM kernel**
（mmvq：每 block 现场反量化+点积，不物化 fp16）；大 batch 用 mmq（激活也
量化成 Q8_1，整数 tile 点积，W4A8 风格）。
→ 追问：为什么大 batch 不用 mmvq？→ mmvq 每 block 只服务 ≤8 行激活
（MMVQ_MAX_BATCH_SIZE=8 [源码 mmvq.cuh L3]），算术强度低；batch 大了
tile 化摊销才划算。

**Q2: Q4_K 的 144 字节怎么来的？**
→ 2B d + 2B dmin + 12B scales + 128B qs = 144B / 256 权重 = 4.5 bpw。
→ 追问：12B 怎么装 16 个数？→ 8 子块的 (scale,min) 各 6-bit = 48bit = 6B？
不——是 8×2×6bit = 96bit = 12B，每对 (sc,m) 的 6-bit 拆在三个字节的不同
位段（j<4 直存、j≥4 高 2 位借邻居字节顶 2 位 [源码 get_scale_min_k4 L880]）。
→ 追问：6-bit scale 够吗？→ scale 量化误差 2^-6 相对，而数据本身 4-bit
步长 2^-4 相对——scale 误差比数据粗度细 4×，不主导。

**Q3: KV cache 显存公式？**
→ `2 × layers × kv_heads × seq × head_dim × dtype_bytes × batch`。
→ 追问：为什么是 kv_heads 不是 head_count？→ GQA：多个 q head 共享一组
kv head，投影矩阵和 cache 都只按 kv_heads 存 [本机] 公式实测 8 组
(batch, ctx) 误差 0.00%。
→ 追问：70B（80L, 64 head, kvh=8, head_dim=8192/64=128, fp16）1 个序列
4096 长度？→ 2×80×8×4096×128×2B = 1.34GB（口算：2×80×8=1280 行；
×4096=5.24M cell；×128 dim=671M 元素；×2B=1.34GB——K/V 的因子 2 已在
乘式里，不要再当「两侧」乘第二次）。

**Q4: top-p 的边界 token 保不保留？**
→ 保留：保留集 = 最小前缀使 cumsum ≥ p，边界 token 就是那个越线的——
llama.cpp CPU 实现里 `cum_sum >= p → last_idx = i+1`（含当前）[源码
L1582]，llama.cpp 还有 min_keep 保底。
→ 追问：怎么验证实现正确？→ 构造分布 {0.4,0.3,0.15,0.1,0.05}：
p=0.6 → {前2个}（0.7≥0.6），p=0.9 → {前4个}（0.95≥0.9）；再 χ² 对
multinomial（N=10k，p=0.406 [本机]）。

**Q5: W4A16 和 W4A8 的区别？**
→ 激活精度：W4A16 反量化权重后 fp16 点积（mmvq 近似这个语义，其实
当场反量化当场点积）；W4A8 把激活也量化到 int8 做整数点积（mmq 路线、
以及 AWQ/GPTQ+TensorRT-LLM 的 int8 方案），误差来自双重量化但算子
全是整数吞吐高。
→ 追问：什么时候必须 W4A16？→ 激活分布尖锐/有离群值时激活量化损失大
（LLM.int-8 的 outlier 问题），这时保激活精度更稳。

## 3. 数字卡片（全部 [本机] AR005 实测）

| # | 卡 | 数字 | 上下文 |
|---|-----|------|--------|
| 1 | Q8_0 压缩比 | 1.88×（1.0625 B/w） | 34B/32 元素 |
| 2 | Q4_K 压缩比 | 3.54×（0.5625 B/w） | 144B/256 元素 |
| 3 | Q8_0 ppl 损失 | Δ −0.01 | 8-bit 噪声在 ppl 上不可见（fp16 15.60） |
| 4 | Q4_K ppl 损失 | Δ +0.98 | 语言结构保留，top-1 一致率 78.6% |
| 5 | KV 公式误差 | 0.00% | 2×4L×2kvh×seq×64×2B×batch，8 组实测 |
| 6 | prefill vs decode | 263×/token 慢 | 30µs vs 7888µs/token；算术强度 256:1 |
| 7 | decode batch 缩放 | 1→64 batch：吞吐 ×63.6，时间 +0.3% | 权重读取与 batch 无关 |
| 8 | on-the-fly dequant 代价 | 慢 9.2×（123.3→13.4 t/s） | torch 多算子 66.9ms/步；triton per-tensor 1.9ms → 104.1 t/s（−16% vs fp16） |
| 9 | 温度→top-p 截断集 | 1.7/2.15/3.45/9.2 个 token @ T=0.5/0.7/1.0/1.5 | vocab 119，模型 top-1 概率 0.736 |
| 10 | 采样核成本 | top-p 2.130ms vs greedy 0.034ms @ vocab 32k, B=32 | 62.5×；vocab 119 时可忽略 |
| 11 | dequant 实现阶梯 | 同规模（4096 块）：naive 9.20s → torch 向量化 2.20ms（4182×）；全尺寸（4096²）：torch 3.96ms/19.3 GB/s → triton 0.29ms/264 GB/s（59% HBM，13.7×） | e3.json 两个数据规模分开对比 |
| 12 | launch 开销 | per-tensor dequant：30 launch/1.887ms ≈ 63µs/launch（WDDM）；decode 步：~30 kernel launch 合计 8.1ms/步（平均 launch+执行 ~270µs，含 kernel 执行与同步，口径不同于纯 launch） | AR002 纯 launch 台阶 25.6-69.4µs |

## 4. 手写骨架

**骨架 1：Q4_K 反量化（面试白板级）**

```python
# block: d(fp16) dmin(fp16) scales[12] qs[128];  256 weights, 8 sub-blocks
def dequant_q4_K(blk):                      # x = d*sc*q - dmin*m
    d, dmin = fp16(blk[0:2]), fp16(blk[2:4])
    x = [0] * 256
    for sb in range(8):                     # sub-block = 32 weights
        sc, m = get_scale_min_k4(sb, blk.scales)   # 6-bit unpack
        for l in range(32):
            byte = blk.qs[(sb // 2) * 32 + l]      # pair = sb//2
            q = byte & 0xF if sb % 2 == 0 else byte >> 4
            x[sb * 32 + l] = d * sc * q - dmin * m
```

要点：低 nibble=偶子块、高 nibble=奇子块（成对交错）；scales 的 6-bit
拆位（j≥4 时高 2 位在邻居字节）；`-` 号（dmin 项是减）。

**骨架 2：KV cache 增量解码**

```python
# cache.k[l]: (B, kvh, ctx_max, hd)  预分配
def forward(ids, cache):                   # ids: (B, T)
    pos = cache.n                          # 本步起始位置（整层共享！）
    x = emb(ids)
    for blk in blocks:
        q = apply_rope(q_proj(x), pos)     # q 只对 T 个新位置
        k = apply_rope(k_proj(x), pos)
        cache.k[l][:, :, pos:pos+T] = k    # 写入槽位
        k = cache.k[l][:, :, :pos+T]       # 读全历史
        x = x + attn(q, k, v)              # T=1 时无 causal mask
    cache.n = pos + T                      # forward 结束才推进
```

要点：`cache.n` 是**整个 forward 的序列长度**，不能每层各自推进（本 AR
实做时踩过：block0 写完 n+=T，block1 位置全错——RoPE 位置错乱直接
输出乱码，maxdiff 18 级别）。

**骨架 3：top-p 采样**

```python
def top_p(logits, p):
    s_logits, s_idx = logits.sort(descending=True)
    probs = s_logits.softmax(-1)
    keep = (probs.cumsum(-1) - probs) < p  # 边界 token：cum-p < p → 保留
    # mask 还原到原词表序再 mask_fill(-inf)，softmax 后采样
```

要点：`(cum − prob) < p` 一个表达式同时处理边界（越线 token 的
cum−prob=前缀和 < p → 保留）与前缀性质；排序代价 O(V log V)。

## 5. 红线清单

1. **「4-bit 权重让推理更快」是错的**——预反量化速度与 fp16 相当
   （123.3 vs 129.5 t/s，差在 WDDM 噪声内；省显存 3.54×），现反量化慢
   9.2×；只有融合进 GEMM（mmvq/mmq）才可能净赢。速度收益只来自「省下
   权重搬运字节」且仅在带宽受限 regime。
2. **KV cache 公式里的头数是 kv_heads**——GQA 折扣不在 attention 里算，
   在 cache 尺寸里省（70B 例：8/64 → 1/8 显存）。公式必须口算级熟。
3. **decode ≠ prefill 的继续**：decode 每 token 是 M=1 的 GEMV（算术强度
   ~2 FLOP/byte），prefill 是 M=ctx 的 GEMM——同一权重 256× 的算术强度差
   （实测 263×/token）。说「模型推理是算力瓶颈」时先分清阶段。
4. **小模型/小 batch 的瓶颈是 launch 不是带宽**：3.2M 模型 decode 距
   权重流下限 577×（121 vs 69.8k t/s）；WDDM per-tensor dequant 口径 63µs/launch。把小模型的
   数字外推到真实 LLM 是常见错误（真实 LLM 权重 GB 级，带宽才主导）。
5. **量化误差门要用对口径**：rel-err（范数）与 max-abs 归一是不同东西；
   「8-bit 量化 → 误差 2^-8」混淆了步长与范数比（本 AR srs 初版就写错了，
   修订记录见 results.md E1）。高斯块量化的 rel-err ≈ amax/127/√12/rms。
6. **top-p 丢边界 token 是经典 bug**：保留集是「cumsum ≥ p 的最小前缀」，
   越线 token 本身在集合里（llama.cpp last_idx = i+1）。
7. **χ² 检验的 bin 边界要整数对齐**：histc 浮点边界 vs 整数切片期望错位
   时，连参考实现自己都「不通过」（p=0.0000）——测试也要被测试。
8. **norm 权重不量化**（llama.cpp 惯例）：1D RMSNorm scale 保持 fp16/f32。
9. **GGUF 维度是 ggml ne 序**（最快维在前）：numpy 序是其反序；读写两端
   谁反转要约定清楚（gguf-py reader 在 L350 统一 reversed）。

## 6. 60 秒电梯陈述

> 推理侧我做过一条完整的复刻链：自训 llama 架构 3.2M char-LM（RoPE+GQA+
> SwiGLU），自己实现 Q8_0/Q4_K 量化——对照 ggml-quants.c 逐行移植，包括
> Q4_K 的 6-bit scale 打包和加权 scale/min 搜索——导出 GGUF 再读回，
> 端到端验证：Q8_0 ppl 零损失、Q4_K +0.98，KV cache 公式实测 0.00% 误差。
> 性能侧我量化了「为什么 llama.cpp 要把 dequant 融进 GEMM」：朴素现反量化
> 慢 9.2×；我还用 triton 写了 flat 单 kernel dequant 做到 264 GB/s（59%
> HBM），中间踩过一个反直觉的坑——每 program 一个块的版本只有 16 GB/s，
> 因为 6.5 万个微型 CTA 调度本身就是瓶颈。decode 的 batch 曲线也实测了：
> 时间恒定、吞吐线性 ×63.6，教科书级 memory-bound 签名。采样侧验证了
> top-p 的前缀语义和 χ² 分布一致性，并测出 32k vocab 下 top-p 比 greedy
> 贵 62.5×——这正是 llama.cpp 用「先 top-256 再全排」启发式的原因。

## 交叉引用

- AR002：launch 台阶 25.6-69.4µs（本 AR per-tensor dequant 63µs/launch 再现）
- AR003：cuBLAS GEMM 基线与 wave 量化（decode GEMV 的对照系）
- AR004：triton @jit 源文件约束、do_bench 协议差异（本 AR flat kernel 直接受益）
- AR005 gguf-lab/results.md：全部实验细节与诚实修订记录
