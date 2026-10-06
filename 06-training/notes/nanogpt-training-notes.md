# 训练侧面试笔记：混合精度 / MFU / scaling / 超参 / eager 开销（AR006）

> 全部数字来自 nanogpt-lab 真机实测（results/*.json，Quadro RTX 5000
> sm_75，WDDM）；行号级源码事实来自 llm.c 读码（notes/llmc-internals.md）。
> 六段结构。

## 1. 高频问法

- 混合精度训练怎么工作？为什么 fp16 要 GradScaler、bf16 不要？（→ 2 段 Q1/Q2）
- MFU 是什么？怎么算？多少算正常？（→ 2 段 Q3、3 段卡 5）
- Chinchilla 是什么？固定预算下模型多大合适？（→ 2 段 Q4）
- LR 怎么调？为什么要 warmup？cosine 还是 constant？（→ 2 段 Q5）
- batch size 怎么选？大 batch 一定好吗？（→ 2 段 Q6）
- torch.compile 为什么快？llm.c 这种手写训练码还有什么意义？（→ 2 段 Q7）
- 你训模型遇到过发散吗，怎么处理的？（→ 2 段 Q1 追问、5 段红线 4）

## 2. 追问链

**Q1: fp16 训练为什么会 NaN？GradScaler 到底做了什么？**
→ fp16 最小正规数 ~6e-5，深层网络的小梯度下溢成 0，更新停滞；反向
传播里再乘大数就出 inf/NaN。GradScaler 把 loss 乘一个大常数（如
65536），梯度整体上移进 fp16 可表示区间；优化器步进前检测梯度里
有无 inf/NaN——有则跳过本步并把 scale 减半，无则 unscale 后正常更新。
→ 追问：scale 怎么涨回来的？→ PyTorch 每 `growth_interval`（默认
2000）个干净步把 scale ×2。[本机] E1 里 scaler 65536→131072、
0 次回退——3e-3 的 LR 也没触发 inf。
→ 追问：llm.c 怎么做这件事？→ 不需要框架级 scaler：
`grad_scale` 是 AdamW kernel 的一个标量参数（adamw.cuh L26，
乘进梯度后参与更新），master weights fp32 保精度、低精度参数
写回用 stochastic rounding（L43）——一套方案全在一个 kernel 里。

**Q2: bf16 和 fp16 差在哪？为什么新卡都推 bf16？**
→ 两者都是 16bit：fp16 是 5 位指数 + 10 位尾数（范围小、精度高），
bf16 是 8 位指数 + 7 位尾数——**指数位与 fp32 相同，动态范围等同
fp32**，所以不需要 GradScaler，训练更省心。
→ 追问：那为什么你的 bf16 更慢？→ **是否有 BF16 Tensor Core 取决于
架构**：sm_75（Turing）没有，bf16 走软件仿真。[本机] E2 实测
bf16 147k tok/s，比 fp32 255k 慢 0.57×；fp16 682k = 2.67× fp32。
「新精度更快」的隐含前提是硬件原生支持。

**Q3: MFU 怎么算？**
→ 分子 = 模型前向+反向的真实 FLOPs：`flops/token = 6N + 12·L·H·Q·T`
（PaLM §2.1；N 取 non-embedding 参数，6 = fwd 2 + bwd 4；第二项是
attention，Q=head_dim）。分母 = GPU 峰值 FLOPs × 时间。
→ 追问：为什么分子不是 2N？→ 前向每权重 2 FLOP、反向约 2 倍前向
（dgrad+wgrad 各一遍），所以 6N。
→ 追问：你机器上 MFU 多少？→ [本机] 10.67M 模型 23.9%（fp16，
峰值 89.2 TF）；固定预算 scaling 实验里 MFU 随规模 7.5→29.9% 单调升
——小模型 launch-bound，每步固定几百个 launch，算术强度上不去。
→ 追问：llm.c 的 MFU 公式一样吗？→ 主体同（train_gpt2.cu L1143
`6N + 6LCT`），attention 项差 2×（causal 因子口径），对 GPT-2 口径
差 ~13%，都在这类估计量的固有粗糙度内；另外 llm.c 的峰值表
（mfu.h gpu_db）**没有任何 Turing 卡**，本机直接返回 -1「don't know」
——所以我们 lab 自算峰值（89.2/11.2 TF 按精度选）。

**Q4: 模型越大越好吗？**
→ 数据够才成立。Chinchilla：给定算力预算，最优 token/参数 ≈ 20:1。
→ 追问：你实测呢？→ [本机] E3 固定 25M token 预算训 4 档模型：
val loss 0.5M=1.755、**3M=1.576（最优）**、10.5M=1.912、25M=2.063
——25M 参数只见 25M token = 1 tok/param，深度欠训练，大模型反而
更差。**这是 regime 物理，不是 bug**。
→ 追问：这个结果和你预期不符怎么办？→ 原预注册门写的是「val 随规模
单调降」，实验证伪了它——按诚实修订协议改门（MFU 单调 + 3M 优于
0.5M + 欠训练确认），修订原因/原门/新门全部写进 results.md 与代码
注释。「与预期不符」本身就是固定预算 scaling 的教学点。

**Q5: LR 怎么选？**
→ 从保守值起，warmup 防止初期大梯度×大 LR 冲飞 Adam 二阶矩估计；
后期退火（cosine）让参数在平坦最小值附近收敛。
→ 追问：实测？→ [本机] E4（3.25M 模型，800 步）：1e-4→3e-3 final
loss 2.42→0.82 单调改善，**3e-3 也没发散**（GradScaler+AdamW 比
想象的稳）；短预算下 constant 常优于 cosine（没有退火收益只有
退火损失）——**短程实验的 schedule 结论不能外推长程**。
→ 追问：为什么不发散？→ AdamW 的自适应步长本身对 LR 不敏感区宽；
发散边界离「手感安全区」很远，扫出来才知道。

**Q6: batch size 怎么选？**
→ 两个方向打架：大 batch 吞吐高（摊薄 launch/带宽固定开销），
但固定 token 预算下更新次数少。真正的概念是 critical batch size：
小于它时增大 batch 几乎不损每步收益（吞吐白赚），大于它开始亏。
→ 追问：实测？→ [本机] E4 等 13.1M token 预算：B16 loss 0.706
（3198 次更新）vs B256 2.411（200 次更新）；吞吐 B256 769k vs
B16 365k tok/s（2.11×）。短预算下小 batch 完胜——先保证更新次数，
再谈吞吐。

**Q7: torch.compile 为什么能加速？那 llm.c 还有意义吗？**
→ eager 每个算子独立 launch + 物化中间结果；inductor 把
elementwise 链融合成少量 kernel、消除大量中间显存交通。
→ 追问：你测出来 eager 的浪费有多大？→ [本机] E5 profiler：
20 步里 GEMM 只占 27%，elementwise/copy 11.3%、attention 9.9%、
layernorm 9.5%、**纯 eager 调度开销 9.1%（4900 次 transpose/20 步
= 245 次/步）**、activation 5.9%——73% 非 GEMM 正是 compile 和
llm.c 想吃掉的部分。
→ 追问：那你为什么没用 compile？→ [本机] 诚实记录：torch 2.5.1
inductor 与本机独立 triton 3.x API 不匹配（`triton_key` ImportError）。
→ 追问：llm.c 相对 compile 的差异化？→ 更激进：整步手写——CE
反向启动融进 loss kernel（logits 永不物化 softmax）、unscale 融进
AdamW kernel、grad accumulation 融进 +=、bias/GELU 融进 cuBLASLt
epilogue（llmc-internals.md §7 有逐项映射）。GEMM 仍是 cuBLASLt
——它不重造 GEMM，只消灭 GEMM 之间的一切。

## 3. 数字卡片（全部 [本机] AR006 实测）

| # | 卡 | 数字 | 上下文 |
|---|-----|------|--------|
| 1 | fp16 加速比 | **2.67×** fp32 | fp16 TC 峰值 89.2 vs fp32 11.2 TF（sm_75） |
| 2 | bf16 陷阱 | **0.57× fp32（更慢）** | Turing 无 bf16 TC → 软件仿真 |
| 3 | 三精度 loss 一致 | 0.727 / 0.729 / 0.728 | fp32/fp16/bf16，混合精度数值健康 |
| 4 | GradScaler 轨迹 | 65536→131072，0 回退 | E1 3000 步；无 inf/NaN |
| 5 | MFU 随规模 | 7.5 / 17.1 / 23.7 / 29.9% | 0.5M→25M，launch-bound 单调升 |
| 6 | E1 过拟合 U 型 | val 1.468@750 → 2.348；train 0.085 | 10.67M 参数 × 58 epochs/854KB 语料 |
| 7 | 固定预算最优内点 | 3M=1.576 < 25M=2.063 | 25M token 预算；25M 模型 1 tok/param（Chinchilla≈20） |
| 8 | LR 阶梯 | 1e-4→3e-3：final 2.42→0.82 | 8 组零发散；3e-3 cosine 最优 |
| 9 | batch 权衡 | B16 loss 0.706 vs B256 2.411；吞吐 365k vs 769k | 等 13.1M token 预算；更新次数 vs 吞吐 |
| 10 | eager 里 GEMM 占比 | **27%**（top5 算子 50.9%） | 其余 73% 是融合对象（llm.c/compile） |
| 11 | transpose 开销 | 4900 次/20 步 = 245 次/步 | eager 调度类开销合计 9.1% |
| 12 | SDPA 后端真相 | flash=False，mem_efficient=True | sm_75 无 flash（需 sm_80+），PyTorch 静默回退 |
| 13 | 吞吐饱和 | B256 769k ≈ E2 同规模 fp16 682k | 大 batch 摊 launch 开销，接近饱和区间 |
| 14 | 步时 vs compute 外推 | 10.5M 实测 55.1ms / 线性预测 43.7ms=1.26×；0.5M 3.98×、3M 1.75× | 实测**高于**零截距外推且模型越小比值越大 → 固定开销占比大 |

## 4. 手写骨架

**骨架 1：AMP + GradScaler 训练循环（面试白板级）**

```python
scaler = torch.GradScaler()                  # fp16 需要；bf16 不需要
for step, (x, y) in enumerate(loader):
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits, y)    # autocast 内只算 loss
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()            # 梯度 × scale，防 fp16 下溢
    scaler.unscale_(optimizer)               # 还原真实梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # unscale 后才 clip！
    scaler.step(optimizer)                   # 内部查 inf/NaN：有则跳过本步
    scaler.update()                          # 干净步 growth ×2；脏步 backoff ÷2
```

要点：`unscale_` 后才能 clip（否则 clip 阈值被 scale 污染）；
`scaler.step` 的 inf-skip 语义；scale 动态调整是框架免费的
「自适应防下溢」。E1 实测：65536→131072、0 回退。

**骨架 2：MFU 计算**

```python
def flops_per_token(model, T):
    cfg = model.config
    N = model.get_num_params(non_embedding=True)   # N 不含 position emb
    L, H, Q = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
    return 6 * N + 12 * L * H * Q * T              # PaLM/nanoGPT 同式
fpt = flops_per_token(model, block)
mfu = fpt * tokens_per_s / (peak_tflops * 1e12)    # 峰值按精度选！
```

要点：6N = fwd 2 + bwd 4；attention 项 = 12LHQT（=12LCT）；峰值
口径必须与训练精度匹配（fp16→TC 峰值、fp32→CUDA 核心峰值），跨
精度不比 MFU 比吞吐。

**骨架 3：warmup + cosine LR**

```python
def lr_lambda(step):
    if step < warmup:
        return step / max(1, warmup)                # 线性热身
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))        # cosine 退火到 0
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

要点：warmup 保护 Adam 二阶矩的初期估计（前几百步梯度噪声大）；
cosine 退火让末端学习率 →0，落在尖锐最小值 vs 平坦最小值的辩论里
属于「平坦派」。E4 教训：短预算（800 步）下 constant 常赢——退火
收益在长程尾部。

## 5. 红线清单

1. **「bf16 比 fp32 快」不成立**——前提是硬件有 BF16 Tensor Core。
   sm_75 实测慢 0.57×（仿真）。同理 TF32 只在 Ampere+ 存在。
2. **跨精度不比 MFU**：分母（峰值）不同，fp32 的 50.7% 和 fp16 的
   17.0% 不可比；跨精度比吞吐（tok/s）。
3. **固定 token 预算下「更大更好」是错的**：Chinchilla ~20 tok/param，
   欠训练 regime 里最优点在内部。预注册的门被证伪时，修订并记录
   ——不要事后凑一个能过的门。
4. **GradScaler 的 scale 回退 ≠ 训练出问题**：偶发 inf 跳步是设计
   行为；连续回退才值得查（LR 过大/数据坏/loss 溢出）。
5. **短程实验的 schedule 结论不能外推**：800 步 constant 赢 cosine，
   但这不意味着预训练也该用 constant。
6. **MFU 分子里的 N 用 non-embedding**：position embedding 不参与
   matmul；weight-tied 的 token embedding 参与输出投影，要算
   （llm.c 注释 train_gpt2.cu L1130-1135 同口径）。
7. **报 MFU 必须说清分母来源**：上游 estimate_mfu 有硬编码陷阱
   （nanoGPT 写死 A100 312 TF；llm.c gpu_db 无 Turing 卡返回 -1）。
   抄上游公式可以，抄上游峰值要查覆盖。
8. **launch-bound regime 的数字不能外推**：0.5M 模型 MFU 7.5%，
   25M 也才 29.9%——小模型世界里「调度开销占比」被放大；真实 LLM
   （GB 级权重）带宽与算力才主导。与 AR005 红线 4 同源。
9. **profiler 归类要防字符串陷阱**：`aten::mm` 不含 "gemm" 字样
   （要匹配 "::mm"/"::mv"/"matmul"）；Windows 上 profiler 的
   activities 必须含 CPU（CUDA-only 直接报错）。归类脚本错一行，
   「GEMM 占比」结论整个作废。

## 6. 60 秒电梯陈述

> 训练侧我在 Turing 上搭了一个 nanoGPT 训练实验室，围绕这台老卡的
> 现实做了一整圈实验：fp16+GradScaler 比 fp32 快 2.67×，而 bf16 反而
> 慢 0.57×——sm_75 没有 BF16 Tensor Core，走软件仿真，这个反例比
> 背十遍「bf16 动态范围大」都深刻。10.67M 模型在 854KB 语料上训
> 58 个 epoch，val 曲线教科书 U 型（1.468@750→2.348）；固定 25M
> token 预算的 scaling 实验里 25M 模型反而最差——1 token/param 的
> Chinchilla 缺口，我当时预注册的门就是被这个结果证伪的，按协议
> 诚实修订并全程记录。超参侧：3e-3 不发散（8 组零发散），等预算下
> B16 完胜 B256（0.706 vs 2.411）——更新次数和吞吐的取舍我拿数据
> 摸过。最有意思的是 profiler 归因：eager 一步里 GEMM 只占 27%，
> 245 次/步的 transpose 纯调度开销占 9.1%——我把这 73% 逐项映射到
> llm.c 的手写 kernel 上读了一遍：CE 反向融进 loss kernel、unscale
> 融进 AdamW、GEMM 交给 cuBLASLt 但 bias 融进 epilogue。SDPA 在
> sm_75 上实际走的是 mem-efficient 不是 flash，torch.compile 因
> triton 版本不匹配没跑起来——都如实记录在实验 JSON 里。

## 交叉引用

- AR002：launch 台阶 25.6-69.4µs（本 AR launch-bound 叙事的底座）
- AR005 红线 4（小模型数字不外推）：本 AR 红线 8 是其训练版
- AR005 诚实修订协议（srs 初版口径错误）：本 AR E3 门修订沿用同套流程
- llmc-internals.md：llm.c 逐 kernel 拆解与 E5 映射清单（F4）
- nanogpt-lab/results.md：全部实验细节 + verify_numbers.py 84 项断言
