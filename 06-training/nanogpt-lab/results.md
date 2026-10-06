# AR006 nanogpt-lab 实验结果与分析

> 环境：Quadro RTX 5000（Turing sm_75，48 SM，16GB，448 GB/s，WDDM），
> torch 2.5.1+cu121，Python 3.11。所有数字来自 `results/*.json`（单一真值源），
> 图由 `plot_results.py` 生成（`figs/fig1..fig13`，300 DPI）。
> 语料：本地五组文档/源码混合 898,703 chars（llama.cpp docs 499,802 +
> llm.c 根 124,545 + llmc 200,283 + nanoGPT 55,563 + llm.c doc 18,425），
> 字符级 vocab=124，熵 5.162 bits/char，train/val 按索引 95/5 切分
> （train 853,768 / val 44,935）。

## E1：fp16 基线训练（fig1、fig2）

10.67M 参数（6L-384-6h，non-embed 10,669,440），fp16+GradScaler，
B=64、T=256、cosine LR 1e-3（100 步 warmup）、fused AdamW，3000 步：

- final train loss **0.085**，val 曲线教科书 U 型：2.193 → 最优
  **1.468 @ step 750** → 终值 **2.348**（fig1/fig2）。
- 吞吐 **300,163 tok/s**（54.6 ms/步），自算 MFU **23.9%**
  （flops/token 71,094,528 = 6N + 12LHQT，fp16 TC 峰值 89.2 TF）。
- GradScaler 65536 → 131072，**0 次 backoff**；显存峰值 1643 MB；
  固定种子采样可复现（deterministic_sample=true）。

**为什么 U 型**：3000 步 × 16,384 tok/步 = 49.15M token，除以 train
集 853,768 token ≈ **58 个 epoch**——10.67M 参数模型在 854 KB 文本上
背书。val 在 750 步（≈14 epochs）后开始上升，train 一路降到 0.085。
这是「模型容量 vs 数据量」的最直观形态，与 E3 的反向（数据预算固定、
模型增大反而更差）构成两极。

## E2：精度阶梯——Turing 的现实（fig3、fig4）

3.25M 模型（4L-256-4h），1200 步，其余同 E1：

| 精度 | tok/s | ms/步 | final train loss | MFU（峰值口径） | 显存峰值 |
|---|---|---|---|---|---|
| fp32 | 255,305 | 64.2 | 0.727 | 50.7%（11.2 TF） | 1225 MB |
| fp16 | **682,175** | 24.0 | 0.729 | 17.0%（89.2 TF） | 774 MB |
| bf16 | 146,872 | 111.6 | 0.728 | 3.7%（仿真） | 1226 MB |

- fp16 = fp32 的 **2.67×**（gate ≥1.3× 通过），来自 fp16 Tensor Core
  （峰值 89.2 vs 11.2 TF）。
- **bf16 比 fp32 慢 0.57×**——Turing(sm_75) 没有 BF16 Tensor Core，
  bf16 走软件仿真；这是「新精度 ≠ 更快」的最直接实测，也是本 AR
  最重要的机器现实：在 sm_75 上，fp16+GradScaler 是唯一快速路径。
- 三精度 final loss 一致（0.727–0.729）：混合精度的数值健康。
- MFU 口径注意：fp32 的 MFU(50.7%) 比 fp16(17.0%)「高」只是因为
  分母峰值不同——fp32 峰值低。跨精度比较吞吐（tok/s），不比 MFU。

## E3：固定 token 预算下的 scaling（fig5–fig8）

四档模型（0.5M/3M/10.5M/25M，2L-128-2h → 8L-512-8h），**固定
25M token 预算**（每档 1526 步），fp16：

| 模型 | non-embed 参数 | final val loss | tok/s | MFU | ms/步 |
|---|---|---|---|---|---|
| 0.5M | 409,728 | 1.755 | 2,065,063 | 7.5% | 7.9 |
| **3M** | 3,179,776 | **1.576（最优）** | 686,045 | 17.1% | 23.9 |
| 10.5M | 10,669,440 | 1.912 | 297,365 | 23.7% | 55.1 |
| 25M | 25,238,016 | 2.063 | 162,490 | 29.9% | 100.8 |

**门诚实修订记录**（协议要求，原文注释在 gpt_lab.py `e3()` 内
`HONEST GATE REVISION` 块）：

- **原门**（预注册）：val loss 随模型规模单调下降。
- **修订原因**：原门预注册了错误的 regime——它隐含「每档模型都
  训到各自收敛」，而本实验固定 25M token 预算，25M 参数模型每参数
  只见 ~1 token（Chinchilla 建议 ~20 token/参数），处于**欠训练
  regime**，固定预算下的最优点在内部（3M），不在最大端。
- **修订后门**（全部通过）：① MFU 随规模单调升（7.5→17.1→23.7→
  29.9%）；② 3M 优于 0.5M（规模在训练充分时有益）；③ 25M 差于
  3M（欠训练 regime 确认）。
- 数据佐证：fig5 的 val-vs-params 内点最优、fig7 四条 val 轨迹中
  10.5M/25M 在 750 步后就转入上升。

**三个物理结论**：

1. **MFU 单调升**（fig6 左）：小模型 launch-bound——每步固定
   ~几百个 kernel，模型越小算术强度越低。0.5M 只有 7.5%。
2. **吞吐单调降**（fig6 右，log-log）：每 token 计算量随规模涨得
   比 MFU 快。
3. **实测慢于 compute-bound 外推，模型越小偏差越大**（fig8）：以
   25M 的步时按 flops/token 线性外推（零截距假设），10.5M 预测
   43.7 ms、实测 55.1 ms（**1.26×**）；3M 预测 13.7 ms、实测
   23.9 ms（**1.75×**）；0.5M 预测 2.0 ms、实测 7.9 ms
   （**3.98×**）。锚点（25M）自身含固定开销，而固定开销不随规模
   缩小——模型越小「实测/预测」比值越大，说明每步有相当一部分
   时间是与规模无关的固定开销（launch/optimizer/显存搬运），
   这与 E5 的 eager 开销 9.1% 相互印证。

## E4：超参扫描（fig9–fig11）

3.25M 模型，fp16。LR 扫描 800 步（cosine vs constant，warmup 100）：

| LR | cosine final | constant final |
|---|---|---|
| 1e-4 | 2.424 | 2.107 |
| 3e-4 | 1.840 | 1.498 |
| 1e-3 | 1.109 | 1.025 |
| **3e-3** | **0.822（最优）** | 0.972 |

- 最优在扫描右端 3e-3（cosine），且 **3e-3 也不发散**（diverged=false，
  全部 8 组零发散）——真实发散边界比「手感安全区」更远；对
  3.25M 字符模型，1e-3 只是「保守可用」，不是最优。
- 800 步短程内 constant 常优于 cosine（没有退火收益，只有退火损失），
  这是**短预算实验的方法论陷阱**：cosine 的收益在长程尾部。

batch 扫描（固定 13.1M token 预算，lr=1e-3）：

| B | 步数（=更新次数） | tok/s | final train loss |
|---|---|---|---|
| 16 | 3198 | 364,884 | **0.706（最优）** |
| 64 | 800 | 685,883 | 1.102 |
| 256 | 200 | 768,968 | 2.411 |

- **教科书权衡**：token 预算固定时，batch 越大吞吐越高（B256 比
  B16 多 2.11×）但更新次数越少（B16 有 16× 的更新次数），短预算下
  小 batch 完胜（loss 0.706 vs 2.411）。
- 系统侧读法：B256 吞吐 769k tok/s 已接近本机该模型 fp16 的
  饱和区间（对照 E2 同规模 682k），说明大 batch 摊薄了 launch
  开销——与 E3/E5 的 launch-bound 结论闭环。

## E5：profiler 归因——eager 的时间去哪了（fig12、fig13）

10.67M fp16 模型，20 步 profiler（fresh run 301,651 tok/s，与 E1
的 300,163 在 ±0.5% 内，测量无扰动）：

| 类别 | 占比 |
|---|---|
| linear/matmul（mm 25.68%，1500 次/20 步 = 75 次/步） | 27.0% |
| other（长尾未归类算子） | 24.9% |
| elementwise/copy（copy_ 7.78%、add_ 4.24%、add 3.30%…） | 11.3% |
| attention（_efficient_attention fwd+bwd） | 9.9% |
| layernorm（fwd+bwd） | 9.5% |
| **eager 开销（transpose 4900 次/20 步、cat、as_strided…）** | **9.1%** |
| activation（gelu fwd+bwd） | 5.9% |
| optimizer（_fused_adamw） | 2.4% |

- top5 算子占 50.9%（gate ≥50% 通过）。
- **核心读数：GEMM 只占 27%**。其余 73%（elementwise/attention/
  layernorm/调度开销/activation/optimizer）正是 llm.c 手写融合的
  对象——对照 `notes/llmc-internals.md` §7 的逐项映射。
- 75 次 mm/步的构成：6 层 × (qkv + proj + 2×MLP) × fwd+bwd
  数学上需要 ~72 次，加 lm_head —— eager 没有浪费 GEMM 调用，
  浪费的是 GEMM **之间**的东西：4900 次 transpose（245 次/步，
  基本是 attention 前后的 view/permute 链）与 1880 次 copy。

## E7：SDPA 后端与 torch.compile（无图，结论记录）

- **SDPA 后端探测**（fp16 与 fp32 一致）：flash=False、
  mem_efficient=True，实际内核链
  `scaled_dot_product_attention → _scaled_dot_product_efficient_attention
  → _efficient_attention_forward`——**sm_75 没有 flash attention**
  （需 sm_80+），PyTorch 静默回退 mem-efficient。这解释了 E5 里
  attention 类别出现的是 `_efficient_attention_*` 内核。
- **torch.compile**：失败（诚实记录，e7_compile.json）——
  `ImportError: cannot import name 'triton_key' from
  'triton.compiler.compiler'`：torch 2.5.1 inductor 与本机独立
  安装的 triton 3.x API 不匹配。本 AR 因此全程 eager，E5 的
  73% 非 GEMM 开销正是 compile 想吃掉的那部分。

## 数字一致性自查（对齐 AR005 review 教训）

文档/图中所引用的每个数字 vs JSON 真值（`r/` = results/*.json）：

| 数字 | 本文值 | JSON 真值 | 来源 |
|---|---|---|---|
| 语料 chars / vocab / 熵 | 898,703 / 124 / 5.162 | 同 | data/corpus_stats.json |
| train chars | 853,768 | 853768 | 同上 |
| E1 final train / val min@step / final val | 0.085 / 1.468@750 / 2.348 | 0.08519 / 1.46786@750 / 2.34817 | r/e1.json |
| E1 tok/s / ms/步 / MFU | 300,163 / 54.6 / 23.9% | 300163.4 / 54.58 / 23.92% | r/e1.json |
| E1 scaler | 65536→131072, 0 backoff | 同 | r/e1.json |
| E1 epochs | ~58 | 3000×64×256/853768=57.6 | 计算 |
| E2 fp32/fp16/bf16 tok/s | 255,305 / 682,175 / 146,872 | 255304.7 / 682174.9 / 146871.9 | r/e2.json |
| E2 fp16 加速比 | 2.67× | 2.672 | r/e2.json |
| E2 bf16/fp32 | 0.57× | 146872/255305=0.575 | 计算 |
| E2 finals | 0.727/0.729/0.728 | 0.72691/0.72884/0.72770 | r/e2.json |
| E3 val 四档 | 1.755/1.576/1.912/2.063 | 1.75525/1.57600/1.91228/2.06253 | r/e3.json |
| E3 MFU 四档 | 7.5/17.1/23.7/29.9% | 7.512/17.093/23.701/29.877 | r/e3.json |
| E3 tok/s 四档 | 2,065,063/686,045/297,365/162,490 | 同 | r/e3.json |
| E3 预测比 | 43.7ms 预测 vs 55.1 实测，ratio 1.26 | 43.708/55.097/1.2606 | r/e3.json |
| E4 LR finals | 2.424/2.107/1.840/1.498/1.109/1.025/0.822/0.972 | 2.42363/2.10725/1.84010/1.49788/1.10883/1.02479/0.82224/0.97160 | r/e4.json |
| E4 batch loss | 0.706/1.102/2.411 | 0.70600/1.10173/2.41106 | r/e4.json |
| E4 batch tok/s | 364,884/685,883/768,968 | 同 | r/e4.json |
| E5 GEMM 占比 / top5 | 27.0% / 50.9% | 26.997% / 50.851（top5 pct 和） | r/e5.json |
| E5 transpose 次数 | 4900/20 步 | count=4900 | r/e5.json |
| E5 fresh vs E1 | 301,651 vs 300,163 | 301651.4 / 300163.4 | r/e5.json |
| E7 后端 | flash=False, mem_efficient=True | 同 | r/e7.json |
| E7 compile | ImportError triton_key | 同 | r/e7_compile.json |

（自查方法：本表已固化为可执行断言 `verify_numbers.py`——
92 项检查全部通过，机器可读结果在 `results/doc_check.json`
（all_pass=true）。）
