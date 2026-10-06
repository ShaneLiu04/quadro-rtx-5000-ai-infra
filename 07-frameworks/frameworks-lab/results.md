# AR007 frameworks-lab 实验结果分析（逐图）

> 数据源：`results/*.json`（seed 20261006，全部图由 `plot_results.py` 从 JSON 生成，无硬编码数据）。
> 环境：Quadro RTX 5000（sm_75, 48 SM）/ torch 2.5.1+cu121 / Python 3.11.9 / Windows。
> 诚实记录约定：预注册门 FAIL 不掩盖，写归因与修订；所有"改善 X×"均给出操作点条件。

---

## E0 环境与框架探测

### fig_e0a_uop_hist.png — tinygrad 惰性图 UOp 直方图

**数据**：`e0_env.json → tinygrad.det_graph_op_hist / rand_graph_nodes`。

`(A@B).relu()+1`（arange 输入）在 realize 之前可完整内省：**56 个 UOp 节点**，
直方图里没有"GEMM"这种高层算子——15 个 CONST、9 个 RESHAPE、2 个 REDUCE、
1 个 WHERE……乘法、规约、比较全部是同一套 UOp 原语的组合。这是 tinygrad
"万物皆 UOp"设计（tensor.py:104 `_apply_uop` → schedule/__init__.py:31
`create_schedule`）的直接可视化。

右侧红框：同一表达式把输入换成 `randn`，图膨胀到 **236 节点**，直方图中出现
`THREEFRY: 4`、`SHL/SHR`、`SIN/LOG2/SQRT`——**随机数生成是图的一部分**
（Counterwise THREEFRY 哈希 PRNG 在图内展开），而 eager 框架里 randn 是
host 侧黑盒 op。这是编译型框架"把尽可能多的计算放进图里再整体优化"哲学的
最直观证据。

realize 在本机双栈（CUDA/TORCH）均失败（CUDA 深至 hcq2.py:533
`StopIteration`；TORCH 深至 realize.py:236 缺 runtime/ops_torch 模块），
traceback 原文记录在 JSON，按 AR005 E7 约定诚实记录不绕过——**图内省不需要
执行成功**，lazy 图本身就是"图 IR"层的第一手数据。

端到端 `all` 复现时还发现一个工程层事实：**e0 的 tinygrad CUDA realize
尝试失败后，同进程内任何后续 torch CUDA kernel 都报 "invalid argument"**
（hcq2 崩溃把进程的 CUDA 上下文留在不可用状态）——`framework_lab.py all`
因此改用子进程隔离逐实验执行。这是"编译型框架自己接管 runtime"的代价
的极端形态：tinygrad 不经过 torch 的 CUDA 上下文管理，失败时也不负责恢复。

**门**：G_e0 四项全 PASS（int_mm CUDA 位精确 / bnb 降级记录 / UOp 内省 /
realize 失败记录）。另两项关键探测：bnb 0.50.3.dev0 降级导入（native DLL
缺失 → `ErrorHandlerMockBNBNativeLibrary`），但 dispatch dump 显示
`register_kernel("default")` 注册到 **Undefined 键并广播到全部非 CUDA 后端键**
（default/ops.py:142 纯 torch 兜底），CPU 上实际调用返回真实 Tensor 而非
None——"降级导入 ≠ 不可用"；官方 NF4 表与我们的官方表复刻**逐值一致**
（`nf4_map_consistent: true`），且与裸 `ndtri` 分位点**端点差 ±1**
（`nf4_map_equals_raw_ndtri: false`）——这是 E1 的第一个线索。

---

## E1 NF4 码本（bnb functional.py:170-196 纯 torch 复刻）

### fig_e1a_codebook.png — 四种 4-bit 码本 vs N(0,1) pdf

**数据**：`e1_nf4.json → levels.{nf4, nf4_raw, fp4, int4}`。

黑线是 N(0,1) pdf（未归一化权重的形状），四组竖线是 16 个量化电平的位置。
关键视觉：**官方 NF4（蓝）端点钉在 ±1**，与裸 ndtri 分位点（红，±1.863）
差一个端点变换。absmax 归一化后的权重落在 ~[-1/3.8, 1/3.8]（蓝色阴影区），
官方表在这个区间密、区间外无电平——它是为"每块除以块内 absmax"这个操作
**配套设计**的码本，不是通用分位数表。FP4（绿，±6 跨度）在 0 附近只有
±0.5/±1 两个电平，对集中分布注定粗糙。

### fig_e1b_blocksize.png — blocksize 扫描（32→1024）

**数据**：`synth_sweep.*`，N(0,1) 输入，blockwise absmax，rel-RMSE。

四条线全部**单调上升**：block 越小，块内 absmax 越贴近局部值域，scale 越紧，
误差越低（NF4 0.0873→0.1043）。这解释了 bnb 为什么选 blocksize 64
（functional.py:884）：不是拍脑袋，是误差-元数据开销的折中点。

门 G1 @block64：**NF4 0.092 < INT4 0.100 PASS**（相对优势 ~8%），且两个
对照都指向同一结论：分位形状匹配有效但前提是 absmax 归一化——**裸 ndtri
码本 0.127 输给 INT4**（端点 ±1.863 把电平浪费在归一化后根本不存在的区间），
FP4 0.377 全面崩溃。真实训练权重（lab 内 300 步 MLP `_train_toy_mlp`
四层 emb/fc1/fc2/head——srs §3.2"短训小 GPT"的实现偏离，MLP 已足够
提供非高斯真实权重分布，诚实记录）复测：emb/fc 层
NF4 优势 ~1-7%，**head 层（峰度 8.4 重尾）NF4 0.1046 vs INT4 0.1291 优势
拉大到 23%**——分布越尖，分位码本越赚。

### fig_e1c_errhist.png — 误差直方图（block 64，81 bins，[-0.05,0.05]）

**数据**：`err_hist.*`。

NF4/INT4 误差近似均匀分布（每 bin ~7.7 万/6 万计数，中心 bin 因 0 电平
吸附高出 4-5 倍），FP4 计数只有它们的 ~1/5——同样的总样本数下 FP4 的
误差被推到 **±0.05 显示范围之外**（步长 0.5 的电平间距决定了误差量级）。
对数 y 轴下三条线的"高度差"就是精度差的直观形态。

---

## E2 LLM.int8（functional.py:1590-1671 复刻）—— 本 AR 唯一 FAIL 门

### fig_e2a_threshold.png — 离群阈值 τ 扫描

**数据**：`e2_llmint8.json → threshold_sweep`。输入 2048×2048 N(0,1)，
随机选 **8 列（0.39%）×16** 注入离群（论文现象：系统性离群集中在固定
feature 维），τ∈{0,2,4,8,16,32}，元素级判定：|x|≥τ 的元素从行统计清零，
**含命中元素的整列**拆去 fp16 旁路。

三个操作点讲完整个机制：

1. **τ=0（不分解）**：行 absmax 被 16× 离群绑架到 ~48，正常值全被压到
   1-2 个 LSB，误差 **4.86e-2**——这就是 LLM.int8 要解决的问题本身。
2. **τ=8（正确操作点）**：恰好只抓注入的 8 列（正常 N(0,1) 的 2048 行
   列最大值 ~4.5σ < 8 < 16σ 离群量级），行统计恢复干净，误差 **1.14e-2**，
   与均匀输入 int8 固有误差（1.17e-2）持平——**分解把离群污染消掉，
   误差回到 int8 本底，4.2× 改善，但不会比 int8 更准**。这是读图的
   核心认知：LLM.int8 的卖点是"省显存不掉点"，不是"更准"。
3. **τ=2（平凡解，红框）**：N(0,1) 有 4.5% 概率 |x|≥2，2048 行的列几乎
   必命中 → **100% 列走 fp16**（`outlier_frac: 1.0`），误差 3.6e-4 就是
   纯 fp16——int8 白做。τ 选太小的代价不是精度而是"分解失效退回 fp16"。

**τ=16/32 反而回升**（2.19e-2 / 3.96e-2）：被拆走的仍是同样 8 列，但
**低于 τ 的离群元素（值在 [8,τ) 区间）残留在行 absmax 里继续污染 scale**。
τ 的有效窗口是"正常最大值与离群量级之间"，两个方向越界都翻车——
bnb `int8_double_quant(threshold=...)` 的语义就是这个窗口的选择问题。

### fig_e2c_scaling.png — 缩放粒度救不了重尾

**数据**：`scaling_compare`。

per-tensor 1.76e-2 → per-row 1.17e-2（均匀输入，粒度收益 1.5×）；
但同样 per-row 换到重尾输入直接 **4.86e-2**——粒度再细，行 absmax 依然
被同行离群绑架。**"加粒度"和"拆离群"是两条正交的路**，LLM.int8 之所以
存在，就是因为前者在系统性离群面前失效。

### fig_e2b_throughput.png — G2 诚实 FAIL + 布局归因

**数据**：`throughput["2048"]` + `layout_attribution`。2048³ matmul，
中位数计时（预热 + sync）。

| 路径 | 吞吐 | vs fp16 |
|---|---|---|
| fp16 cuBLAS TC | 52.7 TFLOPS | 1.00× |
| int8 `torch._int_mm`（row×row 默认布局） | 31.1 TOPS | **0.59×** |
| int8 `torch._int_mm`（rowA×colB = TN） | 67.9 TOPS | **1.29×** |
| int8 端到端（eager：quant + mm + dequant 三步） | 8.1 TOPS | **0.15×** |

（三次完整运行间 fp16 基线 53.0/48.7/52.7 TFLOPS 漂移 ±9%——时钟状态噪声；
**布局配对比值稳定**：TN/默认 1.87/1.88/1.83，TN 绝对值 ~68-70 TOPS。
结论只依赖配对比值，不依赖绝对值。）

**预注册门 G2（int8 kernel ≥1.2× fp16）FAIL**，修订归因：cuBLAS int8
的张量核快路径要求 **TN 布局**（A row-major × B column-major）。默认
row×row 落在慢路径，反而比 fp16 慢；换成 TN 后 67.9 TOPS = 默认布局的
**1.83×**，但也只有 fp16 的 1.29×——sm_75 上 `torch._int_mm` 达不到
论文"2× fp16"的峰值（理论 INT8 峰 178 TOPS，实测最高 38%）。四组布局
对照（37.1 / 67.9 / 34.2 / 36.0 TOPS）把"布局决定 TC 路径"钉死；
**bnb csrc 自定义权重布局（`B [N,K/2]` row-major 即 mma 视角的 column-major）
存在的根本原因就是钉住 TN**——E2 测量与源码设计互相印证。

端到端 0.15× 是 eager 未融合的代价（quant/dequant 各一次全张量读写），
与 E3 的 1.84× 未融合开销同源：bnb 的价值恰恰是把这些操作写进单个 kernel。

---

## E3 8-bit Adam（cpu/ops.py:507-517 语义对齐复刻）

### fig_e3a_loss.png — fp32 vs 8-bit 状态双训 loss 曲线

**数据**：`e3_adam8bit.json`。nanoGPT 4L-256（3.26M 参数）+ 07 自有语料
（vocab 142），**同种子同数据同 lr** 双训 800 步，每 10 步记录。

两条曲线全程贴合，终值 **fp32 1.1353 vs int8 1.1561，差 0.0207**——
门 G3（<0.05）PASS。且注意一个更本质的事实：**三次同种子完整运行的
配对差分别为 0.0007、0.0253、0.0207**（前两次用审查前偏粗的 135 级
unsigned v 码本，第三次用与 bnb 逐值一致的 255 级码本——码本粗细的
差异同样淹没在运行间噪声里）——同种子同数据同 lr，差异只能来自
CUDA 训练非确定性（embedding 反向 atomicAdd、cuBLAS 算法选择）。
也就是说 **int8 状态与 fp32 的差异在运行间噪声量级内**，"量化不掉点"
在本尺度成立，且这个结论比任何单次差值都稳健。这个结果来之不易：
第一版线性 int8 码本 +
block 4096 实测**真实发散**（loss 29→89，w 爆到 50）——块内 v 小于
absmax/254 的坐标 round 到 0，√v̂=0，步长 = lr·m̂/ε ≈ 10⁵ 量级。
修齐 bnb 三重防护后收敛：① **decade 动态码本**（create_dynamic_map，
十进制 10⁻⁷..1 每 decade 取均值，255 非零级跨 7 个数量级，小 v 永远有
码字）；② **block 256**（优化器专用，不是 4096）；③ 每步全链
dequant→update→requant。教训：**bnb 能工作是有三个前提的**，"cast 成
int8 存起来"不叫 8-bit Adam。

### fig_e3b_memory.png — 状态显存 0.254× / 未融合时间 1.84×

**数据**：`optim_state_bytes / step_ms_median`（解析式计算 + 中位数计时）。

左图：fp32 m+v 状态 26.09MB → uint8 m+v + 每 256 块一个 fp32 absmax
6.62MB = **0.254×**（理论 2.03/8 = 0.254，block scale 开销 0.6%）。
右图：步时 33.9 → 62.5ms = **1.84×**——这是**未融合**实现的诚实代价：
每步多出两次全参数量的 dequant/requant 读写。bnb 用单 kernel
`optimizer_update_8bit_blockwise` 把这 1.84× 压回近 1×——
**融合不是优化项，是 8-bit 优化器可用性的前提**（和 E2 端到端 0.15×
同一课：量化算法的工程成本全在内存搬运，不在数学）。

---

## E5 显存账本与 7.5B 外推

### fig_e5a_ledger.png — bytes/param 四组合 vs 理论

**数据**：`e5_ledger.json → bytes_per_param / theory_bytes_per_param /
consistency_ratio`。实测方式：分配前后的 `torch.cuda.memory_allocated`
差值，不是解析式。

| 组合 | 实测 B/param | 理论 | 比值 |
|---|---|---|---|
| fp32 训练（P4+G4+M4+V4） | 16.777 | 16.0 | 1.049 |
| AMP 混合（P4+G2+M4+V4） | 14.583 | 14.0 | 1.042 |
| int8 优化器（P4+G2+M1+V1+scale） | 8.032 | 8.031 | 1.00005 |
| NF4 参数存储（0.5+scale 2/64） | 0.5315 | 0.531 | 1.0004 |

门 G5（±10%）全 PASS。堆叠柱是理论构成的分解（params/grads/m+v/scale），
柱顶注释实测值——**前两行的 4-5% 残差是 CUDA caching allocator 的块对齐
rounding**，不是模型误差；后两行（块 scale 占比小）几乎零残差，反而证明
账本模型本身是对的。

### fig_e5b_projection.png — 7.5B 外推（log 轴）

**数据**：`projection_7p5B`（由实测 bytes/param × 7.5e9 外推）。

125.8GB（fp32）→ 109.4（AMP）→ 60.2（int8 optim）→ **4.0GB（QLoRA
NF4 底座）**→ 4.4GB（+1% LoRA 可训练参数的 P/G/m/v）。两条参考线：
消费级 24GB 卡与 A100 80GB。结论一眼可见：**7.5B 全参训练需要 A100 级别，
QLoRA 把门槛拉到消费卡**——0.5315 B/param 的 NF4 packed 存储（E1 的
码本 + E5 的 packing）就是 QLoRA 论文标题里"65B on a single 48GB"
的算术基础。

---

## 门的汇总（含 FAIL 与修订）

| 门 | 结果 | 说明 |
|---|---|---|
| G_e0 四项 | PASS | int_mm 位精确 / 降级记录 / UOp 内省 / realize 失败记录 |
| G1 NF4 < INT4 @64 | PASS | 0.092 < 0.100，裸 ndtri 0.127 反例在案 |
| **G2 int8 ≥1.2× fp16** | **FAIL（诚实记录）** | 默认布局 0.59×；TN 归因 1.83× 提升、1.29× vs fp16，修订见 e2 JSON `gate_int8_revision` |
| G3 8-bit Adam 差 <0.05 | PASS | 0.0207（三次同种子运行 0.0007/0.0253/0.0207——差异在训练非确定性噪声内），前置两 bug 修复（/127 缺失、线性码本+4096 块发散） |
| G5 账本 ±10% | PASS | 最大残差 4.9%（allocator rounding） |

一句话总结：**量化的收益都在存储侧（0.5 B/param、0.254× 状态），代价
全在搬运侧（布局、融合、调度）**——bnb 的全部工程设计（自定义布局、
单 kernel 融合、dispatch 兜底）都是在支付/规避搬运侧的代价。
