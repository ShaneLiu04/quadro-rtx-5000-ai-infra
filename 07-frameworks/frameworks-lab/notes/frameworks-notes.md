# 量化框架面试笔记：bnb / LLM.int8 / NF4 / 8-bit Adam / 图编译（AR007）

> 全部数字来自 frameworks-lab 真机实测（results/*.json，Quadro RTX 5000
> sm_75，torch 2.5.1+cu121，bnb 0.50.3.dev0 克隆降级导入）；行号级事实来自
> bnb/tinygrad 读码（notes/bnb-internals.md、notes/tinygrad-internals.md）。
> 六段结构。

## 1. 高频问法

- QLoRA 的 NF4 为什么比 INT4 好？双重量化是什么？（→ 2 段 Q1）
- LLM.int8 的离群分解到底解决什么问题？阈值怎么选？（→ 2 段 Q2）
- 8-bit Adam 为什么不发散？直接把状态 cast 成 int8 行吗？（→ 2 段 Q3）
- int8 推理一定比 fp16 快吗？（→ 2 段 Q4，本机反直觉实测）
- 训练 7.5B 到底要多少显存？QLoRA 怎么塞进消费卡？（→ 2 段 Q5）
- 量化框架（bnb）在系统栈里处于什么位置？和图编译器什么关系？（→ 2 段 Q6）
- torch 自定义 op 的多后端注册怎么工作？（→ 2 段 Q6 追问）

## 2. 追问链

**Q1: NF4 为什么比 INT4 好？**
→ NF4 = 对 N(0,1) 分位匹配的 16 级**查表码本**（不是浮点编码）；bnb 的
absmax 分块归一化把权重压进 [-1,1]，NF4 级在中心密集。[本机] N(0,1) 实测
rel-RMSE：NF4 **0.092** < INT4 0.100 < FP4 0.377（E1）。
→ 追问：为什么不直接用 ndtri 分位数？→ **官方表 ≠ 裸分位数**：裸 ndtri 表
跨度 ±1.86，而 absmax 归一化后数据实际住 [-1/3.8, 1/3.8]（本机 absmax 均值
3.80）——裸分位数浪费大半码字，实测 **0.127 反而输给 INT4**（E1）；官方表
钉死 ±1 端点正是为归一化分布设计的（bnb functional.py:772-806，E0 逐值比对
一致）。
→ 追问：FP4 为什么那么差？→ e2m1 只有 1 位尾数，0 附近格点太稀；N(0,1)
权重集中在 0 附近 → 0.377（E1）。NF4 的本质是**把码字花在数据实际住的地方**。
→ 追问：blocksize 怎么选？→ [本机] 扫描 32→1024：NF4 0.0873→0.1043 单调
升——**块越小误差越低**（每块独立 absmax 适配局部尺度）；bnb 默认 64 是
误差/元数据（每 64 个权重一个 fp16 scale=0.031 B/param）的折衷（E1）。
→ 追问：真实训练权重上也成立吗？→ [本机] 300 步 MLP 权重：fc 层（峰度 2.8≈
高斯）NF4 0.906 vs INT4 0.945（×10⁻¹）；**head 层峰度 8.4 重尾，NF4 优势最大
0.1046 vs 0.1291**——越重尾越赚（E1 weights 表）。

**Q2: LLM.int8 的离群分解解决什么？**
→ 行 absmax 量化的死穴：一行里一个 |x|=50 的离群值把 scale 拉到 50/127，
其余小值全被压到 1-2 个 LSB。**粒度救不了**：per-row 在重尾输入下误差
4.86e-2，和 per-tensor 差不多量级。[本机] E2。
→ 追问：那怎么救？→ 把含离群的**整列**拆出来走 fp16（论文：系统性离群
集中在固定 feature 维），其余 int8。[本机] 2048 列中 8 列 ×16 离群：τ=0
误差 4.86e-2 → **τ=8 时 1.14e-2（4.2×），回到 int8 本底**（E2；均匀输入
int8 固有 1.17e-2——分解只是把离群污染消掉，不会比 int8 更准）。
→ 追问：阈值怎么选？→ [本机] τ 扫描 {0,2,4,8,16,32}（元素级判定，
含 ≥τ 元素的整列走 fp16，且这些元素先从行统计里清零）：
**τ 必须落在正常最大值（~4.5σ）和离群量级（16σ）之间**。τ=2 是平凡解
（N(0,1) 有 4.5% |x|≥2，2048 行的列几乎必中 → 100% 列走 fp16，int8 白做）；
τ=8 恰好只抓注入的 8 列（0.4%）；τ=16/32 反而回升（2.19e-2/3.96e-2）——
低于 τ 的离群元素残留在行 absmax 里继续污染 scale。bnb 的
`int8_double_quant(threshold=...)`（functional.py:1590）语义即此。
→ 追问：int8 均匀输入的固有误差？→ [本机] 1.17e-2（vs fp16 3.6e-4）——
int8 的 ~1% 是精度天花板，LLM.int8 的卖点是**省显存不掉点**，不是更准。

**Q3: 8-bit Adam 为什么不炸？**
→ 先说会炸的版本：线性 int8 状态（127 级）+ 4096 大块。[本机] 实测训练
发散（loss 29→89，w 爆到 50）：块内 v 小于 absmax/254 的坐标 round 到 0 →
√v̂=0 → 步长 = lr·m̂/ε ≈ 10⁵ 级。
→ bnb 的三重防护 [源码]：① **decade 码本**（create_dynamic_map，
functional.py:296-348）：十进制 10⁻⁷..10⁻¹ 每 decade 线性取均值，255 个
非零级**跨 7 个数量级**——小 v 永远有码字；② **block 256**（cpu/ops.py:490，
不是通用的 4096）：块内同质，动态范围紧；③ 每步全链
dequant→update→requant。
→ [本机] 修齐语义后：与 fp32 Adam 同种子双训 800 步（nanoGPT 4L-256，3.2M），
**终值 loss 差在训练噪声内**（三次同种子运行配对差 0.0007 / 0.0253 /
0.0207，门 <0.05 三过——同种子差异只能来自 CUDA 非确定性：embedding
反向 atomicAdd、cuBLAS 算法选择；前两次用审查前偏粗的 135 级 unsigned
v 码本、第三次用与 bnb 逐值一致的 255 级——码本粗细同样淹没在噪声里）；
优化器状态
26.09MB→6.62MB = **0.254×**（E3）。
→ 追问：开销？→ 我们未融合的实现 33.9→62.5ms/步（1.84×）；bnb 用**单
kernel 融合**（optimizer_update_8bit_blockwise）把这 1.84× 压回去——
这正是"库的价值 = 把量化开销融合到看不见"。

**Q4: int8 一定比 fp16 快吗？**
→ [本机] 反例：`torch._int_mm` 默认 row×row 布局 **0.59× fp16**
（31.1 vs 52.7 TFLOPS）——cuBLAS int8 TC 快路径要求 **TN 布局**；
换列主序 B 后 **0.253ms = 67.9 TOPS = 1.29× fp16**（1.83× 提升）
（E2 layout_attribution；跨运行 fp16 基线 53.0/48.7/52.7 ±9% 时钟
抖动，但布局**配对比值**稳定 1.87/1.88/1.83，结论只看配对）。
→ 追问：为什么到不了理论 2×？→ Turing INT8 TC 理论 178 TOPS，实测 67.9
= 38%；fp16 达成率 59%——int32 epilogue 写带宽 2× + cuBLAS int8 kernel
在 sm_75 的成熟度；**bnb 的 igemmlt 自定义权重布局就是为了钉死 TN**，
和 E2 实测互证（csrc 拆解）。
→ 追问：端到端呢？→ [本机] 量化+反量化全程 8.1 TOPS（0.15× fp16）——
**eager 下的量化开销吃掉全部收益**；bnb 把 dequant 融进 GEMM kernel
（gemm_4bit_sm75.cu 的 dequant_byte 在 smem 写入时完成）才是净赢路径。
呼应 AR005：「4bit 权重 ≠ 推理提速，融合反量化才提速」。

**Q5: 训练 7.5B 要多少显存？**
→ [本机] 逐组合实测 bytes/param（E5）：fp32 全链 **16.78** / AMP 混合
（fp32 master+fp16 grad+fp32 m/v）**14.58** / int8 优化器（fp32 P+fp16 G+
8bit m/v）**8.03** / NF4 底座（4bit 打包+fp16 块 scale）**0.5315**。
→ 7.5B 外推：**125.8GB → 109.4GB → 60.2GB → 4.0GB**（QLoRA 底座）+1%
LoRA 适配器 ≈ 4.4GB——**消费级 24GB 卡可跑 7.5B 微调**的账本。
→ 追问：为什么实测比理论高 ~5%？→ CUDA caching allocator 的段取整
（fp32 组合 1.049×）；账本公式 + allocator 行为都要算（E5 consistency）。
→ 追问：QLoRA 训练时梯度放哪？→ 底座冻结零梯度零状态，只有 LoRA 参数
有 m/v——**量化省的不只是存储，还有整个优化器状态的乘数**。

**Q6: bnb 在系统栈什么位置？tinygrad 这种图编译器呢？**
→ bnb = **op 级 dispatch 层**：torch.library 三件套（define/register_fake/
register_kernel）注册同名 op 的多套实现；[本机] dispatch dump 实测
"default" 注册落 **DispatchKey::Undefined → 向所有非 CUDA 后端键广播**——
所以降级导入（native DLL 缺失）时纯 torch 兜底在 CPU 照样能跑
（ErrorHandlerMock 使 import 不炸，cextension.py:393）。
→ tinygrad = **全程序图编译**：Tensor 是 UOp 图句柄（tensor.py:104，零物化），
执行 = 图重写（pattern matching）→ codegen → renderer 出 PTX。[本机]
`(a@b).relu()+1` 确定性图 56 节点；randn 图 236 节点——**PRNG 也是图节点**
（THREEFRY）；本机 realize 失败点 = 编译器深处（hcq2.py:533 encode 阶段
StopIteration；TORCH 后端文件干脆不存在）——**编译型框架的失败模式在
"图→机器码"边界，离用户代码 6 层**，与 eager 完全不同（E0）。
→ 追问：和 torch.compile/Inductor 什么关系？→ 同属"把 eager 的隐式优化
显式化成图变换"：Inductor = trace→FX→Triton 源码；tinygrad = 单一 UOp IR
走到底；bnb 根本不是图框架，是精度/硬件双维度的 op 特化层。

## 3. 数字卡片（全部 [本机] AR007 实测）

| # | 卡 | 值 | 出处 |
|---|---|---|---|
| 1 | NF4 vs INT4 vs FP4 rel-RMSE @N(0,1),block64 | **0.092 / 0.100 / 0.377** | e1 |
| 2 | 裸 ndtri 分位码本（官方表的对照） | **0.127（输给 INT4！）** | e1 |
| 3 | blocksize 32→1024 误差趋势 | NF4 0.0873→0.1043 单调升 | e1 |
| 4 | 重尾 head 层 NF4 优势 | 0.1046 vs 0.1291（峰度 8.4） | e1 |
| 5 | LLM.int8 均匀输入固有误差 | 1.17e-2（fp16 基准 3.6e-4） | e2 |
| 6 | 离群分解：τ=0→8 | 4.86e-2 → **1.14e-2（4.2×，回 int8 本底）**，8/2048 列=0.4% | e2 |
| 6b | τ=2/16/32 反例 | τ=2 全列 fp16（平凡解）；τ=16/32 回升 2.19e-2/3.96e-2（行统计残留污染） | e2 |
| 7 | 缩放粒度对比（均匀/重尾） | per-tensor 1.76e-2 / per-row 1.17e-2 / 重尾 per-row 4.86e-2 | e2 |
| 8 | int8 TC 布局效应 | row×row **0.59×** fp16 → TN **1.29×**（67.9 TOPS；配对 1.83× vs 默认） | e2 |
| 9 | 端到端 int8（eager 未融合） | 8.1 TOPS = 0.15× fp16 | e2 |
| 10 | 8bit-Adam 收敛差 | **三次同种子运行 0.0007/0.0253/0.0207（训练噪声内）**；状态显存 **0.254×** | e3 |
| 11 | 8bit-Adam 步时开销（未融合） | 33.9→62.5ms = 1.84× | e3 |
| 12 | bytes/param 全家桶 | 16.78 / 14.58 / 8.03 / **0.5315** | e5 |
| 13 | 7.5B 显存外推 | 125.8 / 109.4 / 60.2 / **4.0GB** | e5 |
| 14 | _int_mm bit-exact | CUDA 与 CPU 逐元素 int32 精确 | e0 |
| 15 | UOp 图规模 | det 56 节点 / randn 236 节点（PRNG 入图） | e0 |

## 4. 手写骨架

**8-bit Adam 状态（bnb 语义核心 12 行）**——面试手写高频：
```python
# quant:  x -> (q=uint8 码, absmax=块绝对值最大)   dequant: qmap[q]*absmax
# m: signed dynamic map; v: unsigned; block=256; 每步:
m = qmap_m[m_q.long()] * m_abs_blk          # dequant（decade 码本, 非 /127）
v = qmap_v[v_q.long()] * v_abs_blk
m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g*g # 更新
m_q,m_abs = requant(m); v_q,v_abs = requant(v)   # requant（round 到码本）
denom = (v.sqrt()/sqrt(1-b2**t)).add_(eps)  # bnb 排列 cpu/ops.py:514
p -= lr/(1-b1**t) * m/denom                 # bnb 排列 cpu/ops.py:517
```
写不对的三种死法（全部 [本机] 踩过）：dequant 忘 /127（127×/步爆炸）；
线性码本替代 decade（v 下溢归零 → 步长 10⁵）；块选 4096（动态范围失控）。

**LLM.int8 前向（离群分解版）**：
```python
qX,sX,ocols,_ = vw_quant_rows(X, tau)   # 行 absmax; |x|>=tau 的列剔除
qW,sW,_,_ = vw_quant_rows(W, 0.0)
C = torch._int_mm(qX, qW.t()).float() * (sX[:,None]*sW[None,:]) / 127**2
if ocols is not None:
    C += (X[:,ocols].half() @ W[:,ocols].t().half()).float()  # 离群列 fp16 旁路
```

## 5. 红线清单

1. **NF4 官方表不是裸 ndtri 分位数**——端点钉 ±1 是为 absmax 归一化分布
   设计；说"分位数查表"可以，别把两个表混为一谈（实测差 0.035 rel-RMSE）。
2. **别把 int8 状态 cast 当 8-bit Adam**——没有 decade 码本+小块，v 下溢
   归零会把步长放大到 10⁵ 级（实测发散）；bnb 能工作是有三个前提的。
3. **"int8 = 2× fp16" 是峰值不是现实**——cuBLAS int8 有布局前提（TN），
   默认布局实测**慢** 0.59×；说加速前先说布局。跨运行绝对吞吐 ±9%
   时钟抖动，结论只依赖布局**配对比值**（1.83-1.88 稳定）。
4. **量化误差 ≠ 端到端误差**——int8 均匀输入固有 1.17e-2；LLM.int8 的
   分解只把离群污染消掉、误差**回到 int8 本底**（4.2×），不是 fp16 量级；
   论文说"不掉点"指下游指标。阈值区间（正常 max 与离群量级之间）选错
   两个方向都翻车：τ=2 平凡解全 fp16，τ>离群量级行统计继续被污染。
5. **编译型框架的失败在编译深处**——tinygrad 本机 realize 失败在
   hcq2.py:533（离用户代码 6 层）；诚实记录 traceback，不臆测不绕过。
6. **降级导入 ≠ 不可用**——bnb 无 native DLL 时 default 纯 torch 兜底
   全后端广播（dispatch dump 实测）；"能不能跑"要看 dispatch 表不是 import。
7. 沿用全局红线：数字必须 JSON 可溯源；预注册门被证伪 → 诚实修订记录
   （本 AR G2 int8 ≥1.2× 被证伪 → 布局归因修订，见 results/e2_llmint8.json）。

## 6. 60 秒电梯陈述

> 「我在 Turing 卡上拆过 bitsandbytes 的量化框架：把 NF4/LLM.int8/8-bit
> Adam 三个算法全部用纯 torch 复刻并与 bnb 官方码本逐值对齐，然后真机测
> 了每个营销数字的成立条件——NF4 比 INT4 好 8% 的前提是 absmax 归一化
> （裸分位数反而更差）；LLM.int8 的离群分解把重尾误差从 4.9e-2 压回
> int8 本底 1.1e-2（4.2×），本质是 'per-row 救不了系统性离群'，且阈值
> 要卡在正常 max 与离群量级之间；8-bit Adam 不发散靠 decade 码本+256 小块，
> 我先用线性 int8 复刻真实地炸了一次再修对，三次同种子运行终值差
> 0.0007/0.0253/0.0207（在训练非确定性噪声内）、状态
> 省 3.9×；int8 比 fp16 慢还是快由 cuBLAS 布局决定，我用四组布局实验
> 把 TN 快路径钉出来——bnb 自定义权重布局存在的根本原因。最后把
> 16.78→0.53 bytes/param 的显存账本外推到 7.5B：QLoRA 底座 4GB，
> 消费卡可跑。」

## 交叉引用

- 源码级事实：notes/bnb-internals.md（dispatch 链/四算法地标/sm75 kernel）、
  notes/tinygrad-internals.md（图编译管线/失败点）
- 实验数据：results/e0_env.json ~ e5_ledger.json；逐图分析 results.md
- 上游互证：AR005「4bit≠提速，融合才提速」（E2 端到端 0.15× 佐证）；
  AR006「sm_75 无 bf16 TC」（csrc 派发树 bf16→SIMT 佐证）
- 总纲：01-foundations/notes/INTERVIEW-INDEX.md（本讲已并入 §1/§5/§6）
