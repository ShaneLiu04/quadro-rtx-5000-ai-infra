# [AR007] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR007 |
| AR 主题 | quant-frameworks-lab（bitsandbytes 量化框架拆解 + LLM.int8/NF4/8bit 优化器复刻 + tinygrad 图编译器拆解） |
| 关联 SR | 无（学习型 AR，Infra 面试准备第 7 站） |
| 日期 | 2026-10-06 |
| 状态 | Draft → Active |

## 1. 背景与目标

AR001-006 覆盖了执行模型、GEMM、DSL、推理量化、训练。本 AR 补上
**框架层**：kernel 之上还有一层「决策软件」——bitsandbytes 决定
「同一 op 在不同硬件/精度下走哪条实现」（torch.library 多后端注册、
cextension 降级、sm_75 专用 gemm_4bit kernel），tinygrad 决定
「一张算子图如何被调度、融合、降级成机器码」。两条线都以
**真机实验 + 源码拆解（file:line）** 双轨进行，全部面向面试：
量化训练（LLM.int8/NF4/8bit Adam）与图编译（lazy tensor/调度/代码生成）
都是 2026 年 infra 岗高频考点。

**本机关键事实（决定实验设计，探针已验证）**：
- Quadro RTX 5000 = Turing sm_75：INT8 TC = 2× FP16 TC 峰值；
  `torch._int_mm` CUDA 路径已验证 bit-exact 可用
- bnb 克隆可导入但**降级**（libbitsandbytes_cuda121.dll 缺失 →
  `ErrorHandlerMockBNBNativeLibrary`）；`torch.library.register_kernel`
  的 "default" 注册含**纯 torch 参考实现**（int8_vectorwise_quant/
  quantize_blockwise/quantize_4bit 等）——降级态下 CPU 调用返回
  (None,None,None) 的 dispatch 机制本身是一个实验对象
- tinygrad 可导入、lazy UOp 图可构造可内省，但 **realize 在本机全挂**
  （CUDA 栈 hcq2.py:533 encode 阶段 / TORCH 栈 realize.py:236 均抛
  StopIteration）——按 AR005 E7 惯例：诚实记录 + 源码拆解
- bnb csrc 存在 **gemm_4bit_sm75.cu**：上游为本机架构写的专用 kernel，
  拆解它 = 「框架如何为你的 GPU 特化」

## 2. 需求范围

**In Scope：**
- bnb 0.50.3.dev0 克隆（sys.path 导入，零改动）源码拆解：
  cextension.py 加载/降级链、_ops.py torch.library 注册模式、
  backends/{default,cuda,cpu,triton} 分发、functional.py 四大算法
  （NF4 / LLM.int8 / blockwise 8bit / 8bit optimizer）、
  csrc gemm_4bit_sm75.cu
- tinygrad 克隆（零改动）图编译管线拆解：lazy UOp → schedule →
  codegen → renderer，UOp 图内省实验，realize 失败诚实记录
- E0-E5 六条实验线（见 §3）+ ≥11 图 + results.md + F4/F5 笔记
- 自包含小语料（bnb/tinygrad/nanoGPT 源码文本，07 lab 内自建）

**Out of Scope：**
- bnb CUDA 二进制编译（无 MSVC/nvcc）——只拆源码
- tinygrad realize 修复（上游 bug/兼容性不在本 AR 修）
- pip install bitsandbytes（克隆版行为即研究对象）
- Triton backend 实跑（Windows 无 triton）
- 多 GPU

## 3. 功能需求

### 3.1 F1 — E0 环境基线与降级机制解剖

**描述：** 三项基线：bnb 降级导入 + dispatch 表 dump；torch._int_mm
CUDA/CPU 能力验证；tinygrad lazy 图构造 + realize 失败记录。
**触发条件：** `python framework_lab.py e0`。
**期望行为：** bnb 导入路径、ErrorHandlerMock 判定、
`torch._C._dispatch_dump` 关键 op 的 kernel 表落 JSON；
`torch._int_mm` 与 fp32 参考逐元素比对（int8 路径 bit-exact 判定）；
CPU AVX512 能力（bnb cpu backend 的 `torch._int_mm` 门控条件）；
tinygrad UOp 图 op-type 直方图 + realize 两栈 traceback 全文落 JSON。
**验收标准：**
- Given 降级 bnb，When 调 int8_vectorwise_quant(CPU)，Then (None,None,None)
  现象与 `_dispatch_dump` 表一起落 JSON，dispatch 机制结论附 file:line
  **[实测修订 2026-10-06]**：(None,None,None) 仅在早期探针中偶现且不可
  复现——受控 E0 的稳定行为是纯 torch default 兜底在 CPU 上正确返回
  `(Tensor, Tensor, None)`（e0_env.json `cpu_call_all_none: false`）；
  早期现象归因为首导入 `__pycache__` 字节码编译期的部分初始化瞬态。
  验收以"现象+机制结论落 JSON"为准，具体返回值以实测为准。
- Given int8 A@B（K≤1024），When torch._int_mm vs fp32 参考，Then
  int32 结果逐元素相等（bit-exact）
- Given tinygrad `a@b+relu` lazy 图，When 内省，Then UOp 节点数与
  op-type 直方图落 JSON（不 realize）

### 3.2 F2 — E1 NF4 信息论与 blocksize

**描述：** 读 functional.py L170-196 量化分位映射；纯 torch 复刻 NF4
编解码（blockwise absmax + NF4 码本 bucketize，对齐
backends/default/ops.py L233+ 语义）；实验：NF4 vs FP4 vs INT4 均匀
在 N(0,σ) 权重上的量化误差、blocksize {32..1024} 扫描、真实权重
（lab 内短训小 GPT 得到）上的端到端误差。
**[实现偏离记录 2026-10-06]**："短训小 GPT" 实现为 lab 内 300 步
4 层 MLP（`_train_toy_mlp`：emb/fc1/fc2/head）——MLP 已足够提供
非高斯真实权重分布（head 层峰度 8.4），E3 的 nanoGPT 训练另行覆盖
"GPT 权重"场景；诚实记录，不改验收数值标准。
**验收标准：**
- Given N(0,1) 权重，When NF4 vs INT4 均匀量化，Then NF4 相对 RMSE
  更低（预注册门；若反转则如实分析——「NF4 最优性有条件」正是
  面试素材）
- Given blocksize 扫描，Then 误差曲线落 JSON/图且单调趋势解释
- 码本图：NF4 16 级 vs N(0,1) pdf 叠加（分位匹配可视化）

### 3.3 F3 — E2 LLM.int8 复刻（向量级缩放 + 离群分解）

**描述：** 按 functional.py L1590-1671 + backends/default/ops.py
L64-177 语义复刻：row/col 向量级 absmax 量化、`torch._int_mm` int8
矩阵乘、双缩放反量化；离群阈值 τ∈{0,1,2,4,8} 稀疏分解（离群列走
fp16，其余 int8）重尾激活实验；GPU 实测 int8 vs fp16 吞吐。
**验收标准：**
- Given 无离群均匀输入，When LLM.int8 复刻 vs fp16 参考，Then 输出
  相对误差落 JSON（预期 ~1e-2 量级，量化固有）
- Given 重尾输入（对数正态激活模拟），When τ 从 0→8 扫描，Then
  误差随离群列分离增加而下降的曲线落 JSON/图
- Given 大 GEMM（M=N=K≥2048，CUDA），When int8 vs fp16 计时，Then
  吞吐比落 JSON；预注册门：int8 ≥ 1.2× fp16（Turing 2× 峰值扣除
  量化开销；WDDM 噪声用多轮中位数；不达门则诚实修订分析）
- 向量级 vs 逐张量 vs 逐行缩放误差三方对比落图

### 3.4 F4 — E3 8-bit Adam：块级 int8 优化器状态

**描述：** 读 functional.py optimizer 路径 + backends/cpu/ops.py
L469-580（`_optimizer_update_8bit_blockwise_cpu` 参考）；纯 torch 复刻
Adam8bit（m/v 每步 blockwise int8 量化存、反量化用，4096 默认块）；
lab 自建语料短训小 GPT：fp32 状态 vs int8 状态 loss 曲线 + 显存实测
（torch.cuda.memory_allocated 差值）+ 吞吐开销。
**验收标准：**
- Given 同配置同种子短训（≥800 步），When fp32-Adam vs int8-Adam，
  Then loss 曲线两者落 JSON 且终值差 < 0.05（预注册门；不达则如实
  分析量化噪声对优化的影响）
- 优化器状态显存：int8 状态实测占用 ≈ fp32 的 1/4（±10%）
- int8 路径每步额外量化开销的时间占比落 JSON

### 3.5 F5 — E4 框架源码拆解（bnb dispatch + tinygrad 管线，只读）

**描述：** 双线 file:line 笔记：
(a) bnb：cextension.py L359/393 加载失败链与 ErrorHandlerMock、
_ops.py torch.library.define/register_kernel/register_fake 三件套、
backends/ 按设备注册与 default 纯 torch 兜底、cpu backend 的
AVX512+torch≥2.6 门控（ops.py L26）、functional.py 四算法地标、
**csrc gemm_4bit_sm75.cu**（Turing 专用 4bit GEMM：kernel 内反量化
+ mma.sync 路径）与 gemv/gemm 分工；
(b) tinygrad：tensor.py lazy UOp 图（不物化）、engine/schedule.py
调度、codegen、runtime/graph/cuda.py 渲染、hcq2.py L533 与
realize.py L236 两个本机失败点的管线定位。
**验收标准：**
- 每条结论附 文件:行号；不编造未运行验证的运行时行为（拆解结论
  与 E0 实测对齐的必须对齐）
- 「框架 dispatch 决策树」（bnb）与「图编译七步管线」（tinygrad）
  两张文字流程图落笔记

### 3.6 F6 — E5 显存账本（实测 + 外推）

**描述：** torch.cuda.memory_allocated 实测小模型下：fp32/fp16 参数
+ Adam 状态全组合 vs 8bit 优化器 vs 4bit NF4 参数（含块缩放）；
外推 7.5B 参数表（fp32 训练 vs 混合 vs QLoRA 全链路）。
**验收标准：**
- Given 实测组合矩阵，Then 显存数字落 JSON 且与理论值 ±10% 互洽
- 7.5B 外推表在 notes 中给出公式与代入过程

### 3.7 F7 — 图表与文档

**描述：** ≥11 图：E0 UOp 直方图；E1 码本-vs-pdf、blocksize 曲线、
误差直方图；E2 τ 扫描、int8/fp16 吞吐、缩放方式对比；E3 loss 双线、
优化器显存；E5 账本堆叠柱 + 外推表。results.md 逐图中文分析；
notes/bnb-internals.md（F4）、notes/frameworks-notes.md（F5 六段
面试笔记）、lab README、INTERVIEW-INDEX 增补、plot_results.py +
verify_numbers.py（文档-JSON 数字一致性对账）。
**验收标准：**
- 每图有标注/单位/参考线，无空图；verify_numbers.py 全绿
- 笔记数字与 JSON 一致（AR005 review 教训沿用）

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 上游完整性 | 文件 | bnb/tinygrad 克隆零改动（mtime 核查） |
| 环境 | 全局 python | 不 pip install；无网络；无 MSVC/nvcc |
| GPU 真实性 | sm_75 | 实验数据真机实测；无法运行的诚实记录 |
| 可重复性 | 基准 | JSON + 一键复现；计时多轮中位数（AR002-006 纪律） |
| 时间预算 | 全套 | E0-E5 总 wall time ≤ 40 分钟（训练线 ≤ 8 分钟） |
| 面试导向 | 笔记 | 六段结构 + [实测]/[源码] 分级 + 题卡 |

## 5. 约束与假设

- bnb/tinygrad 均经 sys.path 导入克隆，零改动；bnb 仓库 CLAUDE.md 的
  worktree/PR 规则是贡献者工作流，本 AR 只读拆解，不适用（记录规避）
- 假设：torch 2.6+ 且无 AVX512 时 bnb cpu backend 的 _int_mm 门控
  关闭（E0 实测确认本机归属）
- 假设：WDDM 噪声对小 kernel 计时显著，吞吐实验用 ≥2048 维 GEMM +
  多轮中位数
- tinygrad realize 失败是上游/环境兼容问题，本 AR 不修不绕，作为
  「图编译器失败模式」的诚实教学样本
- 诚实修订机制：预注册门（F2 NF4 优于 INT4、F3 int8 ≥1.2× fp16、
  F4 终值差 <0.05）被证伪时如实记录修订（AR005 惯例）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| NF4 | 4-bit NormalFloat：对 N(0,1) 分位数匹配的 16 级非浮点码本 |
| LLM.int8 | 向量级 absmax int8 量化 + 离群特征 fp16 稀疏分解 |
| 双重量化 | QLoRA：对 4bit 量的缩放再量化一次省显存 |
| blockwise | 按 64/4096 元素块各自 absmax 的量化粒度 |
| UOp | tinygrad 的统一算子 IR 节点（lazy 图的基本单元） |
| register_fake | torch.library 的 meta/fake 实现，服务 compile tracing |
| ErrorHandlerMock | bnb native 加载失败后的占位 lib，使 import 不炸 |
| QLoRA | 4bit NF4 冻结底座 + LoRA 微调，bnb 作者提出的组合 |
