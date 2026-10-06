# GPU Kernel 面试索引（INTERVIEW-INDEX）

> 面向面试的跨讲总纲：主题→讲次映射、必背数字卡、手写题清单、项目陈述模板、自测清单。
> 六篇笔记各自末尾的「面试深潜」章节是分讲弹药库；本文件是总装与复习入口。
> 事实口径与笔记正文一致（已经过 SDD review + ST 抽查）。

## 1. 主题 → 讲次映射（面试官视角）

| 面试主题 | 主战场 | 弹药位置 | 一句话锚点 |
| --- | --- | --- | --- |
| CUDA 执行模型 / 索引 / guard | L002/L003 | L002 §6.1-6.2、L003 §7.1 | `blockIdx*blockDim+threadIdx` + guard + cdiv |
| 内存层级 / shared memory | L003/L004 | L003 §7.1 Q2、L004 §7.1 Q3 | block 内共享、64KB/SM、协作装载 |
| syncthreads 语义 / 死锁 | L004 | L004 §7.1 Q4、§7.2 追问1 | 生产消费两道栅栏；早退 = 死锁 |
| kernel fusion | L004 | L004 §7.1 Q1 | 省访存不省 FLOPs |
| launch 开销 / 计时纪律 | L001/L004 | L001 §8.1 Q1、L004 §7.1 Q2 | 异步模型；空 kernel µs 级下限 |
| occupancy / 资源算术 | L004 | L004 §7.1 Q5 | 三类资源相除取最小 |
| coalescing / 维度映射 | L003/L004 | L003 §7.1 Q1、L004 §7.2 追问4 | 连续内存给 threadIdx.x |
| tiled matmul | L004 | L004 §7.1 Q3 + §7.4 手写骨架 | 全局读 k → k/16 |
| FlashAttention / online softmax | L012 | L012 §9.1 Q1-Q3 + §9.4 | 不物化 S；递推三要点 |
| shared vs register 布局 | L012 | L012 §9.1 Q4 | 广播进 shared，私累加进寄存器 |
| register spilling 诊断 | L012 | L012 §9.1 Q5 | `.local` depot + ncu L2 |
| Triton vs CUDA / 块级编程 | L014 | L014 §10.1 Q1-Q2 | program=块；mask 是边界契约 |
| swizzling / L2 命中 | L014 | L014 §10.1 Q3 | pid 重排；90→54 块读 |
| autotune / torch.compile 关系 | L014/L001 | L014 §10.1 Q4、L001 §8.1 Q3 | compile 生成的就是 Triton |
| 工具链（profiler 三层） | L001 | L001 §8.1 Q2 | nsys 缩范围 → ncu 定瓶颈 |
| 错误处理（sticky error） | L002 | L002 §6.1 Q3 | API 不抛异常；错误粘滞 |
| GEMM 优化线（naive→tiling→regblock） | **gemm-lab** | gemm-lab-notes §1-§3 + §5.1 Q1-Q5 | 三级跳 0.44→2.04 TFLOPS；AI=T/4、BN/4 |
| GEMM vs cuBLAS 差距构成 | **gemm-lab** | gemm-lab-notes §5.1 Q4、fig4/fig6 | 向量化+双缓冲+swizzle，4.6× 工程差距 |
| tensor core / FP16 路径 | **gemm-lab** | gemm-lab-notes §5.1 Q5、fig5 | mma=64 FMA/指令；同卡 FP16=FP32×5.8 |
| roofline / 算术强度定位 | **gemm-lab** | gemm-lab-notes §2、fig6 | ridge 25 FLOP/B；L2 是模型盲区 |
| 基准方法学（显示 GPU 降频） | **gemm-lab** | gemm-lab-notes §5.4 红线 4-6 | 冷启动漂 25%；持续 warmup+中位数压到 4% |
| GEMM 形状空间（skinny/wave/GEMV） | **gemm-sweep** | gemm-sweep-notes §1 Q1-2、§3.1 | GEMV 带宽平台 86% vs 算力平台；拐点 M*≈51 |
| split-K / 库启发式边界 | **gemm-sweep** | gemm-sweep-notes §1 Q3-4、§3.2 | 饥饿上限 0.93 vs 实测 8.2 TF = 库已自动 split；手工全败 |
| batch GEMM vs flat | **gemm-sweep** | gemm-sweep-notes §1 Q5、§3.2 | flat 稳赢 +13%@B=64；组间调度是纯开销 |
| CUTLASS tile 层级 / 模板参数 | **gemm-sweep** | cutlass-turing-dissection §2-§4、gemm-sweep-notes §3.4 | 15 参数组装；NumStages=2=48KB/64KB 内存预算 |
| layout 效应（TN 神话） | **gemm-sweep** | gemm-sweep-notes §1 Q6、§3.3 | 7 组合极差 5%；ldmatrix+swizzle 吃掉布局差异 |
| FP16 时钟态敏感性 | **gemm-sweep** | gemm-sweep-notes §3.3、红线 6 | 跨会话 54.9~67.4（±10%）；FP32 大形状 ±1.5%；噪声带 ∝ 测量窗口 |
| Triton 三方对照（launch/带宽） | **triton-lab** | triton-notes §2.1-2.3、fig1/fig2 | launch 台阶 25.6/45.6/69.4 µs；大 N 三方同为 82-84% HBM |
| 教程 kernel vs 最优 | **triton-lab** | triton-notes §2.2、fig3/fig4 | softmax 差 1.2-1.6×（bar.sync×占用率）；naive=流量 4× |
| Triton 达成率 / autotune 必要性 | **triton-lab** | triton-notes §2.4-2.5、fig6/fig7 | FP32=87% of cuBLAS；config 随 N 迁移；spread 2.67× |
| lowering 决定论（fp16 无 TC） | **triton-lab** | triton-notes §2.6-2.7、fig8/fig9 | 26 config mma=0；fp16 比 fp32 慢；TC 差 9.5×；数 PTX 不信文档 |
| WDDM 编译/计时分离 | **triton-lab** | triton-notes §5.4 红线 2 | 编译掉频把 cuBLAS 打到 3.44 TF；修复后 9.95（+189%） |
| 混合精度训练 / GradScaler | **nanogpt-lab** | nanogpt-training-notes §2 Q1-Q2 | fp16 梯度下溢→scaler；bf16 免 scaler 但 sm_75 仿真反慢 0.57× |
| MFU / FLOPs 公式 | **nanogpt-lab** | nanogpt-training-notes §2 Q3、§4 骨架2 | 6N+12LHQT；峰值按精度选；跨精度不比 MFU |
| scaling / Chinchilla | **nanogpt-lab** | nanogpt-training-notes §2 Q4 | 固定 25M token 预算最优在 3M；门被证伪→诚实修订 |
| LR / batch 超参 | **nanogpt-lab** | nanogpt-training-notes §2 Q5-Q6 | 3e-3 零发散；等预算 B16 loss 0.706 vs B256 2.411 |
| eager 开销 / 训练融合（llm.c vs compile） | **nanogpt-lab** | nanogpt-training-notes §2 Q7 + llmc-internals §7 | eager 一步 GEMM 只占 27%；四类融合点逐项映射 |
| SDPA 后端选择 | **nanogpt-lab** | nanogpt-training-notes §3 卡 12 | sm_75 无 flash→静默回退 mem-efficient |
| NF4 / QLoRA 码本设计 | **frameworks-lab** | frameworks-notes §2 Q1、bnb-internals §3/§5 | 官方表≠裸分位数（0.092 vs 0.127）；端点钉 ±1 为 absmax 归一化 |
| LLM.int8 / 离群分解 | **frameworks-lab** | frameworks-notes §2 Q2、e2 τ 扫描图 | per-row 救不了系统性离群；τ 卡在正常max与离群量级之间，误差回 int8 本底（4.2×） |
| 8-bit Adam / 量化优化器 | **frameworks-lab** | frameworks-notes §2 Q3/§4 骨架、bnb-internals §4 | decade 码本+block256+全链；线性 int8 会炸（实测）；int8 与 fp32 差在训练噪声内（三次运行 0.0007/0.0253/0.0207） |
| int8 吞吐 / cuBLAS 布局 | **frameworks-lab** | frameworks-notes §2 Q4、e2 layout 图 | TN 才走 TC：0.59×→1.29×（配对 1.83×）；bnb 自定义布局因此存在 |
| 训练显存账本 / QLoRA 外推 | **frameworks-lab** | frameworks-notes §2 Q5、e5 堆叠柱图 | 16.78→0.53 B/param 实测；7.5B QLoRA 4.0GB |
| 框架 dispatch / torch.library 多后端 | **frameworks-lab** | bnb-internals §1-§2、e0 dispatch dump | default 注册→Undefined 键全后端广播；降级可跑 |
| 图编译器 / tinygrad 管线 | **frameworks-lab** | tinygrad-internals §1-§3、e0 UOp 直方图 | 万物皆 UOp（PRNG 入图）；失败在编译深处 |

## 2. 必背数字卡

### 2.1 本机（Quadro RTX 5000 / sm_75 / Turing）——说出这些 = 真机上写过 kernel

| 项 | 值 | 面试用途 |
| --- | --- | --- |
| SM 数 / CUDA core | **48** / 3072（64/SM） | occupancy 口算、规模感 |
| Tensor Core | 384（第 2 代，FP16/INT8/INT4） | Turing 无 BF16/FP8 |
| 显存 / 带宽 | 16 GB GDDR6 / **448 GB/s** | 访存 bound 天花板 |
| L2 | 4 MB | swizzle/L2 话题 |
| shared / SM | **64 KB** | tile 大小口算起点 |
| 线程上限 | 1024/SM，1024/block，warp=32 | grid 配置 |
| 算力 | FP32 **11.2 TFLOPS**；FP16 TC **89.2 TFLOPS**；FP64=FP32/**32** | roofline：拐点 25 vs 199 FLOP/B |
| H2D | PCIe 3.0 ≈ **12 GB/s** | 「拷贝比算贵」 |

### 2.2 通用（任何 GPU 面试都能用）

| 项 | 值 |
| --- | --- |
| kernel launch 开销 | 个位数 µs（空 kernel 实测 4~6µs） |
| grid.x 上限 2³¹ / y,z 上限 2¹⁶ | 每 block ≤ 1024 线程 |
| warp | 32 线程锁步；coalescing = 32 连续 4B → 128B 事务 |
| shared bank | 32 bank × 4B（stride-1 无冲突；padding 破坏冲突） |
| softmax 数值稳定 | 减行最大值再 exp（max-subtraction trick） |

## 3. 手写题清单（按出现频率排序）

| # | 题 | 得分点（漏一条扣一档） | 出处 |
| --- | --- | --- | --- |
| 1 | **vector add** | 索引公式 / guard / cdiv / 拷贝方向 / 错误检查 | L002 §6.1 Q2 |
| 2 | **树状归约（block 求和）** | 全员写 shared / sync / stride 减半 / 每步 sync / 线程 0 写出 / 越界补 0 | GPU-Puzzles P10/P12 |
| 3 | **softmax（含数值稳定）** | 减 max / exp / 求和 / 除法；追问：块间怎么办 → online softmax | L012 §9.1 Q2 |
| 4 | **tiled matmul** | 双 `__shared__` tile / 协作装载+补零 / 两道 syncthreads / 非整除验证 | L004 §7.4 |
| 5 | **online softmax 递推** | m 先更新 / 双系数同源 / `L=m+log(l)` / 对拍 O 和 L 两样 | L012 §9.4 |
| 6 | （加分）Triton 逐元素 kernel | program_id / arange / masked load(other) / masked store / num_warps | L014 §10.4 |
| 7 | **regblock GEMM**（tiled 进阶） | 线程管 TM×TN 输出 / 累加器进寄存器 / 读一次供 TM×TN 次 FMA / 补零 guard / 两道栅栏 | gemm-lab-notes §4 |

## 4. 项目陈述模板（把这套练习讲成项目经历）

### 4.1 60 秒版（「介绍一个你做过的 GPU 项目」）

> 「我系统性地做过一轮 CUDA kernel 工程：14 道题从线程索引一路写到分块 GEMM，
> 全部在 Turing 卡上真实编译运行，并与 numpy 参考实现对拍。配套我写了一个
> **双通道验证框架**——GPU 通道走 numba 真机执行；CPU 通道是一个用 Python 线程
> 和 Barrier 栅栏实现的 CUDA 执行模型模拟器，`cuda.shared.array` 映射为 block 级
> 共享数组、`syncthreads` 映射为栅栏等待。它不只是回退方案——它逼我把
> syncthreads 的栅栏语义理解到能**实现**的程度：比如我验证了『syncthreads 放在
> 非 uniform 分支会死锁』这件事，是靠模拟器的栅栏超时检测机制做到的。」

### 4.2 三分钟展开版（按追问深入）

- **背景（S）**：想在真机（sm_75）上建立 CUDA 手感，而不是只看讲义。
- **任务（T）**：14 道 Numba CUDA 题 + 一个可重复、可回退的验证体系。
- **行动（A）**：
  - kernel 层：索引/guard → shared 协作装载 → 树状归约 → 前缀和 → 分块 matmul（困难情形 6 次全局读）；
  - 验证层：GPU 真机通道 + CPU 栅栏模拟通道 + **负向自检**（故意写错的 kernel 必须被检出）；
  - 纪律层：所有 kernel 的 syncthreads 放在 uniform 路径、shared 全员显式初始化
    （GPU 上 shared 未初始化，不能依赖零值）——这两条同时是 GPU 正确性与模拟可行性的前提。
- **结果（R）**：17 个用例双通道全绿；负向用例、环境异常（强制 GPU 不可用 → 明确报错
  退出码 2）、栅栏死锁超时全部有自动化验证。

### 4.3 可被追问的深度点（提前备好）

1. 「CPU 模拟器的栅栏为什么语义正确？」→ `threading.Barrier` 内部条件变量保证
   唤醒可见性；GIL 下共享数组的写读无撕裂；block 间顺序执行因只写 out 的不相交位置。
2. 「为什么不直接在上游 notebook 里填空？」→ 上游不可 import（含 `!pip` 头）、
   不可自动化回归；独立 solutions + verify 让 17 用例成为可重复的 UT。
3. 「sm_75 跑这些有什么特别的？」→ 64KB shared/1024 线程上限是口算 occupancy 的
   真实约束；FP64 是 FP32 的 1/32，double 字面量的坑在这张卡上惩罚最大。

## 5. 高频追问阶梯（Ladder 汇总——面试官的连环问路线图）

```
计时 ──→ 为什么 time.time 不行 ──→ Event/synchronize ──→ warmup ──→ launch 开销多大
索引 ──→ 2D 怎么映射 ──→ 哪维给 x ──→ coalescing 是什么 ──→ 不连续会怎样
shared ──→ 为什么要它 ──→ 谁能看见 ──→ syncthreads 什么时候调 ──→ 早退线程会怎样（死锁）
matmul ──→ naive 慢在哪 ──→ tiling 怎么改 ──→ 全局读降多少 ──→ tile 越大越好吗（occupancy 口算）
GEMM 深潜 ──→ 寄存器分块省哪层 ──→ AI 相同为何差 62% ──→ cuBLAS 还差 4.6× 是什么 ──→ FP16 tensor core 又是 6×（gemm-lab-notes §5.2）
形状空间 ──→ 方阵外推 skinny？──→ GEMV 瓶颈是带宽 ──→ 拐点 M* 怎么算（AI=25）──→ M/N skinny 不对称（gemm-sweep-notes §2）
split-K ──→ 何时需要 ──→ 怎么验证库已做（饥饿模型 0.93 vs 实测 8.2）──→ 手工为何输 ──→ CUTLASS 怎么显式化（gemm-sweep-notes §2）
CUTLASS ──→ 模板参数怎么读 ──→ NumStages 为何 2（48KB/64KB 预算）──→ Turing vs Ampere（无 cp.async 两跳）（cutlass-turing-dissection）
softmax ──→ 数值稳定怎么做 ──→ 分块怎么办 ──→ online 递推 ──→ FA2 改了什么 ──→ L 为什么输出
Triton ──→ 和 CUDA 区别 ──→ program 是什么 ──→ mask/other ──→ swizzle 为什么 ──→ autotune key
Triton 实测 ──→ 能到 cuBLAS 几成 ──→ 87%（FP32 扫描后）──→ fp16 自动走 TC 吗 ──→ 不，数 PTX：mma=0，9.5× 差距 ──→ 源码有路径构建未启用（triton-notes §5.2）
量化格式 ──▸ Q4_K 144B 怎么来 ──▸ 6-bit scale 打包位序 ──▸ 为什么比 Q4_0 好（子块异质性）（llamacpp-notes §4.1）
量化提速 ──▸ 4bit 更快吗 ──▸ 预反量化速度与 fp16 相当（123.3 vs 129.5 t/s）/现反量化慢 9.2× ──▸ mmvq/mmq 融合才净赢（llamacpp-notes §5.1）
KV cache ──▸ 公式 2×L×kvh×seq×hd×2B ──▸ GQA 折扣在 kv_heads ──▸ 实测 0.00%（llamacpp-notes §3 卡5）
decode 吞吐 ──▸ batch↑ 时间恒定 ──▸ 权重读摊销 ×63.6 ──▸ 小模型 launch-bound 距下限 577×（llamacpp-notes §3 卡7/12）
采样 ──▸ top-p 边界 token 保留 ──▸ 温度→截断集 1.7→9.2 ──▸ 32k vocab 下 top-p 比 greedy 贵 62.5×（llamacpp-notes §2/§3）
混合精度 ──▸ fp16 为何下溢 ──▸ GradScaler 放大/inf-skip/backoff ──▸ bf16 为何免 scaler（指数位=fp32）──▸ 但 sm_75 无 bf16 TC，实测反慢 0.57×（nanogpt-training-notes §2 Q1-Q2）
MFU ──▸ 6N+12LHQT 怎么来 ──▸ N 为何 non-embed ──▸ 峰值按精度选 ──▸ 小模型 launch-bound：7.5%→29.9% 单调升（nanogpt-training-notes §2 Q3）
scaling ──▸ 模型越大越好吗 ──▸ Chinchilla ≈20 tok/param ──▸ 固定预算 25M 模型只见 1 tok/param 最差（1.576 vs 2.063）──▸ 门被证伪→预注册修订（nanogpt-training-notes §2 Q4）
超参 ──▸ LR 怎么选 ──▸ 3e-3 也零发散 ──▸ 短预算 constant 赢 cosine（不可外推）──▸ batch=更新次数 vs 吞吐（0.706 vs 2.411）（nanogpt-training-notes §2 Q5-Q6）
训练融合 ──▸ eager GEMM 只占 27% ──▸ transpose 245 次/步占 9.1% ──▸ llm.c 手写什么 ──▸ CE 反向/unscale/grad accum += /epilogue 四融合，GEMM 仍 cuBLASLt（llmc-internals §7）
NF4 码本 ──▸ 为什么比 INT4 好 ──▸ 分位匹配+absmax 归一化 ──▸ 官方表≠裸 ndtri（实测 0.092 vs 0.127）──▸ blocksize 越小误差越低（0.087→0.104）（frameworks-notes §2 Q1）
离群分解 ──▸ int8 重尾为什么崩 ──▸ 行 absmax 被离群值绑架（per-row 4.86e-2）──▸ 整列拆 fp16 ──▸ τ 卡在正常max(~4.5σ)与离群量级(16σ)之间：τ=8 误差 4.2× 回 int8 本底；τ=2 平凡解全 fp16（frameworks-notes §2 Q2）
8-bit Adam ──▸ 直接 cast int8 行吗 ──▸ v 下溢→步长 10⁵ 爆炸（实测发散）──▸ decade 码本 7 数量级+block256 ──▸ 修齐后与 fp32 差在训练噪声内（三次同种子 0.0007/0.0253/0.0207）/状态 0.254×（frameworks-notes §2 Q3）
int8 布局 ──▸ int8 一定快吗 ──▸ 默认布局 0.59× fp16（慢！）──▸ TN 快路径 1.29×（配对 1.83× vs 默认）──▸ bnb 自定义权重布局就是钉 TN（frameworks-notes §2 Q4）
显存账本 ──▸ 7.5B 训练多大 ──▸ 16.78 B/param 全 fp32=125.8GB ──▸ int8 optim 60.2GB ──▸ QLoRA 底座 4.0GB 消费卡可跑（frameworks-notes §2 Q5）
框架栈位 ──▸ bnb 是什么层 ──▸ torch.library 多后端 op dispatch ──▸ default 注册 Undefined 键全后端广播 ──▸ 降级导入纯 torch 兜底照样跑（bnb-internals §2）
图编译 ──▸ tinygrad 和 eager 区别 ──▸ 万物皆 UOp（randn=236 节点 PRNG 入图）──▸ 执行=图重写+渲染 PTX ──▸ 失败在编译深处（hcq2.py:533 StopIteration）（tinygrad-internals）
```

## 6. 考前自测清单（模拟面试 pass/fail 标准）

- [ ] 60 秒讲清 CUDA 执行模型（SM/block/warp/thread + shared 语义）——不看稿
- [ ] 默写 vector add 全套（kernel + host 五步）无遗漏
- [ ] 口算 occupancy：64KB shared、1024 线程上限、16×16 tile → 各允许几个 block（32/4）
- [ ] 默写 online softmax 递推四行，说清两个 exp 系数为什么同源
- [ ] 说出 tiled matmul 两次 syncthreads 各防什么、越界线程为什么不能早退
- [ ] 报出本机 5 个数字：48 SM / 448 GB/s / 64KB shared / 11.2 TFLOPS / 1024 线程
- [ ] 讲清 fusion 省的是什么（带宽，不是 FLOPs）
- [ ] 报出 GEMM 三级跳实测：0.44 → 1.03 → 2.04 TFLOPS，说出 AI=T/4、BN/4 与 ridge=25 的关系
- [ ] 说清 K1_T32 与 K2a「AI 相同性能差 62%」的原因（片上流量+ILP，不是 DRAM）
- [ ] 说清 torch.compile / Triton / CUDA 的选型顺序与理由
- [ ] 项目陈述 60 秒版脱稿完成，含「CPU 栅栏模拟器」亮点
- [ ] 说出 GEMM 形状空间三个数：饱和区 9.5-10.5 TF / 64³ 地板 0.54 / GEMV 带宽 86%（且能报对指标——带宽区不报 TFLOPS）
- [ ] 讲清「库已自动 split-K」的验证链（饥饿上限 0.93 vs 实测 8.2 TF），及手工全败的三个原因
- [ ] 口算 CUTLASS 08 的 shared 预算：2×(128×64+64×256)B=48KB → NumStages=2 是 64KB 硬限的必然
- [ ] 报出 triton 三方对照五个数：launch 25.6/45.6/69.4 µs、大 N 82-84% HBM、FP32 达成 87%、fp16 mma=0、TC 差 9.5×
- [ ] 说清「fp16 不走 TC」的验证链（PTX 计数 mma=0 + cvt.f32.f16 + fma=BM×BN×BK/threads），及「源码有≠构建启用」红线
- [ ] 口算 KV cache：70B（80L/64h/8kvh/fp16/4096）单序列 ≈ 1.34GB，说清 GQA 折扣在 kv_heads
- [ ] 报出量化五数：Q8_0 1.88×、Q4_K 3.54×（4.5bpw）、Δppl −0.01/+0.98、144B=2+2+12+128、on-the-fly 慢 9.2×
- [ ] 说清「4bit 权重≠推理提速」的完整证据链（预反量化相同 → 现反量化 66.9ms/步 → triton 融合 264GB/s → mmvq 内联）
- [ ] 说清 triton flat dequant 的反直觉坑：每 program 1 块只有 16GB/s（6.5万微型 CTA 调度瓶颈）→ flat 2048 元素/program 才 264GB/s
- [ ] 讲清 prefill vs decode：算术强度 256:1、实测 263×/token、decode 是 GEMV
- [ ] 默写 top-p 一行式 keep=(cum-probs)<p，说清边界 token 语义 + χ² 验证法
- [ ] 默写 AMP 循环五要素（autocast / scale / unscale_ / step / update），说清 clip 必须在 unscale 之后
- [ ] 一句话讲清 bf16 vs fp16（指数位=fp32→免 scaler），并报出本机反例：sm_75 无 bf16 TC，实测慢 0.57×
- [ ] 口算 MFU 公式 6N+12LHQT，报出 E3 四档 7.5/17.1/23.7/29.9% 与「跨精度不比 MFU」红线
- [ ] 讲清固定预算 scaling：3M 最优 1.576、25M 最差 2.063（1 tok/param），外加预注册门被证伪后的诚实修订流程
- [ ] 报出训练五数：fp16 加速 2.67×、bf16 0.57×、scaler 65536→131072 零回退、eager GEMM 占比 27%、top5 50.9%
- [ ] 说清 llm.c 四个融合点（CE 反向进 loss kernel / unscale 融进 adamw kernel / grad accum 融进 += / bias-GELU 进 cuBLASLt epilogue），及其「不重写 GEMM」定位
- [ ] 报出量化框架五数：NF4 0.092 vs INT4 0.100、τ=8 离群分解 4.2× 回 int8 本底、8bit-Adam 差在训练噪声内（三次 0.0007/0.0253/0.0207）/状态 0.254×、int8 TN 1.29× vs 默认 0.59×（配对 1.83×）、QLoRA 7.5B=4.0GB
- [ ] 说清「官方 NF4 ≠ 裸分位数」的证据链（absmax 归一化区间 vs ndtri 跨度，实测 0.092 vs 0.127）
- [ ] 讲清 8-bit Adam 三前提（decade 码本/block256/全链 dequant-requant），并讲出自己实测炸过的线性 int8 版本
- [ ] 说清 int8 布局决定 TC 路径的验证链（4 组布局对照 ~0.59× vs TN 1.29×，配对比值跨运行稳定），并关联 bnb 自定义权重布局的存在原因
- [ ] 每个主题至少准备一个「我踩过/见过这个坑」的真实细节（guard 越界、double 字面量、
      `.local` depot、syncthreads 非 uniform 死锁、只对拍 O 漏掉 L）

## 7. 使用建议

- **复习动线**：先过本索引 §6 自测 → 卡壳的主题回对应讲次的「面试深潜」→ 手写题
  在真机/模拟器上各写一遍（`GPU-Puzzles/solutions/verify.py --cpu` 可当阅卷器）。
- **口径一致性**：所有数字与本机实测或讲义原文一致；不确定的宁可说「量级」不编精确值。
- **红线优先**：面试里「说错一个红线」的伤害大于「少答对一个细节」——先背红线再背加分。
