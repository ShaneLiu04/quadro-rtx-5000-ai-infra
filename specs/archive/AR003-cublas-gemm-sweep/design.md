# [AR003] 技术设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-05 |
| 状态 | Draft |

## 1. 总体架构

```
03-gemm/gemm-sweep/                       ← 新建，上游零改动
├── bench_sweep.py        # F1: --exp {E1..E7,all} 子实验调度 + 统一计时 → results/*.json
├── plot_results.py       # F3: 读 JSON → figs/*.png
├── results/
│   ├── e1.json ... e7.json
│   └── env.json          # 机器/环境快照
├── figs/fig1a..fig7.png  # 8 张（E1 拆 a/b 两图）
├── results.md            # F5: 逐图中文分析
└── README.md             # F5: 复现步骤

03-gemm/notes/
├── cutlass-turing-dissection.md   # F4: CUTLASS 08_turing 源码拆解（独立深潜文档）
└── gemm-sweep-notes.md            # F5: 形状空间面试笔记（六段结构，引用 dissection 与实测数字）
```

**执行入口**：全程全局 python（torch 2.5.1+cu121 + matplotlib）——本 AR 无 venv 依赖：
```powershell
python bench_sweep.py --exp all      # gemm-sweep/ 下，~30-40 min
python plot_results.py
```

**模型常量**（plot/分析统一引用，注释出处）：48 SM、FP32 11.2 / FP16 89.2 / INT8 178.4 TFLOPS(TOPS)、HBM 448 GB/s、L2 4MB、clock 1815 MHz、roofline ridge 25 FLOP/B、wave 模型 tile 128×128（**cuBLAS 内部 tile 由启发式决定，128×128 仅为可视化模型假设**，图注明确声明）。

## 2. F1 — bench_sweep.py 框架设计

### 2.1 统一计时纪律（沿用 AR002 修订版）

1. **会话级热身**：启动后先跑 ~2s FP32 大 GEMM 烧机（时钟爬坡），再进入各配置
2. **配置级**：3 次 warmup → 单次测 t₁ → reps = clamp(3, 10, round(1.0/t₁)) → 取中位数 time_ms
3. CUDA Event 打点；pinned H2D（本 AR 输入直接 device 端生成，无 H2D 环节，pinned 仅在正确性对拍时使用）
4. 每记录落 JSON：`{exp, label, M, N, K, dtype, variant, time_ms, tflops, gbps, reps}`
   - **实现偏离注记（review 修订）**：实际记录按实验携带各自字段（E1a/E1b 增 `model_tiles/waves`，E3 增 `split_s`，E4 增 `batch_B`，E7 增 `copy_bytes/pct_of_runtime`），非全局统一 schema；`label/dtype/gbps` 仅部分实验填充。`save()` 语义为**整实验覆盖**（同 exp 重复调用只留最后一次），故循环内实验须收集后单次落盘（E3seq 修正后口径）。

### 2.2 dtype 路径探测（T001 第一步）

- `hasattr(torch, "_int_mm")` → 真机试跑 128³ int8：异常捕获（CC 不足会 raise）→ 记录 `env.json: {"int_mm": true/false, "error": ...}`
- 可用则**精确对拍**：int8∈[-4,4] 随机 → `_int_mm` vs int32 参考累加，`torch.equal`（INT8×INT8→int32 精确，容差 0）
- 不可用 → E5 降级（srs §5 口径），plot/notes 走理论换算分支

### 2.3 正确性门（先于一切计时）

| 对拍 | 方法 | 判据 |
|------|------|------|
| `_int_mm` | vs int32 循环参考 | `torch.equal` |
| FP16 matmul | vs FP32 参考转 half | `allclose(rtol=1e-3, atol=1e-3)` |
| E7 转置视图 | `(A.t()@B.t()).t()` vs `B@A` | `allclose` |
| E3 split 流水 | 各路部分和相加 vs 全 K 单发 | `allclose(atol=1e-3)` |

## 3. F2 — 七组实验矩阵

### E1 形状网格（a 热图 + b 细粒度阶梯）

- **E1a**：M,N ∈ {64,128,256,512,1024,2048,4096}（49 格），K=2048，FP32 → TFLOPS 热图；叠加 wave 等值线（⌈M/128⌉·⌈N/128⌉/48 向上取整）
- **E1b**：N=K=4096，M ∈ {64,96,128,160,192,224,256,320,384,512,768,1024,1536,2048}（在 wave 边界 128·k 附近加密采样）→ TFLOPS vs M 阶梯线；wave 边界垂直虚线标注
- 分析锚点：同 wave 内平坦、跨 wave 台阶；M=64 行（半 tile）与 M=128 同 wave 的对比

### E2 skinny 边界

- M ∈ {1,2,4,8,16,32,64,128,256,512,1024}，N=K=4096，FP32
- 双轴：TFLOPS + 有效带宽 GB/s（bytes = 4·(MK+KN+MN) / time）
- 参考线：448 GB/s 带宽极限的水平线；roofline 拐点 M*（AI=25 → M ≈ 25/4·… 推导写 notes）
- M=1 即 GEMV：单核带宽可达率分析（对照 AR002 pinned 教训：WDDM 下小 kernel 计时的 launch 抖动，reps≥5）

### E3 split-K 代价/收益（真流并行动画版）

- 形状：M=N=256, K=16384（无 split 时仅 4 个 128×128 tile → 48 SM 中 44 个空转——**并行度饥饿**构造）
- s ∈ {1,2,4,8,16}：把 K 切 s 段，**s 条独立 CUDA stream 并发**跑 s 个 (256×256)×(256×16384/s) 部分积（各自 C 缓冲），再相加
- 计时：全程 Event 包络（含 sum）；s=1 即单发基线
- 对照面：cuBLAS 内部启发式可能已对 256×256×16384 自动 split-K——s=1 基线异常快时在 results.md 记录并分析（面试点：库启发式 vs 手工调度）
- 风险：WDDM stream 串行化 → 若 s 路并发无加速，改测「顺序 s 路 + sum」并把对照改成「cuBLAS 自动 vs 手工顺序」，如实记录
- **执行修订（review 闭环）**：E3 实测 s 路全败且 s=1 基线异常快（库内部已 split-K）；按风险预案补做 **E3seq 顺序对照实验**（同切分、默认流逐段执行，落 `e3seq.json`），实测 stream 在每个 s 都比顺序**更慢**（-9.8%~-43.9%）——「WDDM 串行化」从推测升级为有对照的实测结论，fig3 改为 stream vs sequential 双柱对照

### E4 batch GEMM

- bmm：B ∈ {1,2,4,8,16,32,64} 个 (256×256)@(256×256)，strided batch
- 对照：等价 flop 单大 GEMM ((B·256)×256×256)
- 双系列 TFLOPS vs B + 折线标注「单 kernel 调度 vs 多 tile 并行」权衡

### E5 数据类型断层

- 方阵 N ∈ {512,1024,2048,4096}；FP32 / FP16（半精度存储，FP32 累加） / INT8（`_int_mm`，int8 存储）
- 三峰值线 11.2 / 89.2 / 178.4；达成率百分比标注
- 与 AR002 数字衔接：FP32@2048、FP16@2048 应落在 AR002 实测的会话噪声带内（复现即交叉验证）

### E6 L2 驻留效应

- N ∈ {512,1024,2048,4096}，FP32：**flush**（每次计时前写 64MB dummy buffer）vs **no-flush**（连续迭代）
- 模型：N=512 工作集 3×1MB=3MB < 4MB L2 → no-flush 应显著快（L2 带宽 >> HBM）；N≥1024 工作集 > L2 → 差异收敛
- AR002 教训适用：微小效应量级，**3 次独立观测方向一致**才下结论

### E7 layout 四组合

- 2048³ FP32：`(A,B) / (A.t(),B) / (A,B.t()) / (A.t(),B.t())` 四组合 + `A.t().contiguous()` 对照
- torch 2D matmul 对转置视图直接传 cuBLAS transa/transb（不拷贝）；用 `memory_allocated` 前后差断言无隐式拷贝（防 cuBLAS fallback 路径污染数据）
- 分析：cuBLAS 各 layout 路径的 shared 装载效率（TN 历史上最快——coalesced + 无转置 ldmatrix 需求），对照 CUTLASS `LayoutInputA=RowMajor/ColMajor` 模板维度的自由度

## 4. F3 — 图规范

| 图 | 内容 | 可解释要素 |
|----|------|-----------|
| fig1a | 7×7 TFLOPS 热图 | wave 等值线叠加；色标 TFLOPS；对角线参考 |
| fig1b | TFLOPS vs M 阶梯 | wave 边界虚线；台阶标注 |
| fig2 | skinny 双轴 | 448 GB/s 线；roofline 拐点竖线；GEMV 区/ GEMM 区着色 |
| fig3 | split-K 柱状 | s=1 基线横线；「饥饿→并行→归并开销」三段标注 |
| fig4 | batch 曲线 | bmm vs 等价大 GEMM 双系列；每 batch 的 tile 数标注 |
| fig5 | dtype 断层 | 三峰值线；达成率 % 标注 |
| fig6 | L2 对比柱 | 按 N 分组 flush/no-flush；工作集/L2 容量比标注 |
| fig7 | layout 柱状 | 四组合 + contiguous 对照；cuBLAS op 标注 |

图面：英文标签、300 dpi、配色沿用 AR002（FP32=绿、FP16=橙、INT8=红、模型线=灰虚线）。

## 5. F4 — cutlass-turing-dissection.md 结构

1. **示例定位**：08_turing = 本机 sm_75 INT8 TC 的官方参考实现；`device::Gemm` 15 模板参数逐个释义表
2. **数据流管线**：global→reg→shared→reg→mma→reg→global（示例注释原文 7 步 + 2-stage 相位错位图，文字+ASCII 图）
3. **tile 层级对账表**（每个数字列计算式）：
   - TB 128×256×64：warp 数 = 128·256/(64·64) = 8 → 256 线程
   - shared = 2 stages × (128·64 + 64·256) × 1B = 48KB ≤ 64KB ✓——**恰好解释 NumStages=2：INT8 下 3 stages 需 73.7KB 超限**（本 AR 独有可解释点）
   - mma 8×8×16 → 178.4 TOPS 换算链：1.815 GHz × 48 SM × 512 MAC/SM/clk × 2（TOPS 计 MAC 双操作）→ 每子核 128 MAC/clk → m8n8k16 × 8 拍
   - epilogue LinearCombination：vector width = 128/32 = 4 元素 = 16B/访问
4. **机制三章**：swizzle（identity vs rasterization 目的）、split-K（workspace + reduction）、`can_implement → initialize → operator()` 生命周期
5. **对照**：`media/docs/cpp/efficient_gemm.md` 的 Hierarchical Structure / Pipelining / Rasterization / Parallelized Reductions 逐节挂钩 08 源码
6. **红线**：与 sm_80+ 的差异（无 cp.async → stage 数受 shared 硬限；无 ldmatrix? — Turing **有** ldmatrix，CUDA 10.2 引入，写明事实链）

## 6. F5 — gemm-sweep-notes.md 六段结构

1. **高频问法**：cuBLAS 在什么形状下「翻车」？skinny GEMM 为什么慢？split-K 何时是银弹？batch GEMM vs 大 GEMM 怎么选？INT8/FP16/FP32 断层从哪来？CUTLASS 的 tile 层级怎么读？
2. **追问链**：方阵峰值 → 形状空间不均匀（wave 量化）→ skinny 并行度饥饿 → split-K 补救与代价 → 库启发式边界 → CUTLASS 模板参数如何表达这些自由度
3. **数字卡片**：E1-E7 实测（wave 台阶倍率、GEMV 带宽达成率、split-K crossover s*、batch 盈亏点、三 dtype 达成率、L2 驻留加速、layout 最快/最慢比）+ AR002 衔接
4. **手写骨架**：`device::Gemm` 类型别名组装（08 示例核心 20 行）+ wave 计算 pseudocode + split-K stream 调度骨架
5. **红线清单**：方阵 TFLOPS 外推 skinny；split-K 当免费午餐；忽略 wave 量化讲「利用率」；batch 维度当免费并行（显存×B）；flush 条件不一致的 L2 对比；INT8 无缩放直接当 FP32 用（动态范围）
6. **60 秒电梯陈述**：GEMM 性能 = 形状空间里的调度问题；wave/skinny/split-K 三把刀 + 本机实测数字

## 7. 风险与对策

| 风险 | 概率 | 对策 |
|------|------|------|
| `torch._int_mm` 在 sm_75 不可用（CC 检查） | 中 | §2.2 探测先行；降级为 FP32/FP16 + INT8 理论换算，results.md 记录 |
| E3 stream 并发在 WDDM 串行化 | 中 | 记录实测；切换「顺序 s 路」对照；结论两种路径都写 |
| E6 效应量 < 噪声带 | 中 | 3 次独立观测方向一致才下结论（AR002 会话噪声教训） |
| cuBLAS 内部启发式干扰 E3 基线 | 高 | s=1 基线异常快即证据，写进分析（库启发式 vs 手工） |
| E7 转置视图被隐式拷贝 | 低 | `memory_allocated` 断言；有拷贝改用 cuBLASLt? 不可用则记录 |
| E1 全网格耗时超预算 | 低 | 自适应 reps（1s/配置）+ 49 格 ≈ 1.5min 实测即知 |

## 8. 测试策略

- **正确性门（硬）**：§2.3 表全绿才允许计时；失败修实现不动阈值
- **计时稳定性**：T006 后抽查 E1b 单配置、E5@2048 重跑波动 <5%（会话噪声带口径）
- **数字一致性**：T008 笔记/分析引用数字 vs JSON 逐个核对（AR002 纪律）
- **上游零改动**：T011 mtime 聚类核查 cutlass/ 目录
- **验收**：T011 逐条对 srs F1~F5
