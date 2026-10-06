# [AR004] 详细设计

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-06 |

## 1. 交付物结构

```
04-kernel-dsl/
├── triton-lab/
│   ├── bench_dsl.py          # F1/F2：--exp E1..E4/all 调度 + 计时 + 正确性门 + JSON
│   ├── kernels_triton.py     # triton kernel 定义（@jit 必须源文件，benches import）
│   ├── numba_ref.py          # venv 侧入口：numba add + K1/K2b GEMM（复刻 AR002 kernel），独立 Event 计时 → JSON
│   ├── plot_results.py       # F3：读 JSON → figs/*.png（300 DPI）
│   ├── results/*.json        # e1/e2/e3/e3cfg/e4/env + numba 子 JSON（e1na/e3na）
│   ├── figs/fig1..fig9.png
│   ├── results.md            # 逐图中文分析
│   └── README.md             # 复现步骤
├── notes/
│   ├── triton-lowering-sm75.md   # F4 拆解（PTX 证据链）
│   └── triton-notes.md           # F5 六段面试笔记
└── triton/, tvm/             # 上游克隆，零改动（mtime 核查）
```

## 2. 统一纪律

### 2.1 计时（沿用 AR002/003 修订版）

1. 会话级 2s FP32 大 GEMM 烧机（时钟爬坡）→ 配置级 3 次 warmup → 自适应 reps = clamp(3,10,round(1.0/t₁)) → 中位数 time_ms
2. CUDA Event 打点；全程 device 端数据（无 H2D）
3. **不用** `triton.testing.do_bench`（不同协议不可与本 AR 及 AR002/003 对照——本身就是红线素材；教程的 do_bench 仅在 README 提及差异）
4. numba 侧：venv python 独立进程，numba.cuda 事件计时 + 同样烧机/warmup/reps 中位数协议；主进程背靠背调度（同会话时钟态）；JSON 合并时标注 `{"runtime": "numba-venv", "same_session": true}`

### 2.2 正确性门（先于计时）

| 对象 | 参考 | 判据 |
|------|------|------|
| triton/torch add | 逐元素 | `torch.equal`（精确） |
| triton softmax | torch double softmax | max abs diff ≤ 1e-6；构造行含 ±1000 极值 + 全 -inf 行除外（只测 finite 输入）|
| naive 5-pass softmax | 同上 | 同上（证明展开版本自身正确，流量对比才有意义）|
| triton mm FP32 | torch.matmul（float32） | rel-err = ‖c−ref‖max/‖ref‖max ≤ 1e-5 |
| triton mm FP16 | float64 参考取 fp16 | ≤1e-2（FMA fp32 累加路径，预期远优于此门）|
| numba K1/K2b | AR002 已验证 kernel 复刻 | rel-err ≤ 1e-5 + 与 AR002 数值口径一致 |

### 2.3 JSON schema（记录级）

`{exp, variant, ...params, time_ms, tflops|gbps, reps, [n_regs, n_spills, mma_count, ldmatrix_count, oor_reason]}`
- **save() 语义**：整实验覆盖（同 exp 重复调用留最后一次）→ 循环内收集后单次落盘（AR003 E3seq 教训固化）
- 失败 config 记录 `{...params, "oor_reason": "OutOfResources: shared memory"}`，不算 PASS 记录但保留（fig 热图标注）

## 3. 实验设计

### E1 vector-add（三方带宽）
- N ∈ {2^12..2^26}（对数 15 档）；triton BLOCK_SIZE ∈ {256,1024,4096}（3 系列全扫）+ torch.add + numba add
- 流量模型：读 2N 写 N = 3×N×4B → GB/s；L2 4MB 拐点 = N≈2^20（12MB 流量）处标注
- 预期：小 N 高带宽（L2 命中），大 N → DRAM 峰值 80%+；三方接近（带宽上限问题，抽象层不改变天花板）→「add 是 DSL 的 hello world，差异在 N 很小时 launch/配置开销」
- numba add：朴素 1D grid 128 线程/块（与 AR002 K0 风格一致）

### E2 fused-softmax
- 形状：行宽 N ∈ {1024,2048,4096,8192,16384}，M = 2^20/N（总量 ≈ 4M 元素恒定；宽行 M 小）
- 三系列：triton 教程 kernel（BLOCK=next_pow2(N)，program 跨行 stride 循环）/ `F.softmax(x,dim=1)`（原生融合）/ naive 5-pass（max/sub/exp/sum/div 逐 op）
- 融合口径 GB/s = 2×M×N×4B/t；naive 另记「实际流量口径」= (8MN+4M)×4B/t 对照理论 4× 收益
- num_warps ∈ {4,8} 对宽行（16K）的影响单独记录
- 正确性：构造含 ±1000 的行（数值稳定性验证）

### E3 matmul FP32 全 config 扫描（本 AR 重头）
- N ∈ {256,512,1024,2048} 方阵 FP32
- 手动 config 网格（不用 @autotune 以便逐 config 记录）：BM,BN ∈ {32,64,128}；BK ∈ {16,32,64}；num_warps ∈ {4,8}；num_stages ∈ {2,3}；剪枝规则：BM×BN ≤ 128×128、threads = num_warps×32 ≤ 256、shared ≈ stages×(BM+BN)×BK×4B ≤ 64KB 硬限（编译失败也如实记录）
  - 估计可编译 config ~40-60 个/尺寸 → 全量计时落 e3cfg.json（N, BM, BN, BK, warps, stages, tflops 或 oor_reason）
- 三方对照表（e3.json）：triton 最优 config / cuBLAS / numba K1_T16 / numba K2b_64x64
- GROUP_M swizzle 对照：固定每尺寸最优 config，GROUP_M ∈ {1,8} 各测（L2 复用效应，Triton 教程图解机制的实测版）
- 对账分析：最优 config 的 shared 用量/wave 数（⌈N/BM⌉×⌈N/BN⌉/48）/理论 FFMA 上限（FP32 无 TC，上限 = 11.2 TF）
- 预期（去风险后的假设，需实测验证）：triton FP32 FMA 路径可达 cuBLAS 60-80%（tile+pipeline 由编译器生成，接近手写极限）；numba K2b 4.65× 差距基线（AR002）是「手写无 pipeline」的下界

### E4 matmul FP16 + lowering 证据
- N ∈ {512,1024,2048} 方阵；输入 fp16，累加 fp32（acc dtype=tl.float32）
- triton：E3 中每尺寸最优 FP32 config 复用为起点 + 小范围再扫（BM,BN ∈ {64,128}×BK ∈ {16,32}×warps {4,8}），每 config 记录 mma_count/ldmatrix_count（PTX 字符串计数）/n_regs/n_spills
- cuBLAS fp16 对照（torch.matmul half → TC 54.9-67.4 TF 基线）
- 预期：triton fp16 ≈ FP32 水平（FMA 降级，fp16→fp32 转换反而多一步）vs cuBLAS ~10-30× 差距 →「fp16 输入≠TC 被使用」红线 + 上游 MMAv2 存在性/未触发差距（F4 文档解释）
- PTX 存档：每尺寸最优 config 的 PTX 中 mma/fma 计数 + 代表性片段存 `results/e4_ptx_summary.json`

### 环境探测（env.json）
triton 版本/安装路径、torch/cuda、venv numba 版本、GEMM 尺寸上限显存检查（2048³ fp32 = 48MB×3 充裕）

## 4. 图表规格（figs/*.png，300 DPI，英文图内文字）

| fig | 内容 | 关键标注 |
|-----|------|---------|
| fig1 | E1 三方 add GB/s vs N（对数 x） | 448 GB/s 峰值线、L2 4MB 拐点竖线、三方系列 |
| fig2 | E1 triton BLOCK_SIZE {256,1024,4096} 对比 | 小 N launch 开销差异区标注 |
| fig3 | E2 三方 softmax GB/s vs 行宽 | 融合口径；naive 的双口径（有效/实际流量） |
| fig4 | E2 融合收益：naive vs triton 实测 GB/s 比值 + 理论 4× 线 | 行宽效应 |
| fig5 | E3 三方 matmul TFLOPS vs N | cuBLAS 峰值达成率标注；numba AR002 基线横线参考 |
| fig6 | E3 config 热图（N=1024：BM×BN 网格，cell 内 BK/warps/stages 最优值，OO 格打叉） | 最优 config 高亮 |
| fig7 | E3 最优 config 随 N 变化 + shared/wave 对账表 | 每 N 一行：BM×BN×BK/warps/stages/waves/shared |
| fig8 | E4 fp16：triton vs cuBLAS + 89.2 TC 峰值线 + 11.2 FP32 线 | 「无 mma」注释 |
| fig9 | E4 regs/spills vs config 散点（BM×BN 颜色映射） | mma=0 说明框 |

## 5. F4 文档结构（triton-lowering-sm75.md）

1. 一句话结论（sm_75 上 tl.dot 三 dtype 实测 lowering 表）
2. PTX 证据链（fp32/fp16 计数表 + 代表片段；[本机] 标注）
3. 上游源码定位（MMAv2.cpp mmaInstrPtxTuring/callMmaTuringFp16 行号；[源码] 标注）+ 「路径存在但构建未选择」三层表述（事实/差距/开放问题）
4. block 编程模型要点（program≈CTA、block-uniform 标量、mask/OOB、constexpr）
5. autotune 机制（config 枚举/cache 键=constexpr+launch 元参/首跑编译开销——E3 手动扫描 vs @autotune 等价性说明）
6. 工程约束清单（@jit 需源文件/inspect、do_bench 协议差异、Windows wheel 特有性）
7. 与 CUTLASS/numba 心智对照表（tile↔program、shared 管理权、pipeline 生成权）

## 6. 风险与预案

| 风险 | 概率 | 预案 |
|------|------|------|
| E3 config 网格编译慢（每 config 首编 ~1-3s × 100+） | 高 | 烧机后先扫 N=1024 全网格，其余 N 只测 top-5 config + 失败记录；总编译量封顶 ~200 次 |
| triton fp16 某 config 意外触发 mma | 低 | 如实记录并更新结论（证据优先原则，srs §5 已声明） |
| numba venv 子进程环境漂移（驱动/CRT） | 低 | 探测先行 + K2b 复跑对照 AR002 基线带（2.036±10%），超带即调查 |
| WDDM 短窗口噪声 | 高 | 沿用口径：长窗抽查 <5% 记录；微小效应只报方向 |
| 2048³ 全 config 显存/时间超预算 | 中 | 2048 只测 top config；E3 总 wall time 封顶 ~20 分钟 |

## 7. 验收对照（srs §3 → 交付物）

- §3.1 框架/门/波动 → bench_dsl.py + results/env.json + 复跑抽查记录
- §3.2 E1-E4 → results/e1.json/e2.json/e3.json/e3cfg.json/e4.json(+e1na/e3na)
- §3.3 图表 → figs/fig1..fig9.png（数量 ≥8 即可，合并可）
- §3.4 F4 → notes/triton-lowering-sm75.md
- §3.5 F5 → notes/triton-notes.md + INTERVIEW-INDEX + results.md + README
- §4 NFR → mtime 核查记录 + 全局环境无 pip + 真机 JSON + wall time 记录
