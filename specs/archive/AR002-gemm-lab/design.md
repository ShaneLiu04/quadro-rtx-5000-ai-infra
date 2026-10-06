# [AR002] 技术设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR002 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-05 |
| 状态 | Draft |

## 1. 总体架构

```
02-handwritten-kernels/gemm-lab/          ← 新建，上游零改动
├── kernels_numba.py        # F1: K0/K1/K2 三级 SGEMM kernel（venv 侧）
├── bench_numba.py          # F2: numba 侧基准 → results/bench_numba.json（venv 侧）
├── bench_torch.py          # F2: cuBLAS FP32/FP16 基线 → results/bench_torch.json（全局侧）
├── plot_results.py         # F3: 合并 JSON → figs/fig1..fig6.png（全局侧）
├── results/
│   ├── bench_numba.json
│   ├── bench_torch.json
│   └── env.json            # 机器/环境快照（GPU 型号、驱动版本、库版本）
├── figs/fig1..fig6.png
├── results.md              # F5: 逐图中文分析
└── README.md               # F5: 复现步骤 + 环境矩阵

02-handwritten-kernels/notes/
└── gemm-lab-notes.md       # F4: 面试导向笔记（六段深潜 + 图嵌入）
```

**数据流**：venv python 跑 `bench_numba.py` → JSON；全局 python 跑 `bench_torch.py` → JSON；全局 python 跑 `plot_results.py` 读两份 JSON → 图。JSON 是两环境唯一交换格式。

**执行入口**（README 记录）：
```powershell
& "D:\Infra\01-foundations\GPU-Puzzles\solutions\.venv\Scripts\python.exe" bench_numba.py      # gemm-lab/ 下
python bench_torch.py                                                                        # 全局
python plot_results.py
```

## 2. F1 — Kernel 设计

### 2.1 K0 naive（基线，暴露全局内存带宽瓶颈）

- 网格/块：grid=(⌈N/B⌉, ⌈N/B⌉)，block=(B, B)，B=16（256 线程）
- 每线程 1 个输出 C[i,j]：k 循环直接读 A[i,k], B[k,j]（全局内存，无复用）
- guard：i<N、j<N 才写回；读 A/B 时 k<N 天然安全（N×N 数组）
- 作用：测得「无复用」的 TFLOPS 地板，E4 机制因果的对照原点

### 2.2 K1 tiled（shared memory 分块，消除 AB 重复全局读）

- BLOCK=T×T（T∈{8,16,32}），grid=(⌈N/T⌉,⌈N/T⌉)，block=(T,T)
- k 以 T 为步长循环：协作装载 A[i:i+T, k:k+T] 与 B[k:k+T, j:j+T] 进 shared → `syncthreads()` → 内层 t∈[0,T) 累加 → `syncthreads()`
- guard 装载：越界元素写 0.0（shared 全员显式初始化，非整除 N 正确）
- shared 用量：2·T²·4B（T=32 → 8KB）
- 机制预期：每输出全局读从 2N 降到 2N/T

### 2.3 K2 regblock（寄存器分块，提升每线程算术强度与指令级并行）

两种配置（E3 扫描对象）：
- K2a：BLOCK=32×32，block=(16,16)（256 线程），每线程 2×2 输出，k 步长 32
- K2b：BLOCK=64×64，block=(16,16)（256 线程），每线程 4×4 输出，k 步长 64

实现要点：
- 线程到输出的映射：thread(tx,ty) 负责 C[base_i+ty*TM : +TM, base_j+tx*TN : +TN]（行用 ty 保证同行共享 A tile 读的对齐）
- shared 两块 As[TM_TILE][K_STEP]、Bs[K_STEP][TN_TILE]
- 内层四重循环：k∈K_STEP × m∈TM × n∈TN，累加器为寄存器数组 `acc[TM][TN]`（numba 展开为标量）
- 卸载阶段：协作搬运 As 行（每线程一行）+ Bs 列；guard 补零
- 机制预期：每线程每 k 步只做 1 次 As 读 + 1 次 Bs 读，供 TM·TN 次 FMA → shared→register 流量降 TM·TN/2 倍；全局读每输出进一步降

### 2.4 正确性门（T002，先于一切计时）

对拍矩阵：{K0, K1(T=8/16/32), K2a, K2b} × N∈{96, 128, 200, 500}，与 numpy `A@B` 比 `allclose(atol=1e-3)`。
N 取值刻意含非整除（96 对 T=8 整除但对 32×32 不整除；200/500 对所有 T 非整除）→ 验证补零路径。

## 3. F2 — 基准设计

### 3.1 统一计时纪律（两脚本共用语义）

1. 输入：`np.random.randn(N,N).astype(np.float32)`（torch 侧 `torch.randn`），C 预分配
2. 3 次 warmup → 单次测得 t₁ → reps = clamp(3, 10, round(2.0/t₁)) → 取中位数 time_ms
3. CUDA Event 打点（numba：`cuda.event()`；torch：`torch.cuda.Event`）
4. TFLOPS = 2·N³ / (time_ms·1e-3) / 1e12

### 3.2 N 网格（按 kernel 分档，控制总时长 ≤15min）

| 系列 | N 网格 |
|------|--------|
| K0 | 128, 256, 512, 1024 |
| K1 (每个 T) | 256, 512, 1024, 2048 |
| K2a / K2b | 256, 512, 1024, 2048, 4096 |
| cuBLAS FP32 | 256, 512, 1024, 2048, 4096 |
| cuBLAS FP16 | 256, 512, 1024, 2048, 4096 |

（E1 全 kernel 同框比较取公共 N∈{256,512,1024}；曲线各自延伸到自己的最大 N，图例注明。）

### 3.3 JSON Schema

```json
{
  "env": {"python": "...", "lib": "numba 0.68.0 | torch 2.5.1+cu121", "gpu": "Quadro RTX 5000", "cc": "7.5"},
  "series": [
    {"kernel": "K1_tiled_T16", "dtype": "fp32", "N": 1024, "time_ms": 12.3, "tflops": 0.174, "reps": 10}
  ]
}
```

## 4. F3 — 实验与图规范（E1~E6）

理论常数（README 口径，写死并注释出处）：FP32 峰值 11.2 TFLOPS、FP16 TC 峰值 89.2、HBM 448 GB/s、L2 4MB、shared 64KB/SM、1024 thr/SM、roofline ridge = 11.2e12/448e9 = 25 FLOP/B。

| 图 | 内容 | 数据系列 | 可解释要素 |
|----|------|---------|-----------|
| fig1 | TFLOPS vs N | K0, K1(T16), K2a, K2b, cuBLAS-FP32 | 11.2 峰值横线；各 kernel 图例标机制名 |
| fig2 | tile 扫描 @N=2048 | K1 T=8/16/32 | 柱标注 shared 字节数与 SM 理论驻留 block 数（shared 64KB 与 1024 thr 双约束取小） |
| fig3 | regblock 扫描 @N=2048 | K1 T32（对照）, K2a, K2b | 柱标注每线程 FMA 数/k 步与寄存器累加器估算（TM·TN） |
| fig4 | 机制因果 | 全部 kernel @N=1024 | x=理论每输出全局读次数（对数），y=实测 TFLOPS；标注「读次数↓ → 带宽压力↓ → 算力利用率↑」因果链 |
| fig5 | FP32 vs FP16 | cuBLAS FP32, FP16 @N=2048/4096 | 双峰值线 11.2/89.2；实测达成率百分比标注 |
| fig6 | roofline 定位 | 各 kernel @N=1024 的 (AI, TFLOPS) 点（修订：design 原写 @2048；开发中改用 1024 以覆盖 K0 的最大实测尺寸，使全系列含 K0 同框——K0 超出带宽屋顶的落点是「L2 缓存」教学点，去掉 K0 会损失该图的核心可解释要素。偏离记录于 tasks.md Review 节） | log-log；25 FLOP/B ridge 垂线；带宽斜线和算力平台线；各点旁标 kernel 名与 AI 推导 |

**AI（算术强度）推导口径**（笔记与图一致）：
- K0：每输出 2N 次全局读 ×4B → 总字节 8N³，AI = 2N³/8N³ = 0.25 FLOP/B（与 N 无关，极低）
- K1(T)：每输出 2N/T 次读 → 字节 N²·(2N/T)·4B → AI = T/4（T=16 → 4）
- K2a：AI = (TM·TN/(TM+TN)) · (K_STEP/4)… 统一推导写进笔记 §3，代码里 `plot_results.py` 用同公式常数生成

**图面语言**：英文标签（DejaVu 字体安全），300 dpi，统一配色（naive=灰、tiled=蓝、regblock=橙/红、cuBLAS=绿）。

## 5. F4/F5 — 文档设计

### 5.1 gemm-lab-notes.md（面试六段结构）

1. **高频问法**：手写 GEMM 怎么一步步优化？shared tiling 为什么快？寄存器分块解决什么？cuBLAS 为什么还快 5-10 倍？tensor core 是什么？
2. **追问链**：naive 瓶颈（带宽）→ tiling 机制（复用）→ 为何 T 不能无限大（shared 容量/占用）→ regblock（寄存器压力 vs ILP）→ 到达的墙（无 cp.async 双缓冲、无 tensor core）→ cuBLAS 差距构成
3. **数字卡片**：本机实测——K0 ~0.2-0.4、K1_T16 ~2、K2b ~4-6（预估，以实测为准）、cuBLAS FP32 ~8-10、FP16 ~30-50 TFLOPS；AI 推导表；roofline ridge 25
4. **手写骨架**：K1/K2 关键代码段（shared 声明、双 syncthreads、累加器展开）+ 每步注释
5. **红线清单**：syncthreads 非均匀路径死锁、shared 越界、guard 补零遗漏、bank conflict、非整除 N、计时忘 warmup、只报均值不报中位数
6. **60 秒电梯陈述**：优化三连（复用→占用→指令级并行）+ 本机实测数字

### 5.2 修正与增补

- L014 §8：triton 陈述改为实测结论（venv 无 triton → torch.compile CUDA 不可用）
- INTERVIEW-INDEX：GEMM 条目 → 指向 gemm-lab-notes.md

## 6. 风险与对策

| 风险 | 概率 | 对策 |
|------|------|------|
| K2 在 numba 下因寄存器溢出反而变慢 | 中 | E3 本身就是扫描实验，慢也如实呈现并解释（面试点：寄存器压力） |
| 大 N naive 太慢 | 高 | N 网格分档（K0 止步 1024）；自适应 reps |
| numba kernel 编译时间计入首跑 | 低 | warmup 3 次天然吸收 JIT |
| FP16 cuBLAS 小 N 达成率低 | 中 | E5 注明「小 N 受 launch/带宽主导」，只在大 N 比峰值达成率 |
| matplotlib 中文字体缺失 | 确定 | 图内英文、分析中文（已定） |

## 7. 测试策略

- **正确性门（硬）**：T002 全绿才允许计时；失败即修 kernel，不动阈值
- **计时稳定性**：T005 抽查同配置重跑波动 <5%
- **验收**：T011 逐条对 srs.md F1~F5 + 上游零改动（目录 mtime/文件数核查）
