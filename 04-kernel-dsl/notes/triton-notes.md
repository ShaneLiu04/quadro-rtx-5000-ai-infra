# Triton DSL 实验笔记（triton-lab / AR004）

> 三方框架（triton 3.8 wheel / torch-cuBLAS / numba venv 子进程）在 Quadro RTX 5000（sm_75）
> 上的对照实验：vector-add、融合 softmax、FP32 matmul config 全扫描（76 valid/32 pruned）、
> FP16 lowering 取证（26 config 全 mma=0）。9 张可解释图 + PTX 证据链。
> 代码与图：`../triton-lab/`，逐图分析：`../triton-lab/results.md`，
> lowering 拆解：`./triton-lowering-sm75.md`。
> 协议：编译/计时分离 + 每尺寸 fresh burn + nvidia-smi 时钟遥测（1875-1920 MHz 计时窗口）。

## 1. 三方对照结论（先把数字钉死）

| 实验 | triton | cuBLAS/torch | numba | 一句话 |
| --- | --- | --- | --- | --- |
| E1 大 N 带宽 | 372.6 GB/s（83%） | 377.6（84%） | 368.9（82%） | 带宽型 kernel 框架无关 |
| E1 launch 下限 | 45.6 µs | 25.6 µs | 69.4 µs | 三层台阶 = launch 路径成本 |
| E2 softmax | 142-154 GB/s | 184-221（原生） | — | 教程 kernel ≠ 最优，差 1.2-1.6× |
| E3 FP32 @2048 | **8.67 TF**（87% of cuBLAS） | 9.95 | K2b 2.01（20%） | DSL 代码生成值 4× |
| E4 FP16 @2048 | **6.64 TF（mma=0）** | 63.07（TC） | — | lowering 未触发 TC，差 9.5× |

三个断层（面试可背）：**FP32 上 triton≈cuBLAS 的 87%**（同一 FMA 路径的代码生成差距
只有 13 个点）→ **FP16 上 9.5×**（lowering 决定走不走 TC）→ **fp16 输入反而比 fp32 慢
1-23%**（无 TC 收益还要付转换开销）。

## 2. 核心机制拆解

1. **launch 台阶 25.6/45.6/69.4 µs**：torch 走 C++ dispatch 直达 driver；
   triton 每次 launch 过 Python launcher（薄但仍是解释层）；
   numba 的 cuda.jit 调用路径最厚。WDDM 用户态提交队列放大了三者差异
   （Linux 上台阶会更矮）。triton 逐尺寸时间几乎不变（45.6-46.8 µs），
   torch 反而抖（25.6-39.9）——抖动本身是 WDDM 批提交的证据。
2. **教程 softmax 慢在哪**：PTX 取证排除「缺向量化」（`ld.global.v4`×4、
   `ex2.approx`×16、37 regs、0 spills；mask 不阻止 v4——对照实验证明）。
   剩余机制 = 每行 block 规约的 `bar.sync` 停顿 × 2 CTA/SM 低占用率。
3. **naive 5-pass 实测只慢 2.45×，低于 4× 流量模型**：模型上界 8MN+4M（x 读 2 次 +
   中间张量多 pass 往返）= 4× 融合流量，但本设置中间张量恰 4MB ≈ L2 被
   部分吸收 + 多 kernel 间隙——**融合省流量，但小工作集下 L2 会替 naive 买单**；
   「流量差 × L2 折扣」才是实测口径。
4. **config 扫描 @1024**：spread 2.67×（6.90 vs 2.59 TF）；最优 64×128×BK16 w4 s2；
   top-6 中 5 名 w4（每线程活多 = ILP 链长）；32 config 因
   shared = stages×BK×(BM+BN)×4B > 64KB 被剪。
5. **最优 config 随 N 迁移**（64×128×32_w8 → 64×128×16_w4 → 128×64×16_w4_s3）：
   autotune 是 per-size 的；cuBLAS heuristic 每尺寸都压住扫描最优 = 预扫好的表。
6. **fp16 无 mma 的完整证据链**：26 config × 3 尺寸 PTX 计数全 `mma.sync=0`、
   `ldmatrix=0`，静态 `fma.rn.f32`=BM×BN×BK/threads（如 512 = 64×64×16/128），
   `cvt.f32.f16`×16——fp16 逐元素转 fp32 后进标量 FMA。
   上游 3.9 源码存在 Turing mma 路径（MMAv2.cpp `mmaInstrPtxTuring` m16n8k8），
   但 3.8 Windows wheel 构建未启用。**「源码有」≠「你的二进制有」**。
7. **FMA 路径改变最优 config 形状**：fp16 最优全 64×64×16_w4（196 regs），
   fp32 最优 64×128/128×64——没有 TC 分摊累加器寄存器，大 tile 直接被压力压垮。
   autotune 必须在真实 lowering 下做。

## 3. 与 AR002/AR003 的数字对账（跨 AR 可信度）

- cuBLAS FP32@2048：AR002 9.468 → 本 AR 9.950（+5.1%，会话带内）
- numba K2b@2048：AR002 2.036 → 本 AR 2.008（-1.4%）；K1 0.885 → 0.944（+6.6%）
- cuBLAS FP16@2048：AR003 带 60.7-67.4 → 本 AR 63.07（带内）
- 结论：同一 kernel 复跑落在前 AR 带内 = 协议跨 AR 可比（E1 大 N 重复性 0.1-0.2%）

## 4. 手写骨架（triton 教程 matmul，面试默写版）

```python
@triton.jit
def matmul(A, B, C, M, N, K,
           stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    # L2 分组：让 GROUP_M×BN 个相邻 CTA 同时读同一片 A —— swizzle 的块级版
    num_pid_m = tl.cdiv(M, BM); num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # 累加器显式 fp32；BLOCK 常量编译期特化
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(block_ptr_a, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(block_ptr_b, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc)             # lowering 决定 FMA 还是 mma
    tl.store(block_ptr_c, acc.to(tl.float16), boundary_check=(0, 1))
```

默写得分点：**program_id ↔ blockIdx** / GROUP_M swizzle 意义（L2 复用，实测 ±5% 二阶）/
acc 显式且 fp32 / mask 或 block_ptr+padding_option 处理边界 /
`num_warps`/`num_stages` 是编译参数（每个组合 = 不同 cubin）。

## 5. 面试深潜（Interview Deep-Dive）

### 5.1 高频问法与口述骨架

- **Q1「Triton 和 CUDA 什么关系/区别」**：块级编程——我写「一个 program 算一个
  64×128 的 tile」，编译器决定 256 个线程每人干什么。等价关系能背：
  program_id↔blockIdx、BLOCK/(warps×32)=每线程元素数、num_stages↔手动双缓冲。
  实测证据：BLOCK=4096/nw8 的 softmax PTX 里就是 4 条 `ld.global.v4`/线程。
- **Q2「Triton 能达到 cuBLAS 水平吗」**：FP32 实测 87%（8.67 vs 9.95 @2048，
  config 扫描后）；教程默认 config 只有 ~6.4——**扫描/autotune 是必要环节**。
  同骨架 numba 手写只有 20%，差距主要是软件流水和 shared 访问模式的代码生成。
- **Q3「fp16 数据会自动用 tensor core 吗」**：不一定！我实测 triton 3.8 Windows
  wheel 在 sm_75 上 fp16 `tl.dot` 全部降级标量 FMA（26 config mma=0），
  比 fp32 还慢 1-23%；cuBLAS 同题 9.5×。上游源码有 Turing mma 路径但该构建
  未启用。**验证方法：数 PTX/SASS 里的 mma.sync，不信文档**。
- **Q4「autotune 怎么工作的」**：按 key（如 N）缓存——每个 key 首调用逐 config
  试跑选最优。我的扫描实验证明最优 config 随 N 迁移（附 fig7），所以 key 维度
  设计是必要的；且 fp16/fp32 最优形状不同——**autotune 必须在真实 lowering 下做**。
- **Q5「你的基准方法」**：编译/计时分离（WDDM 下编译让 GPU 掉频、污染紧随的
  计时——我实测 cuBLAS@2048 被打到 3.44 TF，修复后 9.95）+ 每尺寸 fresh burn
  + 时钟遥测 + 中位数。跨 AR 复跑对账（±5% 内）证明协议可比。

### 5.2 追问链

```
launch 为什么 triton 慢 ──→ Python launcher 层 ──→ WDDM 放大 ──→ 大 N 还慢吗（不，稳态相同 83%）
教程 softmax 差在哪 ──→ 缺向量化？PTX 数 v4 排除 ──→ 每行 bar.sync 规约 ──→ 占用率 2 CTA/SM 掩盖不了
naive 为什么慢 ──→ 流量模型 4× ──→ 实测只有 2.45× 为什么 ──→ 中间张量 4MB≈L2 被吸收 + 多 kernel 间隙 ──→ 放大工作集才逼近 4×
为什么 87% 不是 100% ──→ 同为 FFMA ──→ 流水/shared 模式差距 ──→ cuBLAS heuristic=预扫表
fp16 怎么验证走没走 TC ──→ 数 mma.sync（26 config 全 0）──→ 源码有路径构建未启用 ──→ 平台×版本矩阵
config 怎么选 ──→ 扫描 spread 2.67× ──→ 最优随 N 迁移 ──→ shared 预算剪枝 32 个 ──→ autotune per-key
```

### 5.3 数字卡片（本机实测口径）

| 数字 | 值 | 用途 |
| --- | --- | --- |
| launch 台阶 | torch 25.6 < triton 45.6 < numba 69.4 µs | 框架 launch 成本分层 |
| 大 N 带宽收敛 | 369-378 GB/s（82-84% HBM），三方 | 带宽型框架无关 |
| softmax 教程 vs 原生 | 142-154 vs 184-221 GB/s（1.2-1.6×） | 教程 ≠ 最优 |
| naive 实测比值 | 2.45×（vs 4× 流量模型上界；L2 吸收 4MB 中间张量） | 融合省流量 + L2 折扣 |
| FP32 @2048 | triton 8.67 / cuBLAS 9.95（87%）/ numba K2b 2.01（20%） | DSL 代码生成价值 |
| config 扫描 | 76 valid/32 pruned，spread 2.67×，最优 64×128×16 w4 s2 | autotune 必要性 |
| FP16 @2048 | triton 6.64（mma=0，比 fp32 -23%）vs cuBLAS 63.07（9.5×） | lowering 决定论 |
| PTX 证据 | fma=BM×BN×BK/threads（512@64×64×16 w4）、cvt.f32.f16×16 | 无 TC 铁证 |
| 时钟污染案例 | cuBLAS@2048 3.44 → 9.95 TF（+189%，协议修复） | 编译/计时分离 |

### 5.4 红线清单（本 AR 实际踩过/验证过的）

1. **`python -c` 里不能定义 `@triton.jit`**：inspect.getsourcelines 拿不到源码，
   报 "should be defined in a Python file"——kernel 必须落盘 .py。
2. **编译/计时不分离 = 时钟污染**：WDDM 下编译期 GPU 空闲掉频，紧随的计时被
   打到 3.44 TF（-65%）。修复协议：预编译全部 config → 烧机 → 背靠背计时。
   单靠「多跑几遍」救不了，必须物理隔离两个阶段。
3. **fp16 正确性门不能用绝对阈值**：我设 1e-2，但 1 ulp@128 = 0.125 必然超限
   ——门要按 1.5 ulp@max|ref| 原理化（AR003 E5 同款）。「门挂了」先检查门设计。
4. **「源码支持」≠「构建启用」**：上游有 Turing mma 路径，本 wheel 没有。
   特性可用性以 PTX/SASS 计数为准，文档只作线索。
5. **扫描相 vs 隔离相差 7%**（6.90 vs 6.40 TF 同 config）：背靠背循环里 L2/时钟
   处于热态。报数字必须带测量上下文。
6. **单格抖动复跑取证**：nw8@4096 一格 107.5 GB/s（同 config 其余 146-154）、
   numba@32M 首跑 10.4 ms 未复现——单点异常不进结论，复跑或标注。
7. **口算剪枝预算**：shared = stages×BK×(BM+BN)×4B ≤ 64KB，
   128×128×64×s2 = 192KB 直接出局——config 网格先剪再跑，省 30% 编译时间。
8. **fp16 输入 ≠ tensor core 被使用**：dtype 只改变数据布局，提速的前提是
   lowering 真的生成 mma——本 wheel 实测 26 config 全部标量 FMA，fp16 反而
   比 fp32 慢 1-23%。「换精度」不是开关，是「lowering 是否变」的问题。
9. **do_bench 与自建 Event 协议不可混比**：triton.testing.do_bench 的
   warmup/rep 策略与 AR002/003 的烧机+中位数协议不同（时钟态、样本口径都不同），
   跨协议数字放进同一张表 = 口径污染。本 AR 全程自建协议就是为了跨 AR 可比。

### 5.5 60 秒电梯陈述

> 「我在一张 Turing 卡上把 Triton 和 cuBLAS、numba 做了三方对照：带宽型 kernel
> 三方稳态完全一致（83% HBM），差距全在 launch 层（25/45/69µs 三台阶）；
> 计算型上 config 扫描后的 triton 达到 cuBLAS 的 87%，而同骨架手写 numba 只有
> 20%——这 4 倍就是 DSL 代码生成的价值。最有意思的是 fp16：我发现这个
> Windows wheel 在 sm_75 上把 fp16 dot 降级成标量 FMA——我数了 26 个 config 的
> PTX，mma.sync 一条都没有，fp16 反而比 fp32 慢，而 cuBLAS 走 tensor core 快
> 9.5 倍。上游源码里明明有 Turing mma 路径，所以我现在有个肌肉记忆：
> 「源码支持」不等于「你的二进制启用」，硬件特性必须用 PTX/SASS 计数验证。
> 顺带我踩出了一个 WDDM 基准坑：编译和计时不分离会让 GPU 掉频污染数据，
> 我被打到过 3.44 TF、修复协议后 9.95——这是 +189% 的方法论教训。」
