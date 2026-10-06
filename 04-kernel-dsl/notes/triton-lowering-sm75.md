# Triton 在 sm_75 上的 Lowering 实测拆解（triton-lowering-sm75.md）

> AR004 F4 交付。所有断言基于本机实测（PTX 计数 / e3.json / e4.json）或
> 上游源码定位（triton 3.9 克隆，`04-kernel-dsl/triton/`）。
> 标注约定（沿用 AR003 纪律）：**[Local]** = 本机实测，附 JSON/PTX 依据；
> **[Source]** = 上游源码定位，附文件/行号；两者不混用、不互相推断。
> 环境：triton 3.8.0 Windows wheel + Quadro RTX 5000（sm_75）。

## 1. 编译管线：从 Python 函数到 cubin

```
@triton.jit kernel（Python 源码, inspect 取 AST）
  → TTIR   （块级 IR：tl.load/tl.dot 还是抽象块操作）
  → TTGIR  （GPU 化：分配到 warp/线程，插入 shared memory 与 sync）
  → LLVM IR → PTX → cubin（ptxas）
```

- 每层中间产物可经 `handle.asm["ptx"]` / `["ttir"]` / `["ttgir"]` 取出——
  本 AR 的全部证据来自对 PTX 的指令计数，不靠文档推断；
- 编译结果按（源码哈希 + 常量特化 + num_warps/num_stages）缓存：
  这就是 `@triton.jit` 必须定义在源文件里的原因——`inspect.getsourcelines`
  要读源码建 AST，`python -c` 里定义会报
  "could not get source code"（实测复现）；
- `num_warps`/`num_stages` 是 **launch/编译参数**而非 kernel 参数：
  同一份 kernel 源码，每个 (BLOCK, num_warps, num_stages) 组合编译出
  不同的 cubin——这是 config 扫描的物质基础。

## 2. block 编程模型 → CUDA C++ 心智对照 [本机]

| Triton | CUDA C++ | 说明 |
|---|---|---|
| `tl.program_id(0)` | `blockIdx.x` | 一个 program = 一个 CTA |
| `tl.arange(0, BLOCK)` | `threadIdx.x` + 循环展开 | 每线程负责 BLOCK/(warps×32) 个元素 |
| `tl.load(ptr + offs, mask=...)` | 边界判断 + `ld.global` | mask 编译为谓词；实测**不阻止 v4 向量化** |
| `tl.dot(a, b, acc)` | 手写 mma/FFMA + 寄存器分块 | lowering 决定走 TC 还是 FMA（§3） |
| `num_stages=s` | 手写双缓冲/多缓冲 | global→shared 软件流水，triton 自动生成 |
| `tl.store(..., mask)` | 条件写回 | 同上谓词化 |
| GROUP_M 分组 | swizzle 计算 | CTA 排布重排，改善 L2 局部性（实测二阶效应，±5%） |

关键心法：**你写的是块级程序，编译器决定每个线程干什么**。
BLOCK=4096、num_warps=8 时每线程 16 个元素——PTX 里对应 4 条
`ld.global.v4.b32`（softmax kernel 实测：`ld.global.v4` ×4，n_regs=37，0 spills）。

## 3. sm_75 实测 lowering：FP32 = FFMA，FP16 dot = 标量 FMA（无 TC）[本机]

### 3.1 FP16 证据（E4，26 config × 3 尺寸全部一致）[本机]

对每个 config 的 PTX 计数（落 e4.json）：

| 指标 | 实测值 | 含义 |
|---|---|---|
| `mma.sync` | **0**（26/26 config） | 未走 Tensor Core |
| `ldmatrix` | **0**（26/26） | TC 的配套数据搬运也未出现 |
| `fma.rn.f32` | 512（64×64×16 w4） | 标量 FMA，静态计数 = BM×BN×BK/threads |
| `cvt.f32.f16` | 16 | fp16 → fp32 逐元素转换后进 FMA |
| `n_spills` | 0 | 小 tile 无寄存器溢出 |

循环体 PTX 实录（64×64×16 w4 s2，全局→shared 流水正常 `ld.shared.v4.b32`）：

```
ld.shared.v4.b32  {%r244, %r245, %r246, %r247}, [%r9+2560];   // A/B tile 进寄存器
...
fma.rn.f32  %r268, %r200, %r207, %r819;                      // fp32 标量乘加
fma.rn.f32  %r269, %r200, %r206, %r818;
```

即：**数据通路（global→shared→寄存器）是向量化的、健康的；
只有 dot 操作被降级为标量 FMA**。后果（e4.json）：

- triton FP16 最优 2.90/5.38/6.64 TF，**比 FP32 还慢 1-23%**
  （fp16→fp32 转换是净开销，TC 收益为零）；
- cuBLAS FP16（走 mma.sync m16n8k8）同题 8.74/22.47/63.07 TF，
  @2048 差距 **9.5×**；
- 小 tile（64×64×16 w4，196 regs）全面获胜：FMA 路径没有 TC 分摊
  寄存器压力的机制，大 tile 的 fp32 累加器直接压垮吞吐。

### 3.2 FP32 路径：FFMA 是预期（Turing 无 TF32）

FP32 dot → `fma.rn.f32`（FFMA）。E3 config 扫描 @1024 最优 6.90 TF，
三方隔离 @2048 最优 8.67 TF = FP32 峰值 11.2 TF 的 77%，
cuBLAS（同为 FFMA 路径）9.95 TF = 89%。**同为标量 FMA，差距 12 个点
全部来自 tiling/软件流水的代码生成质量**。

### 3.3 softmax：非计算密集 kernel 的 lowering 健康性

教程 softmax kernel（BLOCK=4096, nw8）：`ld.global.v4`×4、
`ex2.approx`×16（exp 降级为快速近似，编译器自动选择）、
`bar.sync`×4 + shared 往返（block 规约）、37 regs / 0 spills。
排除「缺向量化」后，其 142-154 GB/s 平台（vs 原生 1.2-1.6×）的
剩余机制是每行 block 规约的 barrier 停顿在 2 CTA/SM 占用率下无法掩盖。

## 4. 上游源码定位：MMAv2 的 Turing 路径存在但本 wheel 未启用 [源码]

`triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp`（3.9 克隆）：

- `mmaInstrPtxTuring(...)`：为 sm_75 生成 `mma.sync.aligned.m16n8k8`
  fp16 指令模板（L337-463 区域）；
- `callMmaTuringFp16(...)`：fp16 dot 在 Turing 上的完整 lowering 路径。

即 **上游源码「写着」支持 sm_75 fp16 mma**，但本机 triton 3.8.0
Windows wheel 的实测 PTX 里一条都没有。两种可能（未进一步二分定位，
因为构建不可复现——无 nvcc/MSVC 工具链）：版本差异（3.8 vs 3.9）或
Windows 构建矩阵的路径选择。无论哪种：

> **「源码里有」≠「你的二进制里有」。特性可用性必须用 PTX/SASS 计数验证。**
> 这条方法论比结论本身更值钱——面试里它能展开成
> 「你在 Windows+旧架构组合上怎么确认 TC 有没有被用上」的完整故事。

## 5. autotune 机制与手动扫描

- `@triton.autotune(configs, key=['N'])`：每个 key 组合的**首次调用**
  逐 config 试跑、缓存最优——之后同 key 复用；
- 与 AR004 手动扫描的差别：手动扫描能**逐 config 落 JSON**
  （含 32 个超限 config 的 oor_reason：shared = stages×BK×(BM+BN)×4B > 64KB），
  能画出 fig6 热图与 fig7 迁移曲线；autotune 只给你一个黑盒最优；
- 实测最优 config 随 N 迁移（64×128×32_w8 → 64×128×16_w4 → 128×64×16_w4_s3），
  印证 autotune 的 key 维度设计（per-size）是必要的；
- E4 的教训反推：autotune **必须在真实 lowering 下做**——fp16 的最优
  config 形状（64×64×16）与 fp32 完全不同，因为 lowering 路径不同。

## 6. 数字卡片（本文档口径）

- launch 下限：torch 25.6 < triton 45.6 < numba 69.4 µs（WDDM 台阶）
- FP32 FFMA 峰值利用：triton 77% / cuBLAS 89%（@2048）
- FP16：mma=0（26/26 config）、fma=512 = BM×BN×BK/threads、
  TC 差距 9.5×、fp16 比 fp32 慢 1-23%
- config 扫描 spread 2.67×（76 valid / 32 shared 超限）
- 最优 config 随 N 迁移：大 tile 需配小 BK（寄存器/shared 预算）
