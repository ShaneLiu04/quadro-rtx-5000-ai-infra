# triton-lab（AR004）— Triton DSL 三方对照 + sm_75 lowering 取证

在 Quadro RTX 5000（sm_75）上将 triton 3.8.0（Windows wheel）与 cuBLAS（torch）、
numba（AR002 kernels，venv 子进程背靠背）做同协议三方对照，并对 `tl.dot` 的
lowering 做 PTX 级取证。

## 交付物

| 文件 | 内容 |
|------|------|
| `bench_dsl.py` | 四组实验（E1/E2/E3/E4），编译/计时分离协议，内置正确性门（1.5 ulp@max\|ref\|） |
| `kernels_triton.py` | add / softmax / matmul kernel（教程复刻） |
| `numba_ref.py` | venv 侧 runner，sys.path 直连 AR002 `kernels_numba`（零复刻偏差） |
| `plot_results.py` | 9 张可解释图（英文图面，分析见 results.md） |
| `results/*.json` | 全部原始数据（e1/e1na/e2/e3/e3na/e3cfg/e4 + env 快照）+ `best_fp16_kernel.ptx`（fp16 最优 config PTX 证据存档） |
| `figs/*.png` | fig1 带宽 / fig2 launch / fig3 softmax / fig4 流量模型 / fig5 三方 TF / fig6 config 热图 / fig7 config 迁移 / fig8 fp16 TC 差距 / fig9 PTX 证据 |
| `results.md` | 逐图中文分析 + 数字一致性自查 |
| `../notes/triton-lowering-sm75.md` | lowering 拆解：管线 / block 模型对照 / PTX 证据 / MMAv2 源码定位 / autotune 机制 |
| `../notes/triton-notes.md` | 面试六段结构笔记 |

## 复现

```powershell
# 全局 python（triton 3.8 + torch 2.5.1+cu121 + matplotlib）
python bench_dsl.py --exp E1    # vector-add 三方（~1 分钟）
python bench_dsl.py --exp E2    # softmax 三方
python bench_dsl.py --exp E3    # matmul FP32 config 全扫描 + 三方（~1 分钟）
python bench_dsl.py --exp E4    # matmul FP16 + PTX 证据
python plot_results.py          # 9 张图
```

numba 侧经 venv 子进程自动调用（`01-foundations/GPU-Puzzles/solutions/.venv`），
JSON 标注 `same_session: true`。

## 实验矩阵与核心结论

| 实验 | 问题 | 核心结论（详见 results.md） |
|------|------|------------------------------|
| E1 | 带宽型 kernel 三方对照 | 大 N 全部收敛 369-378 GB/s（82-84% HBM）；launch 下限 torch 25.6 < triton 45.6 < numba 69.4 µs |
| E2 | 融合 softmax：教程 vs 原生 vs naive | 教程 142-154 GB/s 平台，原生快 1.2-1.6×；naive 5-pass 是流量 4× 模型上界被 L2 折扣（实测 2.45×，模型反推 239-252 GB/s） |
| E3 | FP32 matmul config 全扫描 | @2048 triton 8.67 TF = cuBLAS 9.95 的 87%，numba K2b 2.01（20%）；76 valid/32 pruned，spread 2.67×；最优 config 随 N 迁移 |
| E4 | FP16 lowering 取证 | 26 config × 3 尺寸全部 `mma.sync=0`（标量 FMA + `cvt.f32.f16`）；fp16 比 fp32 慢 1-23%；cuBLAS TC 9.5× 差距 @2048 |

## 协议（本 AR 方法论产出）

编译/计时分离：Phase A 预编译全部 config → 烧机拉频 → Phase B 背靠背计时
（cudaEvent，reps=10 中位数），每尺寸 fresh burn + nvidia-smi 遥测。
起因：E3 首跑 cuBLAS@2048 被编译期掉频污染至 3.44 TF（正常 9.5-10.3），
修复后 9.95 TF（+189%）。E1/E2 均以新协议复跑。

## 已知限制

- 图面口径：fig6 剪枝格（32 个 shared 超限）以空白表示（原因落 e3cfg.json
  oor_reason）；fig7 的 shared/wave 对账以 results.md §3 表格呈现（非图面）；
  fig1 L2 拐点按工作集 3N×4B = 4MB 口径（N≈350K）
- triton 3.8 wheel 的 fp16 dot 在 sm_75 未触发 mma（26 config 实测）；
  上游 3.9 源码存在 Turing 路径（MMAv2.cpp），本机无 nvcc/MSVC 无法构建验证
- torch.compile CUDA 路径不可用（torch 2.5.1 配对 triton 3.1.0），未纳入对照
- 显示 GPU（WDDM）会话噪声带沿用 AR002/AR003 口径；单格异常（如 nw8@4096
  107.5 GB/s）已标注，不进结论
- E2 教程 kernel 与原生差距的归因基于 PTX 取证 + 占用率口算（bar.sync 停顿 ×
  2 CTA/SM），未做 ncu 级验证（本机 Windows 无 ncu）
