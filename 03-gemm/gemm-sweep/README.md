# gemm-sweep（AR003）— cuBLAS 形状空间扫描 + CUTLASS Turing 拆解

在 Quadro RTX 5000（sm_75）上系统扫描 cuBLAS 在**形状 × 调度 × 数据类型**空间的行为，
并拆解 CUTLASS `08_turing_tensorop_gemm`（本机精确目标硬件的官方 INT8 TC 参考实现）。

## 交付物

| 文件 | 内容 |
|------|------|
| `bench_sweep.py` | 八组实验（E1a/E1b/E2/E3/E3seq/E4/E5/E6/E7），内置正确性门 |
| `plot_results.py` | 8 张可解释图（英文图面，分析见 results.md） |
| `results/*.json` | 全部原始数据 + env 快照（int_mm 探测结果） |
| `figs/*.png` | fig1a 热图 / fig1b wave 阶梯 / fig2 skinny 双轴 / fig3 split-K / fig4 batch / fig5 dtype / fig6 L2 / fig7 layout |
| `results.md` | 逐图中文分析 + 汇总口径表 |
| `../notes/gemm-sweep-notes.md` | 面试六段结构笔记 |
| `../notes/cutlass-turing-dissection.md` | CUTLASS 08_turing 源码拆解（15 模板参数 / shared 预算因果 / mma→TOPS 换算链） |

## 复现

```powershell
# 全局 python（torch 2.5.1+cu121 + matplotlib），无需 venv
python bench_sweep.py --exp probes     # env 探测（int_mm 可用性）
python bench_sweep.py --exp all        # 七组实验约 2-3 分钟（含 2s 烧机）
python plot_results.py                 # 生成 8 张图
```

也可单跑：`--exp E3`、`--exp E1a+E1b` 等。

## 实验矩阵

| 实验 | 问题 | 核心结论（详见 results.md） |
|------|------|------------------------------|
| E1a/E1b | 形状网格 + wave 量化 | M,N≥512 区 16 格中 11 格 9.5-10.24 TF（中位 9.71），3 格启发式凹陷 8.58-8.90；跨波不满形状最差（-10%）；≥4 波损失 <5% |
| E2 | skinny 边界 | M≤32 带宽平台 360-386 GB/s（80-86% HBM）；拐点 M*≈51 与 roofline 吻合 |
| E3 | split-K 手工 vs 库 | cuBLAS 已内部 split-K（8.2 vs 0.93 TF 饥饿界）；手工全败，且 stream 比顺序对照（E3seq）还慢 -9.8%~-43.9% |
| E4 | batch vs flat | flat 一致赢（B=64 +13%）；bmm 仅在无法堆叠时用 |
| E5 | dtype 断层 | FP32→FP16 6.4-6.9×@2048（两跑）；FP16 峰值达成 68-76%；FP16 跨会话 ±10%（时钟态） |
| E6 | L2 驻留 | 仅 ws<4MB 时 ×1.12；GEMM 迭代内 L2 复用是主要机制 |
| E7 | layout | 七组合极差 5%；「TN 显著更快」在本机不成立；为对齐做 copy 必亏 |

## 已知限制

- INT8 TC 未能实测：Windows torch 2.5.1 构建缺 `torch._int_mm` 路径
  （`addmm_cuda not implemented for Int`，env.json 留档）；sm_75 硬件本身支持（CUTLASS 08 即 INT8 示例）
- CUTLASS 不编译运行（本机无 nvcc/MSVC）：只做源码拆解 + 文档对照，笔记区分「源码事实」与「本机实测」
- wave 模型 tile 取 128×128 为可视化假设（cuBLAS 实际多 tile 尺寸，见 fig1a 红线为模型边界）
- 显示 GPU（WDDM）会话噪声带沿用 AR002 口径；E6 的 ±2% 级差异只报方向不判显著

## 与 AR002 的衔接

- FP32@2048：跨观测带 9.47~10.46（AR002 9.468 + 本 AR 两次完整 E5）——大形状方向稳健
- FP16@2048：跨会话带 54.9~67.4（AR002 54.87 轻预热 / 本 AR 重烧机两跑 65.46、67.43）——
  本 AR 发现 FP16 TC 跨会话时钟态敏感带 ±10%（FP32 大形状仅 ±1.5%），且**噪声带宽度
  与测量窗口长度相关**（@4096 复跑差 <1.5%，@2048 差 10%——1.7ms×10reps 窗口短于时钟
  爬坡），详见 results.md fig5 节
- bench 纪律（Event/烧机/中位数/会话噪声口径）全继承 gemm-lab

上游 `03-gemm/cutlass/` 零改动（本目录与 notes/ 为独立新增）。
