# -*- coding: utf-8 -*-
"""生成仓库封面 docs/cover.png（2400x760，matplotlib 绘制，无外部素材依赖）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "cover.png"
OUT.parent.mkdir(exist_ok=True)

zh = fm.FontProperties(family="Microsoft YaHei")
zh_bold = fm.FontProperties(family="Microsoft YaHei", weight="bold")

W, H = 24.0, 7.6
fig = plt.figure(figsize=(W, H), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ---------------------------------------------------------------- background
# vertical gradient: #070b16 -> #0e1830 -> #0a1122
grad = np.linspace(0, 1, 512).reshape(-1, 1)
c_top, c_mid, c_bot = np.array([7, 11, 22]), np.array([16, 28, 52]), np.array([8, 14, 30])
img = np.zeros((512, 2, 3))
for i, t in enumerate(np.linspace(0, 1, 512)):
    img[i, :, :] = c_top + (c_mid - c_top) * (t / 0.55) if t < 0.55 else c_mid + (c_bot - c_mid) * ((t - 0.55) / 0.45)
ax.imshow(img / 255.0, extent=[0, W, 0, H], aspect="auto", zorder=-10, origin="upper")

# subtle grid
rng = np.random.default_rng(7)
for x in np.arange(0, W + 0.6, 0.6):
    ax.plot([x, x], [0, H], color="#1c2a4a", lw=0.5, alpha=0.35, zorder=-9)
for y in np.arange(0, H + 0.6, 0.6):
    ax.plot([0, W], [y, y], color="#1c2a4a", lw=0.5, alpha=0.35, zorder=-9)

# accent glow bands (top-right, bottom-left)
for r, a in [(9.0, 0.10), (6.0, 0.08), (3.5, 0.05)]:
    ax.add_patch(plt.Circle((21.5, 6.8), r, color="#2f6bff", alpha=a, zorder=-8))
    ax.add_patch(plt.Circle((1.5, 0.4), r * 0.9, color="#7c3aed", alpha=a * 0.8, zorder=-8))

# ---------------------------------------------------------------- GPU die motif
# chip upper-left: 6x8 SM grid inside rounded square
chip_x, chip_y, chip_w, chip_h = 1.6, 3.10, 4.3, 3.30
ax.add_patch(FancyBboxPatch((chip_x, chip_y), chip_w, chip_h,
             boxstyle="round,pad=0.12,rounding_size=0.35",
             fc="#0b1226", ec="#3b82f6", lw=2.2, zorder=2))
# pins
pin_len = 0.42
for k in np.linspace(chip_x + 0.45, chip_x + chip_w - 0.45, 7):
    ax.plot([k, k], [chip_y + chip_h + 0.05, chip_y + chip_h + pin_len], color="#3b82f6", lw=2.0, zorder=1, solid_capstyle="round")
    ax.plot([k, k], [chip_y - 0.05, chip_y - pin_len], color="#3b82f6", lw=2.0, zorder=1, solid_capstyle="round")
for k in np.linspace(chip_y + 0.45, chip_y + chip_h - 0.45, 7):
    ax.plot([chip_x - 0.05, chip_x - pin_len], [k, k], color="#3b82f6", lw=2.0, zorder=1, solid_capstyle="round")
    ax.plot([chip_x + chip_w + 0.05, chip_x + chip_w + pin_len], [k, k], color="#3b82f6", lw=2.0, zorder=1, solid_capstyle="round")
# SM grid 6 x 8 (between TURING and TU104 labels)
sm_cols, sm_rows = 8, 6
gx0, gx1 = chip_x + 0.30, chip_x + chip_w - 0.30
gy0, gy1 = chip_y + 0.52, chip_y + chip_h - 0.54
gw = (gx1 - gx0) / sm_cols
gh = (gy1 - gy0) / sm_rows
heat = rng.random((sm_rows, sm_cols))
for i in range(sm_rows):
    for j in range(sm_cols):
        v = heat[i, j]
        if v > 0.82:
            fc = "#60a5fa"
        elif v > 0.55:
            fc = "#2563eb"
        else:
            fc = "#152447"
        ax.add_patch(Rectangle((gx0 + j * gw, gy0 + i * gh),
                               gw * 0.82, gh * 0.82, fc=fc, ec="#0b1226", lw=0.6, zorder=3))
ax.text(chip_x + chip_w / 2, chip_y + chip_h - 0.26, "TU104 · 48 SM · sm_75",
        ha="center", va="center", fontsize=11.5, color="#93c5fd", zorder=4,
        family="DejaVu Sans", fontweight="bold")
ax.text(chip_x + chip_w / 2, chip_y + 0.24, "TURING", ha="center", va="center",
        fontsize=10, color="#475569", zorder=4, family="DejaVu Sans", fontweight="bold")

# ---------------------------------------------------------------- throughput bar motif (below chip)
bar_x0, bar_y0, bar_w, bar_h = 1.6, 0.62, 4.3, 1.50
labels = ["K0", "K1", "K2", "FP32", "FP16"]
vals = [0.44, 0.89, 2.04, 9.47, 54.9]
colors = ["#334e87", "#334e87", "#3b82f6", "#60a5fa", "#22d3ee"]
bar_ymax = 60
for i, (lb, v, c) in enumerate(zip(labels, vals, colors)):
    bw = bar_w / len(vals) * 0.62
    x = bar_x0 + i * (bar_w / len(vals)) + (bar_w / len(vals) - bw) / 2
    hgt = (np.log10(v * 10 + 1) / np.log10(bar_ymax * 10 + 1)) * bar_h
    ax.add_patch(FancyBboxPatch((x, bar_y0), bw, max(hgt, 0.06),
                 boxstyle="round,pad=0.01,rounding_size=0.05", fc=c, ec="none", zorder=3))
    ax.text(x + bw / 2, bar_y0 - 0.18, lb, ha="center", va="center", fontsize=9.5,
            color="#64748b", zorder=4, family="DejaVu Sans", fontweight="bold")
ax.text(bar_x0 + bar_w / 2, bar_y0 + bar_h + 0.28, "GEMM 阶梯 0.44 → 54.9 TFLOPS",
        ha="center", va="center", fontsize=11.5, color="#93c5fd", zorder=4, fontweight="bold",
        fontproperties=zh)

# ---------------------------------------------------------------- title block
tx = 7.6

# small kicker
ax.add_patch(FancyBboxPatch((tx, 5.72), 4.35, 0.62,
             boxstyle="round,pad=0.06,rounding_size=0.16",
             fc="#12203f", ec="#3b82f6", lw=1.4, zorder=3))
ax.text(tx + 0.28, 6.03, "QUADRO RTX 5000", ha="left", va="center", fontsize=19,
        color="#7dd3fc", zorder=4, family="DejaVu Sans", fontweight="bold")

# main title with measured two-tone placement
t1 = ax.text(tx, 4.42, "单卡 AI Infra", ha="left", va="center", fontsize=82,
             color="#f1f5f9", fontproperties=zh_bold, zorder=4)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
x_end = ax.transData.inverted().transform((t1.get_window_extent(renderer).x1, 0))[0]
ax.text(x_end + 0.38, 4.42, "实验室", ha="left", va="center", fontsize=82,
        color="#38bdf8", fontproperties=zh_bold, zorder=4)

# subtitle
ax.text(tx, 3.28, "GPU 内核 · GEMM · DSL · 量化推理 · 训练 · 量化框架",
        ha="left", va="center", fontsize=25, color="#cbd5e1", fontproperties=zh, zorder=4)
ax.text(tx, 2.52, "八个章节 · 七个实验站 · 全部真机实测 · 预注册门 + 诚实修订",
        ha="left", va="center", fontsize=19, color="#7c8db5", fontproperties=zh, zorder=4)

# gradient accent rule
n_seg = 90
for i in range(n_seg):
    t = i / (n_seg - 1)
    r, g, b = 0.22 + 0.38 * t, 0.62 - 0.05 * t, 0.98 - 0.35 * t
    ax.add_patch(Rectangle((tx + i * (12.6 / n_seg), 2.02), 12.6 / n_seg + 0.02, 0.085,
                           color=(r, g, b), zorder=4))

# ---------------------------------------------------------------- stat chips
chips = [
    ("7", "实验站 AR001-007"),
    ("56", "张可解释实验图"),
    ("165+", "文档-JSON 数字断言"),
    ("0", "上游克隆改动"),
]
cw, ch, gap = 3.02, 1.28, 0.22
cy = 0.52
for i, (num, cap) in enumerate(chips):
    cx = tx + i * (cw + gap)
    ax.add_patch(FancyBboxPatch((cx, cy), cw, ch,
                 boxstyle="round,pad=0.05,rounding_size=0.18",
                 fc="#101b36", ec="#26375f", lw=1.3, zorder=3))
    ax.text(cx + 0.24, cy + ch / 2 + 0.12, num, ha="left", va="center", fontsize=34,
            color="#38bdf8", family="DejaVu Sans", fontweight="bold", zorder=4)
    ax.text(cx + 0.24, cy + 0.26, cap, ha="left", va="center", fontsize=12.5,
            color="#8ea3c8", fontproperties=zh, zorder=4)

# ---------------------------------------------------------------- footer
ax.text(W - 0.55, 0.42, "gitee.com/liu-xingyan04/quadro-rtx-5000-ai-infra",
        ha="right", va="center", fontsize=12.5, color="#3f4f75", family="DejaVu Sans", zorder=4)

# full-quality local copy (gitignored) + repo copy quantized to PNG-8
# (upload-channel size budget, see README)
hires = OUT.with_name("cover_hires.png")
fig.savefig(hires, dpi=100, facecolor="#070b16")
from PIL import Image
rgb = Image.open(hires).convert("RGB")
rgb.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG).save(
    OUT, "PNG", optimize=True)
print("wrote", OUT)
print("wrote", hires)
