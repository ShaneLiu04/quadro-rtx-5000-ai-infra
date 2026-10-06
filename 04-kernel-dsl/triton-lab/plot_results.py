"""AR004 triton-lab: generate all figures from results/*.json (T006).

Figures (>=8 required by srs F3):
  fig1  E1 three-framework bandwidth vs N (vector add)
  fig2  E1 small-N launch overhead stratification
  fig3  E2 softmax effective bandwidth, four variants
  fig4  E2 naive 5-pass traffic model (effective vs ~4x DRAM estimate)
  fig5  E3 three-framework TFLOPS vs N + FP32 peak
  fig6  E3 config sweep heatmap @N=1024 (76 valid configs)
  fig7  E3 top-5 config performance across N (config migration)
  fig8  E4 fp16: triton vs cuBLAS (+ fp32 refs), TC gap annotation
  fig9  E4 PTX evidence: n_regs vs TFLOPS, all configs mma=0
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

FP32_PEAK = 11.2
FP16_TC_PEAK = 89.2
HBM_GBPS = 448.0
NUM_SM = 48

# framework colors (consistent across all figures)
C_TORCH = "tab:blue"
C_TRITON = "tab:orange"
C_NUMBA = "tab:green"
C_NAIVE = "tab:red"

plt.rcParams.update({"figure.dpi": 300, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.titlesize": 10})


def load(name):
    with open(os.path.join(RESULTS, name), "r", encoding="utf-8") as f:
        return json.load(f)


def series(recs, variant, xkey, ykey):
    rs = sorted([r for r in recs if r["variant"] == variant], key=lambda r: r[xkey])
    return [r[xkey] for r in rs], [r[ykey] for r in rs]


def fig1_e1_bandwidth():
    doc = load("e1.json")
    recs = doc["records"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for variant, label, color in (
        ("torch", "torch add_", C_TORCH),
        ("triton_bs1024", "triton BLOCK=1024", C_TRITON),
        ("triton_bs256", "triton BLOCK=256", C_TRITON),
        ("triton_bs4096", "triton BLOCK=4096", C_TRITON),
    ):
        xs, ys = series(recs, variant, "n", "gbps")
        ls = "-" if variant == "triton_bs1024" else ":"
        ax.plot(xs, ys, ls, marker="o", ms=3, lw=1.2, color=color, label=label)
    ax.axhline(HBM_GBPS, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(5e3, 460, "HBM 448 GB/s", fontsize=7)
    ax.axhline(0.83 * HBM_GBPS, color="k", lw=0.6, ls=":", alpha=0.5)
    ax.text(5e3, 355, "83% (372 GB/s)", fontsize=7)
    # L2 4MB crossing: working set = 3 arrays x N x 4B -> N ~= 4194304/12
    n_l2 = 4194304 / 12
    ax.axvline(n_l2, color="tab:red", lw=0.7, ls="--", alpha=0.6)
    ax.text(n_l2 * 1.15, 120, "L2 4MB\n(3N x 4B)", fontsize=6.5, color="tab:red")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (fp32 elements)")
    ax.set_ylabel("GB/s (3N x 4B / time)")
    ax.set_title("E1: vector-add bandwidth, three frameworks\n"
                 "large N: all converge to ~369-378 GB/s (83% HBM)")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(os.path.join(FIGS, "fig1_e1_bandwidth.png"), bbox_inches="tight")
    plt.close(fig)


def fig2_e1_launch():
    doc = load("e1.json")
    recs = [r for r in doc["records"] if r["n"] <= 262144]
    docna = load("e1na.json")
    recs += [r for r in docna["records"] if r["n"] <= 262144]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for variant, label, color in (
        ("torch", "torch", C_TORCH),
        ("triton_bs1024", "triton BLOCK=1024", C_TRITON),
        ("numba", "numba", C_NUMBA),
    ):
        xs, ys = series(recs, variant, "n", "time_ms")
        ax.plot(xs, np.array(ys) * 1e3, marker="o", ms=3, lw=1.2, color=color, label=label)
    ax.axhline(25.6, color=C_TORCH, lw=0.6, ls=":", alpha=0.6)
    ax.axhline(45.6, color=C_TRITON, lw=0.6, ls=":", alpha=0.6)
    ax.axhline(69.4, color=C_NUMBA, lw=0.6, ls=":", alpha=0.6)
    ax.text(4.6e3, 27.2, "~25.6 us", fontsize=7, color=C_TORCH)
    ax.text(4.6e3, 47.2, "~45.6 us", fontsize=7, color=C_TRITON)
    ax.text(4.6e3, 71, "~69.4 us", fontsize=7, color=C_NUMBA)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (fp32 elements)")
    ax.set_ylabel("kernel time (us)")
    ax.set_title("E1: small-N launch overhead floor\n"
                 "torch ~25.6 < triton ~45.6 < numba ~69.4 us (WDDM driver stack)")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIGS, "fig2_e1_launch.png"), bbox_inches="tight")
    plt.close(fig)


def fig3_e2_softmax():
    doc = load("e2.json")
    recs = doc["records"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for variant, label, color, ls in (
        ("triton_nw4", "triton tutorial nw=4", C_TRITON, "-"),
        ("triton_nw8", "triton tutorial nw=8", C_TRITON, ":"),
        ("torch_native", "torch native softmax", C_TORCH, "-"),
        ("naive_5pass", "naive 5-pass (torch ops)", C_NAIVE, "-"),
    ):
        xs, ys = series(recs, variant, "N", "gbps_eff")
        ax.plot(xs, ys, ls, marker="o", ms=3, lw=1.2, color=color, label=label)
    ax.axhline(HBM_GBPS, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(1100, 460, "HBM 448 GB/s", fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("row width N (M=1024 rows)")
    ax.set_ylabel("effective GB/s (2MN bytes / time)")
    ax.set_title("E2: fused softmax bandwidth\n"
                 "triton ~142-154 GB/s flat; native 1.2-1.6x faster; naive = multi-pass, 2.4-2.5x slower")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIGS, "fig3_e2_softmax.png"), bbox_inches="tight")
    plt.close(fig)


def fig4_e2_traffic():
    doc = load("e2.json")
    recs = [r for r in doc["records"] if r["variant"] == "naive_5pass"]
    recs = sorted(recs, key=lambda r: r["N"])
    # naive 5-pass traffic model: x read twice, exp tensor e written+read 3x
    # => total bytes ~ 4x fused (2x read x + 3r+3w on e vs 1r+1w fused)
    mult = 4.0
    xs = [r["N"] for r in recs]
    eff = [r["gbps_eff"] for r in recs]
    dram = [g * mult for g in eff]
    x = np.arange(len(xs))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - 0.2, eff, 0.38, color=C_NAIVE, alpha=0.85, label="effective (2MN bytes)")
    ax.bar(x + 0.2, dram, 0.38, color="tab:purple", alpha=0.85,
           label="model upper bound (x4 traffic, if all from DRAM)")
    ax.axhline(HBM_GBPS, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.05, 460, "HBM 448 GB/s", fontsize=7)
    for i, g in enumerate(dram):
        ax.text(i + 0.2, g + 6, f"{g:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in xs])
    ax.set_xlabel("row width N")
    ax.set_ylabel("GB/s")
    ax.set_title("E2: naive 5-pass softmax traffic model\n"
                 "x4 traffic model is an upper bound; intermediates (4MB = L2) partially absorbed;\n"
                 "measured effective ratio only 2.4-2.5x vs triton")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIGS, "fig4_e2_traffic.png"), bbox_inches="tight")
    plt.close(fig)


def fig5_e3_threeway():
    doc = load("e3.json")
    recs = doc["records"]
    docna = load("e3na.json")
    recs = recs + docna["records"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for variant, label, color, marker in (
        ("cublas", "cuBLAS (torch mm)", C_TORCH, "o"),
        ("triton_best_gm1", "triton best config", C_TRITON, "s"),
        ("K2b_regblock_64x64_4x4", "numba K2b regblock (AR002)", C_NUMBA, "^"),
        ("K1_tiled_T16", "numba K1 tiled (AR002)", C_NUMBA, "v"),
    ):
        xs, ys = series(recs, variant, "N", "tflops")
        ax.plot(xs, ys, marker=marker, ms=4, lw=1.2, color=color, label=label)
    ax.axhline(FP32_PEAK, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(270, 10.6, "FP32 peak 11.2 TF", fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (NxNxK FP32)")
    ax.set_ylabel("TFLOPS")
    ax.set_title("E3: FP32 matmul, three frameworks\n"
                 "@2048: triton 8.67 = 87% of cuBLAS 9.95; numba K2b 2.01 (20%)")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(os.path.join(FIGS, "fig5_e3_threeway.png"), bbox_inches="tight")
    plt.close(fig)


def fig6_e3_heatmap():
    doc = load("e3cfg.json")
    recs = [r for r in doc["records"] if r["N"] == 1024 and "tflops" in r]
    tiles = sorted({(r["BM"], r["BN"]) for r in recs})
    deeps = sorted({(r["BK"], r["num_warps"], r["num_stages"]) for r in recs})
    grid = np.full((len(tiles), len(deeps)), np.nan)
    for r in recs:
        i = tiles.index((r["BM"], r["BN"]))
        j = deeps.index((r["BK"], r["num_warps"], r["num_stages"]))
        grid[i, j] = r["tflops"]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    im = ax.imshow(grid, cmap="viridis", vmin=2.5, vmax=7.0, aspect="auto")
    for i in range(len(tiles)):
        for j in range(len(deeps)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                        fontsize=6, color="white" if grid[i, j] < 5.5 else "black")
    ax.set_xticks(range(len(deeps)))
    ax.set_xticklabels([f"BK{b}\nw{w}\ns{s}" for b, w, s in deeps], fontsize=6.5)
    ax.set_yticks(range(len(tiles)))
    ax.set_yticklabels([f"{bm}x{bn}" for bm, bn in tiles], fontsize=7)
    ax.set_xlabel("(BK, num_warps, num_stages)")
    ax.set_ylabel("tile (BM x BN)")
    ax.set_title("E3: triton FP32 config sweep heatmap @N=1024 (76 valid / 32 pruned)\n"
                 "best 6.90 TF = 64x128 BK16 w4 s2; worst 2.59 TF; spread 2.67x")
    fig.colorbar(im, ax=ax, label="TFLOPS")
    fig.savefig(os.path.join(FIGS, "fig6_e3_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def fig7_e3_cfgshift():
    doc = load("e3cfg.json")
    recs = doc["records"]
    at1024 = sorted([r for r in recs if r["N"] == 1024 and "tflops" in r],
                    key=lambda r: -r["tflops"])
    top5 = [(r["BM"], r["BN"], r["BK"], r["num_warps"], r["num_stages"])
            for r in at1024[:5]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, (bm, bn, bk, w, s) in enumerate(top5):
        rs = [r for r in recs
              if (r["BM"], r["BN"], r["BK"], r["num_warps"], r["num_stages"]) == (bm, bn, bk, w, s)
              and "tflops" in r]
        rs = sorted(rs, key=lambda r: r["N"])
        ax.plot([r["N"] for r in rs], [r["tflops"] for r in rs],
                marker="o", ms=3, lw=1.2, color=f"C{k}",
                label=f"{bm}x{bn}x{bk} w{w} s{s}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("TFLOPS")
    ax.set_title("E3: top-5 configs (ranked @1024) across sizes\n"
                 "best config migrates with N - autotune is per-size, not universal")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIGS, "fig7_e3_cfgshift.png"), bbox_inches="tight")
    plt.close(fig)


def fig8_e4_fp16():
    doc = load("e4.json")
    recs = doc["records"]
    best = {}
    for r in recs:
        if "BM" in r:
            best[r["N"]] = max(best.get(r["N"], 0.0), r["tflops"])
    cub_f16 = {r["N"]: r["tflops"] for r in recs if r.get("variant") == "cublas_fp16"}
    e3 = load("e3.json")["records"]
    cub_f32 = {r["N"]: r["tflops"] for r in e3 if r["variant"] == "cublas"}
    tri_f32 = {r["N"]: r["tflops"] for r in e3 if r["variant"] == "triton_best_gm1"}
    ns = sorted(best)
    x = np.arange(len(ns))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for off, vals, label, color in (
        (-0.30, [cub_f32[n] for n in ns], "cuBLAS FP32", C_TORCH),
        (-0.10, [tri_f32[n] for n in ns], "triton FP32", C_TRITON),
        (0.10, [best[n] for n in ns], "triton FP16 (best of 26)", C_TRITON),
        (0.30, [cub_f16[n] for n in ns], "cuBLAS FP16 (tensor core)", "tab:purple"),
    ):
        ax.bar(x + off, vals, 0.18, label=label, color=color, alpha=0.85)
    ax.axhline(FP16_TC_PEAK, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(len(ns) - 0.6, FP16_TC_PEAK + 1, "FP16 TC peak 89.2 TF", fontsize=7, ha="right")
    for i, n in enumerate(ns):
        gap = cub_f16[n] / best[n]
        ax.text(i + 0.10, best[n] + 0.8, f"{best[n]:.2f}", ha="center", fontsize=6.5)
        ax.annotate(f"{gap:.1f}x", xy=(i + 0.30, cub_f16[n]),
                    xytext=(i + 0.34, cub_f16[n] * 0.55),
                    fontsize=6.5, color="tab:purple",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="tab:purple"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("N (NxNxK)")
    ax.set_ylabel("TFLOPS")
    ax.set_title("E4: FP16 matmul - triton wheel lowers to scalar FMA on sm_75\n"
                 "mma.sync = 0 in all 26 configs; triton FP16 even slower than its FP32;\n"
                 "cuBLAS FP16 hits tensor cores (9.5x gap @2048)")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIGS, "fig8_e4_fp16.png"), bbox_inches="tight")
    plt.close(fig)


def fig9_e4_ptx():
    doc = load("e4.json")
    recs = [r for r in doc["records"] if r["N"] == 2048 and "BM" in r]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in recs:
        bm, bn, bk, w = r["BM"], r["BN"], r["BK"], r["num_warps"]
        per_thread = bm * bn * bk / (w * 32)
        ax.scatter(r["n_regs"], r["tflops"], s=30, color=C_TRITON,
                   alpha=0.8, edgecolors="none")
        ax.annotate(f"{bm}x{bn}x{bk}", (r["n_regs"], r["tflops"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=5.5)
    ax.set_xlabel("n_regs (registers per thread)")
    ax.set_ylabel("TFLOPS @N=2048")
    ax.set_title("E4: PTX evidence per fp16 config (26 configs)\n"
                 "all: mma.sync=0, ldmatrix=0, spills=0; FMA count = BM*BN*BK/threads\n"
                 "performance peaks at small tiles (64x64x16 w4, 196 regs)")
    fig.savefig(os.path.join(FIGS, "fig9_e4_ptx.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    fig1_e1_bandwidth()
    fig2_e1_launch()
    fig3_e2_softmax()
    fig4_e2_traffic()
    fig5_e3_threeway()
    fig6_e3_heatmap()
    fig7_e3_cfgshift()
    fig8_e4_fp16()
    fig9_e4_ptx()
    print("wrote 9 figures ->", FIGS)


if __name__ == "__main__":
    main()
