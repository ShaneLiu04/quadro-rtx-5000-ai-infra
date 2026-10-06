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
FP16_PEAK = 89.2
INT8_PEAK = 178.4
HBM_GBPS = 448.0
NUM_SM = 48
L2_MB = 4.0

plt.rcParams.update({"figure.dpi": 300, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.titlesize": 10})


def load(name):
    with open(os.path.join(RESULTS, name), "r", encoding="utf-8") as f:
        return json.load(f)


def fig1a():
    doc = load("e1a.json")
    recs = doc["experiments"]["E1a"]["records"]
    sizes = sorted({r["M"] for r in recs})
    grid = np.zeros((len(sizes), len(sizes)))
    tiles = np.zeros_like(grid)
    for r in recs:
        i = sizes.index(r["M"])
        j = sizes.index(r["N"])
        grid[i, j] = r["tflops"]
        tiles[i, j] = r["model_tiles"]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(grid, origin="lower", cmap="viridis",
                   extent=[sizes[0] / 2, sizes[-1] * 1.5, sizes[0] / 2, sizes[-1] * 1.5],
                   norm=matplotlib.colors.LogNorm(vmin=0.5, vmax=grid.max()))
    for r in recs:
        x = sizes.index(r["N"])
        y = sizes.index(r["M"])
        ax.text(sizes[x], sizes[y], f"{r['tflops']:.1f}", ha="center", va="center",
                fontsize=6.5, color="white" if r["tflops"] < 6 else "black")
    wave = np.ceil(tiles / NUM_SM)
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if j + 1 < len(sizes) and wave[i, j] != wave[i, j + 1]:
                ax.axvline((sizes[j + 1] + sizes[j]) / 2 if sizes[j + 1] / sizes[j] == 2
                           else sizes[j] + 32, color="red", lw=0.4, alpha=0.7)
            if i + 1 < len(sizes) and wave[i, j] != wave[i + 1, j]:
                ax.axhline(sizes[i] + 32 if sizes[i + 1] / sizes[i] != 2 else (sizes[i + 1] + sizes[i]) / 2,
                           color="red", lw=0.4, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_yticks(sizes)
    lims = [sizes[0] / 2, sizes[-1] * 1.5]
    ax.plot(lims, lims, color="white", lw=0.8, ls="--", alpha=0.8)
    ax.text(sizes[2], sizes[2] * 1.35, "M = N (diagonal)", color="white", fontsize=6,
            rotation=0, ha="left", va="bottom")
    ax.set_xlabel("N")
    ax.set_ylabel("M")
    ax.set_title("E1a: cuBLAS FP32 TFLOPS over M x N grid (K=2048)\n"
                 "red lines = model wave boundaries (128x128 tile, 48 SMs)")
    fig.colorbar(im, ax=ax, label="TFLOPS")
    fig.savefig(os.path.join(FIGS, "fig1a_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def fig1b():
    doc = load("e1b.json")
    recs = doc["experiments"]["E1b"]["records"]
    m = [r["M"] for r in recs]
    tf = [r["tflops"] for r in recs]
    tiles = [r["model_tiles"] for r in recs]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(m, tf, "o-", color="tab:blue", label="measured TFLOPS")
    ax.axhline(FP32_PEAK, color="gray", ls="--", lw=1, label=f"FP32 peak {FP32_PEAK} TFLOPS")
    prev_w = None
    for x, t in zip(m, tiles):
        w = int(np.ceil(t / NUM_SM))
        if w != prev_w:
            ax.axvline(x, color="red", lw=0.5, alpha=0.6)
            ax.text(x, 1.2, f"tiles={t}\n={w} wave" + ("s" if w > 1 else ""),
                    fontsize=6, color="red", ha="left", va="bottom")
            prev_w = w
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M (N=K=4096)")
    ax.set_ylabel("TFLOPS")
    ax.set_title("E1b: wave quantization staircase (fine M sweep)\n"
                 "red = model tile-count changes (128x128 tile, 48 SMs)")
    fig.legend(loc="lower right", fontsize=7)
    fig.savefig(os.path.join(FIGS, "fig1b_staircase.png"), bbox_inches="tight")
    plt.close(fig)


def fig2():
    doc = load("e2.json")
    recs = doc["experiments"]["E2"]["records"]
    m = np.array([r["M"] for r in recs])
    tf = [r["tflops"] for r in recs]
    gb = [r["gbps"] for r in recs]

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax2 = ax1.twinx()
    ax2.grid(False)
    l1, = ax1.plot(m, tf, "o-", color="tab:orange", label="TFLOPS")
    l2, = ax2.plot(m, gb, "s-", color="tab:blue", label="effective GB/s")
    ax2.axhline(HBM_GBPS, color="blue", ls="--", lw=1)
    ax2.text(1.05, HBM_GBPS * 0.97, "HBM peak 448 GB/s", color="blue", fontsize=7, va="top")
    ax1.axhline(FP32_PEAK, color="darkorange", ls="--", lw=1)
    ax1.text(1024, FP32_PEAK * 1.02, "FP32 peak 11.2 TFLOPS", color="darkorange", fontsize=7)
    m_star = 2048 * 102400 / (1998 * 1) * 0 + 51.2
    ax1.axvline(m_star, color="green", ls=":", lw=1.2)
    ax1.text(m_star * 1.1, 4.5, "roofline crossover\nAI=25 FLOP/B at M*≈51", color="green", fontsize=7)
    ax1.axvspan(1, 64, color="blue", alpha=0.05)
    ax1.text(2.2, 10.6, "GEMV regime\n(bandwidth-bound)", fontsize=7, color="blue")
    ax1.text(300, 3.2, "compute regime", fontsize=7, color="darkorange")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("M (N=K=4096, FP32)")
    ax1.set_ylabel("TFLOPS", color="tab:orange")
    ax2.set_ylabel("GB/s", color="tab:blue")
    ax1.set_title("E2: skinny GEMM boundary — bandwidth plateau vs compute plateau")
    ax1.legend(handles=[l1, l2], loc="center right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_skinny.png"))
    plt.close(fig)


def fig3():
    doc = load("e3.json")
    recs = doc["experiments"]["E3"]["records"]
    s = [r["split_s"] for r in recs]
    tf = [r["tflops"] for r in recs]
    base = tf[0]

    seq = {2: None, 4: None, 8: None, 16: None}
    try:
        doc_seq = load("e3seq.json")
        for r in doc_seq["experiments"]["E3seq"]["records"]:
            seq[r["split_s"]] = r["tflops"]
    except (FileNotFoundError, KeyError):
        pass

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if all(v is not None for v in seq.values()):
        x = np.arange(len(seq))
        w = 0.38
        ax.bar(x - w / 2, [seq[k] for k in sorted(seq)], w, color="tab:purple",
               label="manual split, sequential (default stream)")
        stream_map = {r["split_s"]: r["tflops"] for r in recs}
        ax.bar(x + w / 2, [stream_map[k] for k in sorted(seq)], w, color="tab:blue",
               label="manual split, s CUDA streams (parallel intent)")
        for xi, k in zip(x, sorted(seq)):
            ax.text(xi - w / 2, seq[k] + 0.12, f"{seq[k]:.2f}", ha="center", fontsize=6.5, color="tab:purple")
            ax.text(xi + w / 2, stream_map[k] + 0.12, f"{stream_map[k]:.2f}", ha="center", fontsize=6.5,
                    color="tab:blue")
        ax.set_xticks(x)
        ax.set_xticklabels([f"s={k}" for k in sorted(seq)])
    else:
        bars = ax.bar([str(x) for x in s], tf,
                      color=["tab:green"] + ["tab:blue"] * (len(s) - 1))
        for b, v in zip(bars, tf):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(base, color="tab:green", ls="--", lw=1)
    ax.text(0.05, base + 0.25, f"s=1 baseline {base:.2f} TFLOPS (cuBLAS internal split-K)",
            color="tab:green", fontsize=7)
    starve_bound = 4 * FP32_PEAK / NUM_SM
    ax.axhline(starve_bound, color="red", ls=":", lw=1.2)
    ax.text(0.05, starve_bound + 0.25,
            f"4-tile starved bound {starve_bound:.2f} TFLOPS\n(baseline far above => cuBLAS internal split-K active)",
            color="red", fontsize=6.5)
    ax.set_xlabel("manual split-K factor s (M=N=256, K=16384)")
    ax.set_ylabel("TFLOPS")
    ax.set_title("E3: manual split-K vs cuBLAS heuristic (M=N=256, K=16384)\n"
                 "streams are WORSE than sequential on WDDM at every s — stream concurrency does not materialize")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_splitk.png"))
    plt.close(fig)


def fig4():
    doc = load("e4.json")
    recs = doc["experiments"]["E4"]["records"]
    b = [r["batch"] for r in recs]
    bmm = [r["bmm_tflops"] for r in recs]
    flat = [r["flat_tflops"] for r in recs]
    tiles = [r["bmm_tiles"] for r in recs]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(b, bmm, "o-", color="tab:blue", label="torch.bmm (batched kernel)")
    ax.plot(b, flat, "s-", color="tab:orange", label="single flat GEMM (B*256 x 256)")
    ax.axhline(FP32_PEAK, color="gray", ls="--", lw=1)
    ax.text(1, FP32_PEAK * 1.02, "FP32 peak", fontsize=7, color="gray")
    for x, t in zip(b, tiles):
        ax.annotate(f"{t}t", (x, 0.35), fontsize=6, color="gray", ha="center")
    ax.text(30, 5.6, "tile count (128x128 model)\ngrows with B", fontsize=6.5, color="gray")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size B (each 256x256 @ 256x256)")
    ax.set_ylabel("TFLOPS")
    ax.set_title("E4: batched GEMM vs equivalent flat GEMM\nflat wins for B>=4: batch scheduling overhead per group")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_batch.png"))
    plt.close(fig)


def fig5():
    doc = load("e5.json")
    recs = doc["experiments"]["E5"]["records"]
    n = [r["N"] for r in recs]
    fp32 = [r["fp32_tflops"] for r in recs]
    fp16 = [r["fp16_tflops"] for r in recs]
    int8_avail = "int8_tops" in recs[0]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(n, fp32, "o-", color="tab:green", label="FP32 (CUDA cores)")
    ax.plot(n, fp16, "s-", color="tab:orange", label="FP16 (tensor cores)")
    ax.axhline(FP32_PEAK, color="tab:green", ls="--", lw=1)
    ax.text(520, FP32_PEAK * 1.04, "FP32 peak 11.2 TFLOPS", fontsize=7, color="tab:green")
    ax.axhline(FP16_PEAK, color="tab:orange", ls="--", lw=1)
    ax.text(520, FP16_PEAK + 2, "FP16 TC peak 89.2 TFLOPS", fontsize=7, color="tab:orange")
    ax.axhline(INT8_PEAK, color="red", ls=":", lw=1.2)
    ax.text(520, INT8_PEAK + 3, "INT8 TC peak 178.4 TOPS", fontsize=7, color="red")
    if int8_avail:
        int8 = [r["int8_tops"] for r in recs]
        ax.plot(n, int8, "^-", color="red", label="INT8 (tensor cores, torch._int_mm)")
    else:
        ax.fill_between([512, 4096], INT8_PEAK * 0.9, INT8_PEAK * 1.02, color="red", alpha=0.06)
        ax.text(1500, 160, "INT8 not measurable: torch._int_mm unimplemented\n"
                           "in Windows torch 2.5.1 build (env.json); peak line only",
                fontsize=6.5, color="red")
    for x, v in zip(n, fp16):
        ax.annotate(f"{v / FP16_PEAK * 100:.0f}%", (x, v + 2.5), fontsize=6.5, color="tab:orange", ha="center")
    for x, v in zip(n, fp32):
        ax.annotate(f"{v / FP32_PEAK * 100:.0f}%", (x, v - 4.6), fontsize=6.5, color="tab:green", ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (square N x N x N)")
    ax.set_ylabel("TFLOPS / TOPS")
    ax.set_title("E5: dtype ladder on Turing sm_75 — peak utilization per dtype\n"
                 "labels = % of respective peak; FP16 TC gap 6.4-6.9x over FP32 at large N (two runs)")
    ax.legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_dtype.png"))
    plt.close(fig)


def fig6():
    doc = load("e6.json")
    recs = doc["experiments"]["E6"]["records"]
    n = [r["N"] for r in recs]
    fl = [r["flush_tflops"] for r in recs]
    nf = [r["noflush_tflops"] for r in recs]
    ws = [r["working_set_mb"] for r in recs]
    sp = [r["speedup_noflush_vs_flush"] for r in recs]

    x = np.arange(len(n))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - w / 2, fl, w, color="tab:blue", label="L2 flushed (64MB write before each iter)")
    ax.bar(x + w / 2, nf, w, color="tab:orange", label="no flush (cache-resident)")
    for i, (xi, s_, w_) in enumerate(zip(x, sp, ws)):
        ratio = w_ / L2_MB
        ax.text(xi, max(fl[i], nf[i]) + 0.12,
                f"x{s_:.2f}\nws={w_:.0f}MB ({ratio:.1f}x L2)", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={v}" for v in n])
    ax.set_ylabel("TFLOPS")
    ax.set_title("E6: L2 residency effect (per-iteration event timing, 5 reps x 3 passes)\n"
                 "only N=512 (working set 3MB < 4MB L2) shows a cache-residency gain")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig6_l2.png"))
    plt.close(fig)


def fig7():
    doc = load("e7.json")
    recs = doc["experiments"]["E7"]["records"]
    recs = sorted(recs, key=lambda r: -r["tflops"])
    names = [r["variant"] for r in recs]
    tf = [r["tflops"] for r in recs]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    colors = ["tab:green" if n in ("AB",) else "tab:blue" if "col" not in n else "tab:purple"
              for n in names]
    bars = ax.bar(names, tf, color=colors)
    for b, v in zip(bars, tf):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=7)
    spread = max(tf) / min(tf)
    ax.set_ylabel("TFLOPS")
    ax.set_ylim(9.0, 10.8)
    ax.set_title(f"E7: operand layout combos at 2048^3 FP32 — total spread only x{spread:.3f}\n"
                 "AB/AtB/ABt/AtBt = 4 products via cuBLAS op flags; colA/colB = col-major storage, same math as AB")
    ax.text(0.02, 0.03, "B-side transposed ops (ABt, A_colB) are the slowest (~3%):\n"
                        "ldmatrix + swizzled shared loads absorb layout differences on sm_75 cuBLAS",
            transform=ax.transAxes, fontsize=6.5, color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig7_layout.png"))
    plt.close(fig)


def main():
    fig1a()
    fig1b()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print("figures written to", FIGS)
    for f in sorted(os.listdir(FIGS)):
        print(" -", f)


if __name__ == "__main__":
    main()
