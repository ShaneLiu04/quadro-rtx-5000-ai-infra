"""AR002 gemm-lab F3: merge bench JSONs -> 6 explainable figures (figs/*.png).

Machine constants (repo README machine-boundary table, Quadro RTX 5000 / sm_75):
  FP32 peak 11.2 TFLOPS, FP16 TC peak 89.2 TFLOPS, HBM 448 GB/s,
  shared 64 KB/SM, 1024 threads/SM, roofline ridge = 11.2e12 / 448e9 = 25 FLOP/B.

Arithmetic-intensity (AI) derivations (FLOP per DRAM byte, N-independent):
  K0        : every output reads 2N floats -> AI = 2N^3 / (N^2 * 2N * 4B) = 0.25
  K1 tile T : loads per output = 2N/T      -> AI = T/4
  K2 (BN)   : loads per output = 2N/BN     -> AI = BN/4   (independent of TM!)
fig4 and fig6 use N=1024 (K0's largest measured size) so the full series
including K0 shares one frame -- K0 sitting ABOVE the bandwidth roof is the
L2-cache teaching point of fig6.
Figure language is English (font-safe); Chinese analysis lives in results.md.
Run with the GLOBAL python (matplotlib).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PEAK_FP32 = 11.2
PEAK_FP16 = 89.2
BW_GBPS = 448.0
RIDGE = PEAK_FP32 * 1e12 / (BW_GBPS * 1e9)  # FLOP/B

COLORS = {
    "K0_naive": "#7f7f7f",
    "K1_tiled_T8": "#9ecae1",
    "K1_tiled_T16": "#4292c6",
    "K1_tiled_T32": "#08519c",
    "K2a_regblock_32x32_2x2": "#fdae6b",
    "K2b_regblock_64x64_4x4": "#e6550d",
    "cublas_fp32": "#31a354",
    "cublas_fp16": "#006d2c",
}
LABELS = {
    "K0_naive": "K0 naive",
    "K1_tiled_T8": "K1 tiled T=8",
    "K1_tiled_T16": "K1 tiled T=16",
    "K1_tiled_T32": "K1 tiled T=32",
    "K2a_regblock_32x32_2x2": "K2a regblock 32x32 (2x2/thread)",
    "K2b_regblock_64x64_4x4": "K2b regblock 64x64 (4x4/thread)",
    "cublas_fp32": "cuBLAS FP32",
    "cublas_fp16": "cuBLAS FP16 (tensor cores)",
}


def load():
    with open(os.path.join(HERE, "results", "bench_numba.json")) as f:
        nb = json.load(f)
    with open(os.path.join(HERE, "results", "bench_torch.json")) as f:
        tb = json.load(f)
    recs = nb["series"] + tb["series"]
    return {(r["kernel"], r["N"]): r for r in recs}


def get(data, kernel, N):
    return data[(kernel, N)]


def fig1_scaling(data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k in ["K0_naive", "K1_tiled_T16", "K2a_regblock_32x32_2x2", "K2b_regblock_64x64_4x4", "cublas_fp32"]:
        pts = sorted([(N, r["tflops"]) for (kk, N), r in data.items() if kk == k])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COLORS[k], label=LABELS[k])
    ax.axhline(PEAK_FP32, color="k", ls="--", lw=1, alpha=0.6)
    ax.text(260, PEAK_FP32 * 1.03, "FP32 peak 11.2 TFLOPS", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("matrix size N (N x N x N FP32 GEMM)")
    ax.set_ylabel("TFLOPS (median)")
    ax.set_title("fig1  SGEMM optimization progression vs cuBLAS (Quadro RTX 5000, sm_75)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "fig1_scaling.png"), dpi=300)


def fig2_tiles(data):
    N = 2048
    ks = ["K1_tiled_T8", "K1_tiled_T16", "K1_tiled_T32"]
    tf = [get(data, k, N)["tflops"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(["T=8", "T=16", "T=32"], tf, color=[COLORS[k] for k in ks], width=0.55)
    # occupancy annotations: threads/block, shared bytes/block, resident blocks/SM
    meta = [
        "64 thr/block\nshared 0.5 KB\n16 blocks/SM (1024 thr)",
        "256 thr/block\nshared 2 KB\n4 blocks/SM (1024 thr)",
        "1024 thr/block\nshared 8 KB\n1 block/SM (1024 thr)",
    ]
    for b, m, v in zip(bars, meta, tf):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f} TFLOPS\n{m}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(tf) * 1.45)
    ax.set_ylabel("TFLOPS (median)")
    ax.set_title(f"fig2  shared-tile size sweep, K1 tiled @ N={N}\n(smaller tiles -> more resident blocks; all fill 1024 thr/SM)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "fig2_tiles.png"), dpi=300)


def fig3_regblock(data):
    N = 2048
    ks = ["K1_tiled_T32", "K2a_regblock_32x32_2x2", "K2b_regblock_64x64_4x4"]
    tf = [get(data, k, N)["tflops"] for k in ks]
    labels = ["K1 T=32\n(1 output/thread)", "K2a 32x32\n(2x2/thread)", "K2b 64x64\n(4x4/thread)"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(labels, tf, color=[COLORS[k] for k in ks], width=0.55)
    fma = ["FMA per shared-load pair: 1\nacc registers: 1",
           "FMA per shared-load pair: 2\nacc registers: 4",
           "FMA per shared-load pair: 4\nacc registers: 16"]
    for b, m, v in zip(bars, fma, tf):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f} TFLOPS\n{m}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(tf) * 1.4)
    ax.set_ylabel("TFLOPS (median)")
    ax.set_title(f"fig3  register-blocking sweep @ N={N}\n(same global-memory traffic as K1 T=32 at AI=8; the win is on-chip reuse + ILP)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "fig3_regblock.png"), dpi=300)


def fig4_causality(data):
    N = 1024
    # global loads per output element: K0=2N; K1=2N/T; K2=2N/BN
    rows = [
        ("K0_naive", 2 * N, get(data, "K0_naive", N)["tflops"]),
        ("K1_tiled_T8", 2 * N / 8, get(data, "K1_tiled_T8", N)["tflops"]),
        ("K1_tiled_T16", 2 * N / 16, get(data, "K1_tiled_T16", N)["tflops"]),
        ("K1_tiled_T32", 2 * N / 32, get(data, "K1_tiled_T32", N)["tflops"]),
        ("K2a_regblock_32x32_2x2", 2 * N / 32, get(data, "K2a_regblock_32x32_2x2", N)["tflops"]),
        ("K2b_regblock_64x64_4x4", 2 * N / 64, get(data, "K2b_regblock_64x64_4x4", N)["tflops"]),
        ("cublas_fp32", 2 * N / 128, get(data, "cublas_fp32", N)["tflops"]),  # est. ~128-wide tiling
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, x, y in rows:
        is_est = name.startswith("cublas")
        ax.scatter(x, y, s=70, color=COLORS[name], zorder=3,
                   marker="s" if is_est else "o")
        ax.annotate(LABELS[name] + (" (est. tile)" if is_est else ""), (x, y),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("global loads per output element (log2)  --  lower = less DRAM traffic")
    ax.set_ylabel("TFLOPS (median, log2)")
    ax.set_title(f"fig4  mechanism causality @ N={N}\n fewer global loads per output -> higher bandwidth headroom -> higher TFLOPS")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "fig4_causality.png"), dpi=300)


def fig5_precision(data):
    Ns = [256, 512, 1024, 2048, 4096]
    fp32 = [get(data, "cublas_fp32", N)["tflops"] for N in Ns]
    fp16 = [get(data, "cublas_fp16", N)["tflops"] for N in Ns]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(Ns, fp32, "o-", color=COLORS["cublas_fp32"], label="cuBLAS FP32 (CUDA cores)")
    ax.plot(Ns, fp16, "o-", color=COLORS["cublas_fp16"], label="cuBLAS FP16 (tensor cores)")
    ax.axhline(PEAK_FP32, color=COLORS["cublas_fp32"], ls="--", lw=1, alpha=0.6)
    ax.axhline(PEAK_FP16, color=COLORS["cublas_fp16"], ls="--", lw=1, alpha=0.6)
    ax.text(260, PEAK_FP32 * 1.05, "FP32 peak 11.2", fontsize=9, color=COLORS["cublas_fp32"])
    ax.text(260, PEAK_FP16 * 1.05, "FP16 TC peak 89.2", fontsize=9, color=COLORS["cublas_fp16"])
    ax.annotate(f"{fp32[-1]/PEAK_FP32*100:.0f}% of peak", (Ns[-1], fp32[-1]),
                textcoords="offset points", xytext=(-90, 10), fontsize=8.5)
    ax.annotate(f"{fp16[-1]/PEAK_FP16*100:.0f}% of peak", (Ns[-1], fp16[-1]),
                textcoords="offset points", xytext=(-90, -15), fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("matrix size N")
    ax.set_ylabel("TFLOPS (median)")
    ax.set_title("fig5  same cuBLAS, FP32 vs FP16: tensor cores are a different hardware path\n(small N is launch/latency-bound: peak % is only meaningful at large N)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "fig5_precision.png"), dpi=300)


def fig6_roofline(data):
    N = 1024
    # AI (FLOP/DRAM byte): K0=0.25; K1=T/4; K2=BN/4
    rows = [
        ("K0_naive", 0.25, get(data, "K0_naive", N)["tflops"]),
        ("K1_tiled_T8", 8 / 4, get(data, "K1_tiled_T8", N)["tflops"]),
        ("K1_tiled_T16", 16 / 4, get(data, "K1_tiled_T16", N)["tflops"]),
        ("K1_tiled_T32", 32 / 4, get(data, "K1_tiled_T32", N)["tflops"]),
        ("K2a_regblock_32x32_2x2", 32 / 4, get(data, "K2a_regblock_32x32_2x2", N)["tflops"]),
        ("K2b_regblock_64x64_4x4", 64 / 4, get(data, "K2b_regblock_64x64_4x4", N)["tflops"]),
        ("cublas_fp32", 32.0, get(data, "cublas_fp32", N)["tflops"]),  # est. AI ~32 (deep tiling)
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [0.2, 2, 128]
    ax.plot(xs, [x * BW_GBPS / 1000 for x in xs], "k-", lw=1.2, alpha=0.7,
            label="bandwidth roof: 448 GB/s x AI")
    ax.axhline(PEAK_FP32, color="k", ls="--", lw=1.2, alpha=0.7, label="FP32 compute roof: 11.2 TFLOPS")
    ax.axvline(RIDGE, color="r", ls=":", lw=1.2)
    ax.text(RIDGE * 1.1, 0.015, f"ridge point = {RIDGE:.0f} FLOP/B", color="r", fontsize=9)
    for name, ai, y in rows:
        is_est = name.startswith("cublas")
        ax.scatter(ai, y, s=80, color=COLORS[name], zorder=3, marker="s" if is_est else "o")
        ax.annotate(LABELS[name] + (" (est. AI)" if is_est else ""), (ai, y),
                    textcoords="offset points", xytext=(9, 4), fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("arithmetic intensity (FLOP per DRAM byte, log2)")
    ax.set_ylabel("attained TFLOPS (median, log2)")
    ax.set_title(f"fig6  roofline positioning @ N={N}\n every hand-written kernel sits left of the ridge: still (partly) bandwidth-limited\n K1 T=32 and K2a share AI=8 yet differ 0.95 vs 1.31 TFLOPS: on-chip effects, not DRAM")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "fig6_roofline.png"), dpi=300)


def main():
    data = load()
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig1_scaling(data)
    fig2_tiles(data)
    fig3_regblock(data)
    fig4_causality(data)
    fig5_precision(data)
    fig6_roofline(data)
    # console summary used by results.md
    print("=== achieved-vs-peak summary (medians) ===")
    for k, N in [("K2b_regblock_64x64_4x4", 2048), ("cublas_fp32", 2048), ("cublas_fp32", 4096),
                 ("cublas_fp16", 2048), ("cublas_fp16", 4096)]:
        r = get(data, k, N)
        peak = PEAK_FP16 if "fp16" in k else PEAK_FP32
        print(f"{k:<26} N={N:<5} {r['tflops']:>8.3f} TFLOPS  = {r['tflops']/peak*100:5.1f}% of {'FP16' if 'fp16' in k else 'FP32'} peak")
    print("figs written:", sorted(os.listdir(os.path.join(HERE, "figs"))))


if __name__ == "__main__":
    main()
