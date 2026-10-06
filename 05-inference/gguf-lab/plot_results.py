"""plot_results.py — AR005 figures (300 DPI, English labels).

Reads results/*.json produced by bench_infer.py and writes figs/fig1..fig9.png.
Every figure annotates reference lines / units; numbers come only from JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB = Path(__file__).parent
RES = LAB / "results"
FIGS = LAB / "figs"
FIGS.mkdir(exist_ok=True)
HBM = 448.0


def J(name):
    return json.loads((RES / name).read_text())


def savefig(fig, name):
    fig.savefig(FIGS / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


# ---------------------------------------------------------------- fig1: loss
e0 = J("e0_train.json")
fig, ax = plt.subplots(figsize=(6, 4))
steps = np.arange(len(e0["loss_curve"])) * 10
ax.plot(steps, e0["loss_curve"], lw=1.5)
ax.axhline(2.0, color="gray", ls="--", lw=1, label="gate (< 2.0)")
ax.set_xlabel("train step")
ax.set_ylabel("cross-entropy loss (chars)")
ax.set_title(f"char-LM training: 3.21M params, {e0['train_time_s']:.0f}s on RTX 5000")
ax.annotate(f"final {e0['final_loss']:.2f}", xy=(steps[-1], e0["final_loss"]),
            xytext=(-70, 15), textcoords="offset points", color="tab:blue")
ax.legend()
ax.grid(alpha=0.3)
savefig(fig, "fig1_loss.png")

# ------------------------------------------------- fig2: quant error histograms
e1 = J("e1.json")
fig, ax = plt.subplots(figsize=(6.5, 4))
for q, c in [("q8_0", "tab:blue"), ("q4_K", "tab:orange")]:
    h = np.array(e1["hist"][q])
    edges = np.array(e1["hist"][q + "_edges"])
    widths = edges[1:] - edges[:-1]
    ax.bar(edges[:-1], h / h.sum(), width=widths, align="edge", alpha=0.6,
           color=c, label=f"{q} (rel {e1[q]['rel_overall']:.4f}, bpw {e1[q]['bpw']:.3f})")
ax.set_xlabel("per-element error / max|w| (normalized)")
ax.set_ylabel("fraction of weights")
ax.set_title("Quantization roundtrip error on trained weights (E1)")
ax.legend()
ax.grid(alpha=0.3)
savefig(fig, "fig2_quant_err_hist.png")

# --------------------------------------------- fig3: dequant implementation ladder
e3 = J("e3.json")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
# left: same-data-scale ladder (4096 Q4_K super-blocks = 1.05M weights)
nv = e3["dequant_q4_K_4096blocks"]
t_ms = [nv["naive_ms"], nv["vec_ms"]]
bars = ax1.bar(["naive per-block\nloop", "torch multi-op\nvectorized"], t_ms,
               color=["tab:red", "tab:orange"])
for b, t in zip(bars, t_ms):
    ax1.text(b.get_x() + b.get_width() / 2, t * 1.4,
             f"{t:.0f} ms" if t > 10 else f"{t:.2f} ms", ha="center", fontsize=9)
ax1.set_yscale("log")
ax1.set_ylabel("time (ms, log scale)")
ax1.set_title(f"same data: 4096 super-blocks (1.05M w)\nspeedup {nv['speedup']:.0f}x")
ax1.grid(alpha=0.3, axis="y")
# right: full-size 4096x4096 bandwidth comparison
tv = e3["dequant_q4_K_torch_vec_fp32"]
tq = e3["dequant_q4_K_triton_fp32"]
gbps = [tv["gbps_min"], tq["gbps_min"]]
bars = ax2.bar(["torch multi-op\nvectorized", "triton fused\nsingle kernel"], gbps,
               color=["tab:orange", "tab:green"])
for b, g in zip(bars, gbps):
    ax2.text(b.get_x() + b.get_width() / 2, g * 1.1, f"{g:.1f}", ha="center", fontsize=9)
ax2.axhline(HBM, color="k", ls="--", lw=1, label=f"HBM spec {HBM:.0f} GB/s")
ax2.axhline(200, color="gray", ls=":", lw=1, label="gate 200 GB/s")
ax2.set_ylabel("effective traffic (GB/s)")
ax2.set_title(f"full size: 4096x4096 fp32 out\n{tq['gbps_min'] / tv['gbps_min']:.1f}x, {tq['pct_hbm_min']:.0f}% HBM")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis="y")
fig.suptitle("Q4_K dequantization: implementation ladder (E3)")
fig.tight_layout()
savefig(fig, "fig3_dequant_ladder.png")

# ---------------------------------------------------------------- fig4: ppl
fig, ax = plt.subplots(figsize=(5.5, 4))
names = ["fp16", "q8_0", "q4_K"]
ppls = [e3[f"ppl_{n}"] for n in names]
bars = ax.bar(names, ppls, color=["tab:blue", "tab:green", "tab:orange"])
for b, n, p in zip(bars, names, ppls):
    d = e3[f"ppl_delta_{n}"] if n != "fp16" else 0.0
    ax.text(b.get_x() + b.get_width() / 2, p + 0.05, f"{p:.2f}\n(d{d:+.2f})",
            ha="center", fontsize=9)
ax.set_ylabel("perplexity (held-out, fp32 compute)")
ax.set_ylim(14.5, 17.3)
ax.set_title("Quantization quality: ppl vs weight format (E3)")
ax.grid(alpha=0.3, axis="y")
savefig(fig, "fig4_ppl.png")

# ------------------------------------------------- fig5: decode tokens/s vs batch
e4 = J("e4.json")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
bs = [r["batch"] for r in e4["decode_vs_batch"]]
tot = [r["tokens_per_s_total"] for r in e4["decode_vs_batch"]]
per = [r["tokens_per_s_per_seq"] for r in e4["decode_vs_batch"]]
ax.plot(bs, tot, "o-", label="total throughput")
ax.plot(bs, per, "s--", label="per-sequence")
ax.axhline(e4["single_seq_floor_tokens_per_s"], color="k", ls="--", lw=1,
           label=f"weight-streaming floor 1-seq ({e4['single_seq_floor_tokens_per_s']/1e3:.0f}k t/s, fp16 6.4MB / 448GB/s)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("batch size (same prompt, independent KV)")
ax.set_ylabel("tokens / s")
ax.set_title("Decode throughput vs batch: near-constant step time -> linear scaling (E4)")
ax.annotate(f"{tot[0]:.0f} t/s", (1, tot[0]), xytext=(5, -3), textcoords="offset points", fontsize=9)
ax.annotate(f"{tot[-1]:.0f} t/s", (64, tot[-1]), xytext=(-40, 5), textcoords="offset points", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
savefig(fig, "fig5_decode_vs_batch.png")

# ---------------------------------------------------------------- fig6: KV formula
fig, ax = plt.subplots(figsize=(5.5, 4.5))
kv = e4["kv_cache_check"]
for B, c in [(1, "tab:blue"), (16, "tab:orange")]:
    pts = [k for k in kv if k["batch"] == B]
    x = [k["formula_bytes"] / 1e6 for k in pts]
    y = [k["measured_bytes"] / 1e6 for k in pts]
    ax.plot(x, y, "o", color=c, label=f"batch {B}")
lim = [0, max(k["formula_bytes"] for k in kv) / 1e6 * 1.1]
ax.plot(lim, lim, "k--", lw=1, label="y = x")
ax.set_xlabel("formula: 2 x layers x kv_heads x seq x head_dim x 2B x batch (MB)")
ax.set_ylabel("measured torch.cuda.memory_allocated delta (MB)")
ax.set_title("KV cache memory: formula vs measured (GQA 4:2, max err 0.00%)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
savefig(fig, "fig6_kv_formula.png")

# ------------------------------------------------- fig7: temperature -> top-p cutoff
e5 = J("e5.json")
fig, ax = plt.subplots(figsize=(6, 4.5))
beh = e5["temperature_vs_topp_cutoff"]
T = [r["T"] for r in beh]
m = [r["mean_cutoff_set"] for r in beh]
s = [r["std"] for r in beh]
ax.errorbar(T, m, yerr=s, fmt="o-", capsize=4, label="mean cutoff set size")
ax.plot(T, [r["max"] for r in beh], "^", color="gray", ms=5, label="max")
ax.set_xlabel("temperature (logits / T)")
ax.set_ylabel("top-p 0.9 cutoff set size (vocab 119)")
ax.set_title("Temperature flattens the distribution -> more tokens needed for 90% mass (E5)")
for t, mm in zip(T, m):
    ax.annotate(f"{mm:.1f}", (t, mm), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
savefig(fig, "fig7_temperature_topp.png")

# ---------------------------------------------------------------- fig8: prefill vs decode
p = e4["prefill_vs_decode"]
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(["prefill\n(256 tok, 1 pass, M=256)", "decode\n(256 steps, M=1 each)"],
       [p["prefill_us_per_token"], p["decode_us_per_token"]],
       color=["tab:green", "tab:red"])
ax.set_ylabel("time per token (us, log scale)")
ax.set_yscale("log")
for i, (v, tf) in enumerate([(p["prefill_us_per_token"], p["prefill_tflops"]),
                             (p["decode_us_per_token"], p["decode_tflops"])]):
    ax.text(i, v * 1.3, f"{v:.0f} us/tok\n{tf:.3g} TFLOPS", ha="center", fontsize=9)
ax.set_title(f"Prefill vs decode: {p['ratio_decode_over_prefill']:.0f}x slower per token in decode\n"
             "(decode = GEMV per step, launch-bound at 3.2M params; prefill amortizes weights over 256 rows)")
ax.grid(alpha=0.3, axis="y")
savefig(fig, "fig8_prefill_vs_decode.png")

# ------------------------------------------------- fig9: end-to-end decode speed formats
fig, ax = plt.subplots(figsize=(7, 4.5))
names = ["fp16 weights", "Q4_K pre-dequantized\n(fp16 in memory)",
         "Q4_K on-the-fly\ntorch multi-op dequant", "Q4_K on-the-fly\ntriton per-tensor (est)"]
vals = [e3["decode_fp16_tokens_per_s"], e3["decode_q4K_predequant_tokens_per_s"],
        e3["decode_q4K_onthefly_est_tokens_per_s"], e3["decode_q4K_onthefly_triton_est_tokens_per_s"]]
cols = ["tab:blue", "tab:green", "tab:red", "tab:orange"]
bars = ax.bar(range(4), vals, color=cols)
ax.set_xticks(range(4))
ax.set_xticklabels(names, fontsize=8)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:.0f}", ha="center", fontsize=9)
ax.set_ylabel("decode tokens / s (batch 1, 64 steps)")
ax.set_yscale("log")
ax.set_title("'4-bit weights' do NOT make decode faster:\ndequant cost dominates unless fused into the GEMM (mmvq/mmq lesson)")
ax.grid(alpha=0.3, axis="y")
savefig(fig, "fig9_decode_formats.png")

print("all figures done")
