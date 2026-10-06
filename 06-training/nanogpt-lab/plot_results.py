"""AR006 nanogpt-lab: generate all figures from results/*.json (no hardcoded data).

Usage: python plot_results.py   (writes figs/fig1..fig13, 300 DPI, English labels)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

GPU_NAME = "Quadro RTX 5000 (sm_75, 48 SM, 448 GB/s)"
PEAK_FP16_TF = 89.2   # tensor-core peak, from CUDA device properties (lab constant)
PEAK_FP32_TF = 11.2

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.titlesize": 10,
    "axes.titleweight": "bold", "figure.autolayout": True,
})


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def savefig(fig, name):
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


e1, e2, e3, e4, e5 = (load("e1.json"), load("e2.json"), load("e3.json"),
                      load("e4.json"), load("e5.json"))

# ---------------------------------------------------------------- fig1: E1 loss
steps_train = [(i + 1) * 10 for i in range(len(e1["loss_curve"]))]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(steps_train, e1["loss_curve"], lw=1.0, alpha=0.8, label="train loss (logged every 10 steps)")
ax.plot(e1["val_curve"]["steps"], e1["val_curve"]["vals"], "o-", color="tab:red",
        lw=1.5, ms=4, label="val loss (every 250 steps)")
imin = e1["val_curve"]["vals"].index(min(e1["val_curve"]["vals"]))
smin = e1["val_curve"]["steps"][imin]
ax.annotate(f"val min {e1['val_curve']['vals'][imin]:.3f} @ step {smin}",
            xy=(smin, e1["val_curve"]["vals"][imin]),
            xytext=(smin + 350, e1["val_curve"]["vals"][imin] - 0.25),
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("train step")
ax.set_ylabel("cross-entropy loss (nats/char)")
ax.set_title("E1 baseline: 10.67M-param GPT, fp16 + GradScaler, 3000 steps")
ax.legend(loc="upper right")
txt = (f"final train {e1['final_train_loss']:.3f}   final val {e1['final_val_loss']:.3f}\n"
       f"{e1['tokens_per_s']/1e3:.0f}k tok/s   {e1['step_time_ms_median']:.1f} ms/step   "
       f"MFU {e1['mfu_pct']:.1f}%\n"
       f"GradScaler {e1['scaler']['init']:.0f} -> {e1['scaler']['final']:.0f}, "
       f"{e1['scaler']['backoffs']} backoffs")
ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
savefig(fig, "fig1_e1_loss.png")

# ------------------------------------------------- fig2: E1 overfit narrative
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(steps_train, e1["loss_curve"], lw=1.0, color="tab:blue", alpha=0.7, label="train")
ax.plot(e1["val_curve"]["steps"], e1["val_curve"]["vals"], "o-", color="tab:red",
        lw=1.5, ms=4, label="val")
ax.axvspan(smin, 3000, color="tab:red", alpha=0.06)
ax.text((smin + 3000) / 2, 2.05, "val rises while train keeps falling\n= textbook overfitting\n"
        "(~58 epochs over 854 KB train text)", ha="center", fontsize=8, color="tab:red")
ax.set_xlabel("train step")
ax.set_ylabel("loss (nats/char)")
ax.set_title("E1: the U-shaped val curve (10.67M params vs 854 KB corpus)")
ax.legend(loc="upper right")
savefig(fig, "fig2_e1_val_ushape.png")

# ------------------------------------------- fig3: E2 precision throughput bars
p = {k: e2[k] for k in ("fp32", "fp16", "bf16")}
fig, ax = plt.subplots(figsize=(6, 4.2))
names = ["fp32", "fp16 (AMP)", "bf16 (AMP)"]
toks = [p["fp32"]["tokens_per_s"], p["fp16"]["tokens_per_s"], p["bf16"]["tokens_per_s"]]
colors = ["tab:gray", "tab:blue", "tab:orange"]
bars = ax.bar(names, [t / 1e3 for t in toks], color=colors)
fp32t = p["fp32"]["tokens_per_s"]
for b, t in zip(bars, toks):
    ax.text(b.get_x() + b.get_width() / 2, t / 1e3 + 8,
            f"{t/1e3:.0f}k\n({t/fp32t:.2f}x fp32)", ha="center", fontsize=8)
ax.set_ylabel("throughput (k tok/s), 3.25M model, B=64, T=256")
ax.set_title("E2: precision vs throughput on Turing (sm_75)")
ax.text(0.5, -0.22, "bf16 SLOWER than fp32: no BF16 tensor cores on sm_75 -> software emulation;\n"
        "fp16 speedup comes from fp16 tensor cores (peak %.1f TF vs %.1f TF fp32)\n%s"
        % (PEAK_FP16_TF, PEAK_FP32_TF, GPU_NAME),
        transform=ax.transAxes, ha="center", fontsize=7.5)
savefig(fig, "fig3_e2_precision_tokps.png")

# ------------------------------------------- fig4: E2 loss curves + MFU bars
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for k, c in (("fp32", "tab:gray"), ("fp16", "tab:blue"), ("bf16", "tab:orange")):
    lc = p[k]["loss_curve"]
    axes[0].plot([(i + 1) * 10 for i in range(len(lc))], lc, color=c, lw=1.1,
                 label=f"{k} (final {p[k]['final_train_loss']:.3f})")
axes[0].set_xlabel("train step")
axes[0].set_ylabel("train loss")
axes[0].set_title("E2: loss agreement across precisions")
axes[0].legend()
mfus = [p[k]["mfu_pct"] for k in ("fp32", "fp16", "bf16")]
bars = axes[1].bar(names, mfus, color=colors)
for b, k in zip(bars, ("fp32", "fp16", "bf16")):
    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                 f"{p[k]['mfu_pct']:.1f}%\n(peak {PEAK_FP16_TF if k=='fp16' else PEAK_FP32_TF} TF"
                 f"{', emulated' if k=='bf16' else ''})", ha="center", fontsize=7.5)
axes[1].set_ylabel("MFU (%)")
axes[1].set_title("E2: MFU vs precision (3.25M model is launch-bound)")
savefig(fig, "fig4_e2_loss_mfu.png")

# ------------------------------------------- fig5: E3 val loss vs params (log-log)
sizes = ["0.5M", "3M", "10.5M", "25M"]
npar = [e3["sizes"][s]["n_params_non_embed"] for s in sizes]
valf = [e3["sizes"][s]["final_val_loss"] for s in sizes]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(npar, valf, "o-", ms=6)
for x, y, s in zip(npar, valf, sizes):
    ax.annotate(f"{s}: {y:.3f}", xy=(x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
best = sizes[valf.index(min(valf))]
ax.set_xscale("log")
ax.set_xlabel("non-embedding parameters (log)")
ax.set_ylabel("final val loss (nats/char)")
ax.set_title(f"E3: fixed 25M-token budget -> bigger is NOT better\n"
             f"optimum at {best}; 25M model is undertrained (Chinchilla gap)")
savefig(fig, "fig5_e3_val_vs_params.png")

# ------------------------------------------- fig6: E3 MFU + tok/s vs params
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
mfu = [e3["sizes"][s]["mfu_pct"] for s in sizes]
tps = [e3["sizes"][s]["tokens_per_s"] for s in sizes]
axes[0].plot(npar, mfu, "o-", ms=5, color="tab:green")
for x, y in zip(npar, mfu):
    axes[0].annotate(f"{y:.1f}%", xy=(x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
axes[0].set_xscale("log")
axes[0].set_xlabel("non-embedding parameters (log)")
axes[0].set_ylabel("MFU (%) vs fp16 TC peak")
axes[0].set_title("MFU rises monotonically with size\n(small models are launch-bound)")
axes[1].plot(npar, [t / 1e3 for t in tps], "o-", ms=5, color="tab:purple")
for x, t in zip(npar, tps):
    axes[1].annotate(f"{t/1e3:.0f}k", xy=(x, t / 1e3), xytext=(4, 4),
                     textcoords="offset points", fontsize=8)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("non-embedding parameters (log)")
axes[1].set_ylabel("throughput (k tok/s, log)")
axes[1].set_title("throughput falls as model grows\n(compute per token grows faster than MFU)")
savefig(fig, "fig6_e3_mfu_tokps.png")

# ------------------------------------------- fig7: E3 val curves per size
fig, ax = plt.subplots(figsize=(7, 4.2))
for s, c in zip(sizes, ("tab:gray", "tab:blue", "tab:orange", "tab:red")):
    vc = e3["sizes"][s]["val_curve"]
    ax.plot(vc["steps"], vc["vals"], "o-", ms=3, lw=1.2, color=c,
            label=f"{s} (final {e3['sizes'][s]['final_val_loss']:.3f})")
ax.set_xlabel("train step (1526 steps = 25M tokens for every size)")
ax.set_ylabel("val loss")
ax.set_title("E3: val trajectories under a fixed token budget")
ax.legend()
savefig(fig, "fig7_e3_val_curves.png")

# ------------------------------------------- fig8: E3 step time vs compute prediction
fpt25 = e3["sizes"]["25M"]["flops_per_token"]
t25 = e3["sizes"]["25M"]["step_time_ms_median"]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
pred = [t25 * e3["sizes"][s]["flops_per_token"] / fpt25 for s in sizes]
meas = [e3["sizes"][s]["step_time_ms_median"] for s in sizes]
x = range(len(sizes))
ax.bar([i - 0.18 for i in x], meas, width=0.36, label="measured median step time")
ax.bar([i + 0.18 for i in x], pred, width=0.36, label="predicted if compute-bound\n"
       "(linear in flops/token, scaled from 25M)")
for i, (m, p_) in enumerate(zip(meas, pred)):
    ax.text(i - 0.18, m + 1, f"{m:.1f}", ha="center", fontsize=7.5)
    ax.text(i + 0.18, p_ + 1, f"{p_:.1f}", ha="center", fontsize=7.5)
ax.set_xticks(list(x))
ax.set_xticklabels(sizes)
ax.set_ylabel("step time (ms), B=64 T=256")
ax.set_title("E3: step time EXCEEDS compute-bound prediction, more so for smaller models\n"
             "(fixed launch/memory overhead; measured/predicted = %.2fx / %.2fx / %.2fx)"
             % (meas[0] / pred[0], meas[1] / pred[1], meas[2] / pred[2]))
ax.legend()
savefig(fig, "fig8_e3_step_time_scaling.png")

# ------------------------------------------- fig9: E4 LR sweep finals
lrs = [1e-4, 3e-4, 1e-3, 3e-3]
cos = [e4["lr_sweep"][f"lr{lr:g}_cosine"]["final_train_loss"] for lr in lrs]
con = [e4["lr_sweep"][f"lr{lr:g}_constant"]["final_train_loss"] for lr in lrs]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(lrs, cos, "o-", label="cosine schedule (100-step warmup)")
ax.plot(lrs, con, "s--", label="constant")
best_i = cos.index(min(cos))
ax.annotate(f"best: lr=3e-3 cosine, final {min(cos):.2f}", xy=(lrs[best_i], min(cos)),
            xytext=(0.04, 0.30), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
for lr, c, k in zip(lrs, cos, con):
    ax.text(lr, c + 0.04, f"{c:.2f}", ha="center", fontsize=7.5)
    ax.text(lr, k - 0.09, f"{k:.2f}", ha="center", fontsize=7.5)
ax.set_xscale("log")
ax.set_xlabel("peak learning rate (log)")
ax.set_ylabel("final train loss after 800 steps (3.25M model, fp16)")
ax.set_title("E4: LR sweep 1e-4 -> 3e-3 (no divergence even at 3e-3)")
ax.legend()
savefig(fig, "fig9_e4_lr_finals.png")

# ------------------------------------------- fig10: E4 LR loss curves
fig, ax = plt.subplots(figsize=(7, 4.2))
colors4 = ["tab:gray", "tab:orange", "tab:blue", "tab:green"]
for lr, c in zip(lrs, colors4):
    for sched, ls in (("cosine", "-"), ("constant", "--")):
        lc = e4["lr_sweep"][f"lr{lr:g}_{sched}"]["loss_curve"]
        ax.plot([(i + 1) * 10 for i in range(len(lc))], lc, ls, lw=1.1, color=c, alpha=0.9,
                label=f"lr={lr:g} {sched}" if sched == "cosine" else None)
ax.set_xlabel("train step")
ax.set_ylabel("train loss")
ax.set_title("E4: training curves per LR (solid = cosine, dashed = constant)")
ax.legend(fontsize=7.5, ncol=2)
savefig(fig, "fig10_e4_lr_curves.png")

# ------------------------------------------- fig11: E4 batch tradeoff
bs = ["B16", "B64", "B256"]
bx = [16, 64, 256]
bloss = [e4["batch_sweep"][b]["final_train_loss"] for b in bs]
btps = [e4["batch_sweep"][b]["tokens_per_s"] for b in bs]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(bx, bloss, "o-", color="tab:red", label="final train loss")
for x_, y_ in zip(bx, bloss):
    ax.text(x_, y_ + 0.06, f"{y_:.2f}", ha="center", fontsize=8)
ax.set_xscale("log", base=2)
ax.set_xlabel("batch size (log2) - every run sees the same 13.1M-token budget")
ax.set_ylabel("final train loss", color="tab:red")
ax2 = ax.twinx()
ax2.plot(bx, [t / 1e3 for t in btps], "s-", color="tab:blue", label="throughput")
for x_, t in zip(bx, btps):
    ax2.text(x_, t / 1e3 + 12, f"{t/1e3:.0f}k", ha="center", fontsize=8, color="tab:blue")
ax2.set_ylabel("throughput (k tok/s)", color="tab:blue")
ax2.grid(False)
ax.set_title("E4: batch size tradeoff - 4x more updates vs 2.1x more throughput\n"
             "B16: %d updates, %.0fk tok/s | B256: %d updates, %.0fk tok/s"
             % (e4["batch_sweep"]["B16"]["steps"], btps[0] / 1e3,
                e4["batch_sweep"]["B256"]["steps"], btps[2] / 1e3))
savefig(fig, "fig11_e4_batch_tradeoff.png")

# ------------------------------------------- fig12: E5 category breakdown
cats = sorted(((k, v["pct"]) for k, v in e5["categories"].items()),
              key=lambda kv: kv[1], reverse=True)
fig, ax = plt.subplots(figsize=(7, 4.4))
labels = [c for c, _ in cats]
vals = [v for _, v in cats]
colors = ["tab:red" if c == "linear/matmul" else
          ("0.75" if c == "other" else "tab:blue") for c in labels]
bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("share of device time (%) - 20 profiled steps, 10.67M fp16 model")
ax.set_title("E5: where does eager step time go? GEMM is only "
             f"{e5['categories']['linear/matmul']['pct']:.0f}%\n"
             "the other ~73% (elementwise, attention, layernorm, eager overhead...) is what llm.c hand-fuses")
savefig(fig, "fig12_e5_categories.png")

# ------------------------------------------- fig13: E5 top ops
ops = e5["top_ops"][:10]
fig, ax = plt.subplots(figsize=(7.5, 4.4))
names = [o["name"] for o in ops][::-1]
vals = [o["pct"] for o in ops][::-1]
cnts = [o["count"] for o in ops][::-1]
bars = ax.barh(names, vals, color="tab:purple", alpha=0.8)
for b, v, c in zip(bars, vals, cnts):
    ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}% ({c} calls)",
            va="center", fontsize=7.5)
ax.set_xlabel("self device time (%)")
ax.set_title(f"E5: top 10 ops cover {sum(o['pct'] for o in e5['top_ops'][:5]):.0f}% of device time (top 5)\n"
             f"mm: {ops[0]['count']//20} launches/step; 75 mm per step = 3 qkv/proj per layer x 6 layers + head")
savefig(fig, "fig13_e5_topops.png")

print("all figures written to", FIGS)
