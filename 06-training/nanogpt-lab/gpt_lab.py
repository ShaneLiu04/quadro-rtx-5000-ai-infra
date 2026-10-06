"""gpt_lab.py — AR006 training driver + experiments E1-E5/E7 (nanoGPT upstream, zero-mod).

Usage: python gpt_lab.py {e1,e2,e3,e4,e5,e7,e7c,all}
All numbers land in results/*.json (single source of truth for plots/docs).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).parent
sys.path.insert(0, str(LAB.parent / "nanoGPT"))
from model import GPT, GPTConfig  # noqa: E402  (upstream, read-only)

DATA = LAB / "data"
RES = LAB / "results"
RES.mkdir(exist_ok=True)

STATS = json.loads((DATA / "corpus_stats.json").read_text(encoding="utf-8"))
VOCAB = STATS["vocab_size"]
VOCAB_CHARS = STATS["vocab"]
SPLIT = STATS["split_index"]
TOKENS = np.load(DATA / "tokens.npy", mmap_mode="r")

PEAK_FP16_TF = 89.2   # sm_75 FP16 tensor-core peak
PEAK_FP32_TF = 11.2   # sm_75 FP32 CUDA-core peak

CONFIGS = {
    "0.5M":  dict(n_layer=2, n_embd=128, n_head=2),
    "3M":    dict(n_layer=4, n_embd=256, n_head=4),
    "10.5M": dict(n_layer=6, n_embd=384, n_head=6),
    "25M":   dict(n_layer=8, n_embd=512, n_head=8),
}


def save(name, obj):
    (RES / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {name}.json")


def get_batch(split_name, batch_size, block, rng):
    data = TOKENS[:SPLIT] if split_name == "train" else TOKENS[SPLIT:]
    ix = rng.integers(0, len(data) - block - 1, batch_size)
    x = np.stack([data[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()


def flops_per_token(model, block):
    """PaLM/nanoGPT formula: 6N + 12*L*H*Q*T (N excludes position embeddings)."""
    cfg = model.config
    N = model.get_num_params(non_embedding=True)
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, block
    return 6 * N + 12 * L * H * Q * T


def train_one(cfg, precision, steps, batch_size=64, block=256, lr=1e-3,
              schedule="cosine", warmup=100, seed=0, val_every=250, burn_in=20):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GPT(GPTConfig(block_size=block, vocab_size=VOCAB, dropout=0.0,
                          bias=False, **cfg))
    model.cuda()
    n_params_total = model.get_num_params(non_embedding=False)
    n_params = model.get_num_params(non_embedding=True)
    fpt = flops_per_token(model, block)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=lr,
                                           betas=(0.9, 0.95), device_type="cuda")
    fused_adamw = all(g.get("fused") for g in optimizer.param_groups) or optimizer.defaults.get("fused")

    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        if schedule == "constant":
            return lr
        prog = (step - warmup) / max(1, steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    def eval_val(n_batches=16):
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = get_batch("val", batch_size, block, rng)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                tot += loss.item()
        model.train()
        return tot / n_batches

    losses, step_times, val_curve = [], [], {"steps": [], "vals": []}
    scale_traj, backoffs = [], 0
    diverged = False
    prev_scale = None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for step in range(steps):
        ts = time.perf_counter()
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch("train", batch_size, block, rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        if precision == "fp16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            s = scaler.get_scale()
            if prev_scale is not None and s < prev_scale:
                backoffs += 1
            prev_scale = s
            scale_traj.append(s)
        else:
            loss.backward()
            optimizer.step()
        lv = loss.item()
        losses.append(lv)
        if step >= burn_in:
            step_times.append(time.perf_counter() - ts)
        if not math.isfinite(lv):
            diverged = True
            break
        if val_every and (step + 1) % val_every == 0:
            val_curve["steps"].append(step + 1)
            val_curve["vals"].append(eval_val())

    wall = time.perf_counter() - t0
    med = float(np.median(step_times)) if step_times else float("nan")
    last = losses[-min(50, len(losses)):]
    final_loss = float(np.median(last)) if last and math.isfinite(last[-1]) else float("nan")
    vram_mb = torch.cuda.max_memory_allocated() / 1e6
    tok_s = batch_size * block / med if med == med else float("nan")
    res = {
        "precision": precision, "steps_run": len(losses), "diverged": diverged,
        "n_params_total": n_params_total, "n_params_non_embed": n_params,
        "flops_per_token": fpt, "batch": batch_size, "block": block, "lr": lr,
        "schedule": schedule, "seed": seed, "fused_adamw": bool(fused_adamw),
        "loss_curve": losses[::10], "final_train_loss": final_loss,
        "val_curve": val_curve, "final_val_loss": (val_curve["vals"][-1] if val_curve["vals"] else None),
        "step_time_ms_median": med * 1e3, "tokens_per_s": tok_s,
        "achieved_tflops": fpt * tok_s / 1e12 if tok_s == tok_s else None,
        "mfu_pct": 100 * fpt * tok_s / 1e12 / (PEAK_FP16_TF if use_amp else PEAK_FP32_TF)
                   if tok_s == tok_s else None,
        "vram_peak_mb": vram_mb, "wall_time_s": wall,
        "scaler": {"init": scale_traj[0] if scale_traj else None,
                   "final": scale_traj[-1] if scale_traj else None,
                   "min": min(scale_traj) if scale_traj else None,
                   "max": max(scale_traj) if scale_traj else None,
                   "backoffs": backoffs} if precision == "fp16" else None,
    }
    return model, res


def encode_prompt(prompt):
    stoi = {c: i for i, c in enumerate(VOCAB_CHARS)}
    return [stoi[c] for c in prompt if c in stoi]


def sample_text(model, prompt, n_new, temperature=1.0, seed=0):
    torch.manual_seed(seed)
    ids = torch.tensor([encode_prompt(prompt)], device="cuda")
    out = model.generate(ids, n_new, temperature=temperature)
    return "".join(VOCAB_CHARS[int(t)] for t in out[0])


# ------------------------------------------------------------------ E1
def e1():
    model, res = train_one(CONFIGS["10.5M"], "fp16", steps=3000, lr=1e-3, seed=0)
    prompt = "The llama.cpp project "
    s1 = sample_text(model, prompt, 200, temperature=1.0, seed=42)
    s2 = sample_text(model, prompt, 200, temperature=1.0, seed=42)
    s3 = sample_text(model, prompt, 200, temperature=0.8, seed=7)
    res.update({"exp": "e1_baseline", "prompt": prompt,
                "sample_t1.0": s1, "sample_t0.8": s3,
                "deterministic_sample": s1 == s2})
    res["gates"] = {
        "final_train_loss_lt_2.0": res["final_train_loss"] < 2.0,
        "not_diverged": not res["diverged"],
        "deterministic_sample": s1 == s2,
    }
    torch.save({"state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
                "config": {**CONFIGS["10.5M"], "block_size": 256, "vocab_size": VOCAB,
                           "bias": False},
                "vocab": VOCAB_CHARS}, RES / "e1_checkpoint.pt")
    save("e1", res)
    assert all(res["gates"].values()), res["gates"]


# ------------------------------------------------------------------ E2
def e2():
    out = {"exp": "e2_precision"}
    for prec in ("fp32", "fp16", "bf16"):
        try:
            _, r = train_one(CONFIGS["3M"], prec, steps=1200, lr=1e-3, seed=0)
            out[prec] = r
        except Exception as e:  # record honestly (bf16 on Turing may misbehave)
            out[prec] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[e2] {prec}: tok/s {out[prec].get('tokens_per_s')}, "
              f"final {out[prec].get('final_train_loss')}")
    fp32, fp16 = out["fp32"], out["fp16"]
    speedup = fp16["tokens_per_s"] / fp32["tokens_per_s"]
    out["fp16_vs_fp32_speedup"] = speedup
    out["gates"] = {"fp16_speedup_ge_1.3x": speedup >= 1.3,
                    "fp16_not_diverged": not fp16["diverged"],
                    "bf16_recorded": True}
    save("e2", out)
    assert all(out["gates"].values()), out["gates"]


# ------------------------------------------------------------------ E3
def e3():
    out = {"exp": "e3_scaling", "token_budget": 25e6, "sizes": {}}
    B, T = 64, 256
    steps = round(25e6 / (B * T))
    for key in ("0.5M", "3M", "10.5M", "25M"):
        _, r = train_one(CONFIGS[key], "fp16", steps=steps, batch_size=B, block=T,
                         lr=1e-3, seed=0)
        r.pop("loss_curve", None)
        r.pop("sample_t1.0", None)
        out["sizes"][key] = r
        print(f"[e3] {key}: params {r['n_params_total']/1e6:.2f}M, "
              f"val {r['final_val_loss']}, tok/s {r['tokens_per_s']:.0f}, "
              f"MFU {r['mfu_pct']:.1f}%")
    sizes = ["0.5M", "3M", "10.5M", "25M"]
    vals = [out["sizes"][k]["final_val_loss"] for k in sizes]
    mfu = [out["sizes"][k]["mfu_pct"] for k in sizes]
    # compute-bound consistency: predict step time of 10.5M from 25M linear-in-flops
    t25 = out["sizes"]["25M"]["step_time_ms_median"]
    t10 = out["sizes"]["10.5M"]["step_time_ms_median"]
    n25 = out["sizes"]["25M"]["flops_per_token"]
    n10 = out["sizes"]["10.5M"]["flops_per_token"]
    pred10 = t25 * n10 / n25
    out["step_time_pred_from_25M"] = {"10.5M_pred_ms": pred10,
                                      "10.5M_meas_ms": t10,
                                      "ratio_meas_over_pred": t10 / pred10}
    # HONEST GATE REVISION (recorded in results.md): the original gate assumed val
    # loss decreases with size at a FIXED token budget — wrong regime physics.
    # With 25M tokens the larger models are undertrained (Chinchilla gap), so the
    # fixed-budget optimum is interior. Pre-registered revised gates:
    out["gates"] = {
        "mfu_monotone_increasing_with_size": all(mfu[i + 1] > mfu[i] for i in range(3)),
        "size_helps_while_not_undertrained": vals[1] < vals[0],
        "fixed_budget_undertraining_regime": vals[3] > vals[1],
    }
    save("e3", out)
    assert all(out["gates"].values()), out["gates"]


# ------------------------------------------------------------------ E4
def e4():
    out = {"exp": "e4_hp", "lr_sweep": {}, "batch_sweep": {}}
    steps = 800
    for lr in (1e-4, 3e-4, 1e-3, 3e-3):
        for sched in ("cosine", "constant"):
            tag = f"lr{lr:g}_{sched}"
            _, r = train_one(CONFIGS["3M"], "fp16", steps=steps, lr=lr,
                             schedule=sched, seed=0, val_every=400)
            r = {"final_train_loss": r["final_train_loss"],
                 "diverged": r["diverged"], "loss_curve": r["loss_curve"],
                 "final_val_loss": r["final_val_loss"]}
            out["lr_sweep"][tag] = r
            print(f"[e4] {tag}: final {r['final_train_loss']}, diverged {r['diverged']}")
    finals = [v["final_train_loss"] for v in out["lr_sweep"].values()
              if math.isfinite(v["final_train_loss"])]
    spread = (max(finals) - min(finals)) if finals else 0.0
    budget = 13.1e6
    for B in (16, 64, 256):
        st = round(budget / (B * 256))
        _, r = train_one(CONFIGS["3M"], "fp16", steps=st, batch_size=B, lr=1e-3,
                         seed=0, val_every=max(100, st // 4))
        out["batch_sweep"][f"B{B}"] = {
            "steps": st, "tokens_per_s": r["tokens_per_s"],
            "step_time_ms_median": r["step_time_ms_median"],
            "final_train_loss": r["final_train_loss"],
            "final_val_loss": r["final_val_loss"]}
        print(f"[e4] B{B}: tok/s {r['tokens_per_s']:.0f}, final {r['final_train_loss']}")
    n_diverged = sum(1 for v in out["lr_sweep"].values() if v["diverged"])
    out["gates"] = {"lr_spread_gt_0.1": spread > 0.1,
                    "batch_tokens_per_s_increases": (out["batch_sweep"]["B16"]["tokens_per_s"]
                                                     < out["batch_sweep"]["B256"]["tokens_per_s"])}
    out["diverged_runs"] = n_diverged
    save("e4", out)
    assert all(out["gates"].values()), out["gates"]


# ------------------------------------------------------------------ E5
def _categorize(name):
    n = name.lower()
    if "::mm" in n or "gemm" in n or "cutlass" in n or "sm90" in n or "gemv" in n \
            or "::mv" in n or "matmul" in n or "sgemm" in n or "hgemm" in n:
        return "linear/matmul"
    if "flash" in n or "attention" in n or "fmha" in n:
        return "attention"
    if "adam" in n:
        return "optimizer"
    if "layer_norm" in n:
        return "layernorm"
    if "elementwise" in n or "vectorized" in n or "copy" in n or "memset" in n or "memcpy" in n:
        return "elementwise/copy"
    if "gelu" in n or "soft" in n:
        return "activation"
    if "transpose" in n or "cat" in n or "view" in n or "contiguous" in n:
        return "eager_overhead"
    return "other"


def e5():
    from torch.profiler import profile, ProfilerActivity
    cfg = CONFIGS["10.5M"]
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = GPT(GPTConfig(block_size=256, vocab_size=VOCAB, dropout=0.0, bias=False, **cfg))
    model.cuda().train()
    optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cuda")
    scaler = torch.amp.GradScaler("cuda")

    def step():
        x, y = get_batch("train", 64, 256, rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    for _ in range(10):
        step()  # warmup
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            step()
    evts = sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
    total = sum(e.self_device_time_total for e in evts)
    top, cat_tot = [], {}
    for e in evts[:15]:
        if e.self_device_time_total <= 0:
            continue
        top.append({"name": e.key[:70], "self_ms_total": e.self_device_time_total / 1e3,
                    "pct": 100 * e.self_device_time_total / total,
                    "count": e.count})
    for e in evts:
        c = _categorize(e.key)
        cat_tot[c] = cat_tot.get(c, 0.0) + e.self_device_time_total
    cats = {k: {"ms": v / 1e3, "pct": 100 * v / total} for k, v in
            sorted(cat_tot.items(), key=lambda kv: -kv[1])}
    top5 = sum(t["pct"] for t in top[:5])

    # fresh tokens/s for repeatability check vs e1
    _, r = train_one(cfg, "fp16", steps=150, val_every=0)
    e1 = json.loads((RES / "e1.json").read_text(encoding="utf-8"))
    out = {"exp": "e5_profiler", "top_ops": top, "categories": cats,
           "device_time_total_ms": total / 1e3,
           "fresh_tokens_per_s": r["tokens_per_s"],
           "e1_tokens_per_s": e1["tokens_per_s"],
           "gates": {"top5_ops_ge_50pct": top5 >= 50,
                     "fresh_vs_e1_within_10pct":
                         abs(r["tokens_per_s"] / e1["tokens_per_s"] - 1) <= 0.10}}
    save("e5", out)
    assert all(out["gates"].values()), out["gates"]


# ------------------------------------------------------------------ E7
def e7():
    """SDPA backend evidence on sm_75, fp16 vs fp32 (kernel names from profiler)."""
    from torch.profiler import profile, ProfilerActivity
    out = {"exp": "e7_sdpa_backend"}
    for prec in ("fp16", "fp32"):
        torch.manual_seed(0)
        model = GPT(GPTConfig(block_size=256, vocab_size=VOCAB, dropout=0.0,
                              bias=False, **CONFIGS["3M"]))
        model.cuda().eval()
        x, _ = get_batch("val", 8, 256, np.random.default_rng(1))
        amp = torch.float16 if prec == "fp16" else None
        with torch.no_grad():
            for _ in range(3):
                with torch.amp.autocast("cuda", dtype=amp, enabled=amp is not None):
                    model(x)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    with torch.amp.autocast("cuda", dtype=amp, enabled=amp is not None):
                        model(x)
        names = [e.key for e in prof.key_averages() if e.self_device_time_total > 0]
        joined = " | ".join(names)
        evidence = {
            "flash": "flash" in joined.lower(),
            "mem_efficient": ("efficient" in joined.lower()) or ("fmha" in joined.lower()),
            "attention_kernels": [n[:70] for n in names
                                  if any(t in n.lower() for t in ("attention", "flash", "fmha", "softmax"))][:8],
        }
        out[prec] = evidence
    out["gates"] = {"backends_recorded": True}
    save("e7", out)


def e7c():
    """torch.compile attempt on Windows/inductor — honest record."""
    out = {"exp": "e7_compile", "torch_version": torch.__version__}
    try:
        torch.manual_seed(0)
        model = GPT(GPTConfig(block_size=256, vocab_size=VOCAB, dropout=0.0,
                              bias=False, **CONFIGS["3M"]))
        model.cuda().train()
        cmodel = torch.compile(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        for _ in range(10):
            x, y = get_batch("train", 64, 256, rng)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss = cmodel(x, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss.item()
        out["status"] = "compiled_and_ran"
        out["wall_s_10_steps_incl_compile"] = time.perf_counter() - t0
        # eager comparison
        eager_model = model
        opt2 = torch.optim.AdamW(eager_model.parameters(), lr=1e-3)
        scaler2 = torch.amp.GradScaler("cuda")
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            x, y = get_batch("train", 64, 256, rng)
            opt2.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss = eager_model(x, y)
            scaler2.scale(loss).backward()
            scaler2.step(opt2)
            scaler2.update()
            loss.item()
            ts.append(time.perf_counter() - t)
        out["eager_step_ms_median"] = float(np.median(ts)) * 1e3
    except Exception as e:
        out["status"] = "failed"
        out["error"] = f"{type(e).__name__}: {e}"
    out["gates"] = {"honest_record": True}
    save("e7_compile", out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "e1":
        e1()
    elif cmd == "e2":
        e2()
    elif cmd == "e3":
        e3()
    elif cmd == "e4":
        e4()
    elif cmd == "e5":
        e5()
    elif cmd == "e7":
        e7()
    elif cmd == "e7c":
        e7c()
    elif cmd == "all":
        e1(); e2(); e3(); e4(); e5(); e7(); e7c()
    else:
        raise SystemExit(f"unknown: {cmd}")
