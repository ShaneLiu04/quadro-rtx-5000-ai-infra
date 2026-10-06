"""framework_lab.py — AR007 experiment driver (E0-E5), gates + JSON (frameworks lab).

Usage: python framework_lab.py {prep|e0|e1|e2|e3|e5|all}

Upstream clones are imported via sys.path with ZERO modification:
- bitsandbytes 0.50.3.dev0 clone (degraded: native DLL absent)
- tinygrad clone (lazy graph OK; realize fails on this machine — recorded honestly)
- nanoGPT model.py (06-training, proven in AR006)
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).parent
INFRA = LAB.parent.parent
RESULTS = LAB / "results"
FIGS = LAB / "figs"
DATA = LAB / "data"

BNB = str(INFRA / "07-frameworks" / "bitsandbytes")
TINYGRAD = str(INFRA / "07-frameworks" / "tinygrad")
NANOGPT = str(INFRA / "06-training" / "nanoGPT")

SEED = 20261006


def env_block() -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_cc": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "cpu": platform.processor(),
    }


def jdump(name: str, obj: dict):
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[json] {p.name} written ({p.stat().st_size} bytes)")


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED % (2**32))


# ------------------------------------------------------------------ E0

def e0():
    """Environment baseline: bnb degraded dispatch, _int_mm bit-exactness, tinygrad UOp introspection."""
    set_seed()
    out = {"exp": "e0_env", "env": env_block()}

    # --- 1. bnb degraded import ------------------------------------------------
    sys.path.insert(0, BNB)
    bnb_probe = {}
    try:
        import bitsandbytes as bnb  # noqa: F401  (prints degraded load error to stderr)

        from bitsandbytes import cextension
        from bitsandbytes import functional as bF

        bnb_probe["version"] = bnb.__version__
        bnb_probe["lib_class"] = type(cextension.lib).__name__
        bnb_probe["degraded"] = "ErrorHandlerMock" in type(cextension.lib).__name__
        bnb_probe["backend_env_default"] = "cuda"  # cextension BNB_BACKEND default

        # dispatch table dump for 3 representative ops
        dumps = {}
        for op in ["bitsandbytes::int8_vectorwise_quant", "bitsandbytes::int8_linear_matmul", "bitsandbytes::quantize_4bit"]:
            try:
                dumps[op] = torch._C._dispatch_dump_table(op)
            except Exception as e:  # pragma: no cover
                dumps[op] = f"<dump failed: {e}>"
        bnb_probe["dispatch_dump"] = dumps

        # the (None, None, None) mystery call on CPU
        A = torch.randn(32, 64)
        r = bF.int8_vectorwise_quant(A)
        bnb_probe["cpu_call_returns"] = [type(x).__name__ if not isinstance(x, torch.Tensor) else f"Tensor{tuple(x.shape)}" for x in r]
        bnb_probe["cpu_call_all_none"] = all(x is None for x in r)

        # same op on CUDA: expect native-lib error (registered cuda kernel calls missing DLL)
        if torch.cuda.is_available():
            try:
                bF.int8_vectorwise_quant(A.cuda())
                bnb_probe["cuda_call"] = "ok"
            except Exception as e:
                bnb_probe["cuda_call_error"] = f"{type(e).__name__}: {str(e)[:200]}"

        # NF4 codebook cross-check: bnb get_4bit_type('nf4') vs our official-table
        # replication AND the raw ndtri-quantile variant (they differ: endpoints pinned to +-1)
        try:
            t = bF.get_4bit_type("nf4", device="cpu")
            bnb_first16 = t.flatten()[:16].tolist()
            sys.path.insert(0, str(LAB))
            from quant_ops import nf4_levels, nf4_raw_levels

            ours = nf4_levels().tolist()
            raw = nf4_raw_levels().tolist()
            bnb_probe["nf4_map_shape"] = list(t.shape)
            bnb_probe["nf4_first16_bnb"] = [round(v, 6) for v in bnb_first16]
            bnb_probe["nf4_first16_ours"] = [round(v, 6) for v in ours]
            bnb_probe["nf4_first16_ndtri_raw"] = [round(v, 6) for v in raw]
            bnb_probe["nf4_map_consistent"] = bool(np.allclose(bnb_first16, ours, atol=1e-5))
            bnb_probe["nf4_map_equals_raw_ndtri"] = bool(np.allclose(bnb_first16, raw, atol=1e-5))
        except Exception:
            bnb_probe["nf4_map_consistent"] = None
            bnb_probe["nf4_map_error"] = traceback.format_exc(limit=2)
    except Exception:
        bnb_probe["import_error"] = traceback.format_exc(limit=4)
    out["bnb"] = bnb_probe

    # --- 2. torch._int_mm bit-exactness ---------------------------------------
    mm = {}
    if torch.cuda.is_available():
        k = 512
        a = torch.randint(-127, 128, (k, k), dtype=torch.int8, device="cuda")
        b = torch.randint(-127, 128, (k, k), dtype=torch.int8, device="cuda")
        c = torch._int_mm(a, b)
        ref = (a.double() @ b.double()).to(torch.int64)  # exact: |sum| <= 127*127*512 < 2^53
        mm["cuda_bitexact"] = bool((c.to(torch.int64) == ref).all().item())
        mm["cuda_max_abs_diff"] = int((c.to(torch.int64) - ref).abs().max().item())
    ac = torch.randint(-127, 128, (256, 256), dtype=torch.int8)
    bc = torch.randint(-127, 128, (256, 256), dtype=torch.int8)
    try:
        cc = torch._int_mm(ac, bc)
        refc = (ac.double() @ bc.double()).to(torch.int64)
        mm["cpu_works"] = bool((cc.to(torch.int64) == refc).all().item())
    except Exception as e:
        mm["cpu_works"] = False
        mm["cpu_error"] = f"{type(e).__name__}: {str(e)[:150]}"
    out["int_mm"] = mm

    # --- 3. tinygrad UOp introspection + realize failure (honest record) --------
    tg = {}
    try:
        sys.path.insert(0, TINYGRAD)
        from tinygrad import Tensor

        def uop_hist(t: Tensor):
            root = t.uop
            seen = set()
            stack = [root]
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                stack.extend(u.src)
            hist = {}
            for u in seen:
                k = str(u.op).replace("Ops.", "")
                hist[k] = hist.get(k, 0) + 1
            return len(seen), dict(sorted(hist.items(), key=lambda kv: -kv[1]))

        # deterministic compute graph (no PRNG noise): matmul + relu + add
        n = 64
        ta = Tensor.arange(n * n).reshape(n, n) / 100.0
        tb = Tensor.arange(n * n).reshape(n, n) / 97.0
        tc = (ta @ tb).relu() + 1.0
        nodes, hist = uop_hist(tc)
        tg["det_graph_nodes"] = nodes
        tg["det_graph_op_hist"] = hist

        # random graph for contrast (PRNG ops visible: THREEFRY etc.)
        tr = (Tensor.randn(n, n) @ Tensor.randn(n, n)).relu()
        nodes2, hist2 = uop_hist(tr)
        tg["rand_graph_nodes"] = nodes2
        tg["rand_graph_op_hist"] = hist2

        # realize failures: CUDA + TORCH, full tracebacks (honest record)
        fails = {}
        for dev in ["CUDA", "TORCH"]:
            try:
                x = Tensor.randn(128, 128).to(dev)
                y = Tensor.randn(128, 128).to(dev)
                (x @ y).realize()
                fails[dev.lower()] = "ok"
            except Exception:
                fails[dev.lower()] = traceback.format_exc()
        tg["realize_fail"] = fails
    except Exception:
        tg["import_error"] = traceback.format_exc(limit=4)
    out["tinygrad"] = tg

    # gates
    gates = {
        "g_intmm_cuda_bitexact": mm.get("cuda_bitexact") is True,
        "g_bnb_degraded_recorded": bnb_probe.get("degraded") is True,
        "g_tinygrad_uop_introspected": tg.get("det_graph_nodes", 0) > 10,
        "g_tinygrad_realize_failure_recorded": isinstance(tg.get("realize_fail", {}).get("cuda"), str),
    }
    out["gates"] = gates

    jdump("e0_env.json", out)
    print(json.dumps(gates, indent=1))
    assert all(gates.values()), f"E0 gates failed: {gates}"
    print("[E0] all gates PASS")


# ------------------------------------------------------------------ E1

def _train_toy_mlp(steps=300):
    """Train a small next-char MLP on the lab corpus to obtain REAL trained weights.

    Reuses data/tokens.npy from prep. Returns dict of fp32 weight tensors (cuda).
    """
    tokens = np.load(DATA / "tokens.npy")
    vocab = int(tokens.max()) + 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(SEED)
    tok_gpu = torch.from_numpy(tokens.astype(np.int64)).to(dev)
    ctx = torch.arange(4, device=dev)

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, 256)
            self.fc1 = torch.nn.Linear(256 * 4, 1024)
            self.fc2 = torch.nn.Linear(1024, 1024)
            self.head = torch.nn.Linear(1024, vocab)

        def forward(self, x):
            h = self.emb(x).flatten(1)
            h = torch.relu(self.fc1(h))
            h = torch.relu(self.fc2(h))
            return self.head(h)

    m = MLP().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    n = len(tokens)
    losses = []
    for i in range(steps):
        pos = torch.randint(0, len(tokens) - 8, (512,), generator=g, device=dev)
        xb = tok_gpu[pos.unsqueeze(1) + ctx]  # (512, 4) int64
        yb = tok_gpu[pos + 4]  # (512,)
        loss = torch.nn.functional.cross_entropy(m(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    weights = {name: p.detach().float().cpu() for name, p in m.named_parameters()}
    stats = {"steps": steps, "final_loss": losses[-1], "loss_first": losses[0], "vocab": vocab}
    return weights, stats


def e1():
    """NF4 vs FP4 vs INT4: codebook fidelity, blocksize sweep, real-weight end-to-end."""
    set_seed()
    from quant_ops import get_levels, blockwise_quant, NF4_OFFICIAL, nf4_raw_levels

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"exp": "e1_nf4", "env": env_block(), "device": dev}

    methods = ["nf4", "nf4_raw", "fp4", "int4"]
    levels = {m: get_levels(m).tolist() for m in methods}
    out["levels"] = levels

    # ---- synthetic N(0,1) sweep ------------------------------------------------
    W = torch.randn(4096, 4096, device=dev)  # 16.7M weights, absmax-blocked
    sweep = {}
    blocksizes = [32, 64, 128, 256, 512, 1024]
    for m in methods:
        lv = get_levels(m, device=dev)
        sweep[m] = {}
        for b in blocksizes:
            r = blockwise_quant(W, lv, b)
            sweep[m][str(b)] = r["rel_rmse"]
    out["synth_sweep"] = sweep
    out["synth_absmax_mean"] = float(W.abs().amax(dim=1).mean())

    # gate G1: NF4 < INT4 at default blocksize 64 on N(0,1)
    g1 = sweep["nf4"]["64"] < sweep["int4"]["64"]
    out["gate_nf4_vs_int4"] = bool(g1)
    out["gate_detail"] = {"nf4_64": sweep["nf4"]["64"], "int4_64": sweep["int4"]["64"], "fp4_64": sweep["fp4"]["64"], "nf4_raw_64": sweep["nf4_raw"]["64"]}

    # error histogram data (for fig): method × per-element err at b=64
    hists = {}
    for m in methods:
        lv = get_levels(m, device=dev)
        r = blockwise_quant(W, lv, 64)
        e = (r["dequant"] - W).flatten()
        hists[m] = torch.histc(e, bins=81, min=-0.05, max=0.05).cpu().tolist()
    out["err_hist_bins"] = 81
    out["err_hist_range"] = [-0.05, 0.05]
    out["err_hist"] = hists
    del W
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---- real trained weights ---------------------------------------------------
    weights, tstats = _train_toy_mlp(steps=300)
    wres = {}
    for name, p in weights.items():
        if p.numel() < 4096:
            continue
        pt = p.to(dev)
        st = {"numel": p.numel(), "std": float(pt.std()), "kurtosis": float(((pt - pt.mean()) ** 4).mean() / pt.var() ** 2)}
        for m in ["nf4", "int4"]:
            r = blockwise_quant(pt, get_levels(m, device=dev), 64)
            st[f"rel_rmse_{m}"] = r["rel_rmse"]
        wres[name] = st
    out["weights"] = wres
    out["toy_train"] = tstats

    gates = {"g1_nf4_beats_int4_on_N01": bool(g1), "g1_real_weights_recorded": len(wres) >= 3}
    out["gates"] = gates
    jdump("e1_nf4.json", out)
    print(json.dumps({"gate_detail": out["gate_detail"], "gates": gates, "toy": tstats}, indent=1))
    assert gates["g1_nf4_beats_int4_on_N01"], "G1 failed: NF4 did not beat INT4 on N(0,1)"
    assert gates["g1_real_weights_recorded"]
    print("[E1] all gates PASS")


# ------------------------------------------------------------------ E2

def _bench(fn, warmup=20, reps=50):
    """Median wall time (s) with CUDA sync, AR002-006 timing discipline."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def e2():
    """LLM.int8: vectorwise int8 + outlier decomposition + INT8-TC throughput."""
    set_seed()
    from quant_ops import vw_quant_rows, int8_mm_dequant, llm_int8_matmul

    assert torch.cuda.is_available(), "E2 requires CUDA"
    dev = "cuda"
    out = {"exp": "e2_llmint8", "env": env_block()}

    # ---- 1. uniform-input fidelity ---------------------------------------------
    M = K = N = 2048
    X = torch.randn(M, K, device=dev)
    W = torch.randn(N, K, device=dev) / K**0.5
    ref = X @ W.t()
    C8, meta = llm_int8_matmul(X, W, 0.0)
    rel = float((C8 - ref).norm() / ref.norm())
    # fp16 baseline for context: what plain half precision costs
    C16 = (X.half() @ W.t().half()).float()
    rel16 = float((C16 - ref).norm() / ref.norm())
    out["uniform"] = {"rel_err_int8": rel, "rel_err_fp16": rel16, "shape": [M, K, N]}

    # ---- 2. heavy-tail activations + threshold sweep ----------------------------
    # LLM.int8 paper phenomenon: a few systematic outlier features (columns) with
    # magnitudes >> rest destroy per-row absmax scales. Simulate: 8 of 2048 cols x16.
    Xh = torch.randn(M, K, device=dev)
    outlier_feature_cols = torch.randperm(K, device=dev)[:8]
    Xh[:, outlier_feature_cols] *= 16.0
    Wh = torch.randn(N, K, device=dev) / K**0.5
    refh = Xh @ Wh.t()
    base_scale = refh.norm()

    sweep = {}
    for tau in [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        C, meta = llm_int8_matmul(Xh, Wh, tau)
        err = float((C - refh).norm() / base_scale)
        sweep[str(tau)] = {"rel_err": err, **meta}
    out["threshold_sweep"] = sweep
    out["outlier_feature_cols"] = 8

    # ---- 3. scaling granularity comparison (same heavy-tail input) ---------------
    sc = {}
    # per-tensor
    sX = X.abs().max(); sW = W.abs().max()
    qX = torch.round(X * 127 / sX).clamp(-127, 127).to(torch.int8)
    qW = torch.round(W * 127 / sW).clamp(-127, 127).to(torch.int8)
    sc["per_tensor"] = float((int8_mm_dequant(qX, sX.expand(M), qW, sW.expand(N)) - ref).norm() / ref.norm())
    # per-row (activations only)
    qX2, sX2, _, _ = vw_quant_rows(X, 0.0)
    qW2, sW2, _, _ = vw_quant_rows(W, 0.0)
    sc["per_row_act"] = float((int8_mm_dequant(qX2, sX2, qW2, sW2) - ref).norm() / ref.norm())
    # per-row on heavy-tail input (shows why granularity alone is not enough)
    qXh, sXh, _, _ = vw_quant_rows(Xh, 0.0)
    qWh, sWh, _, _ = vw_quant_rows(Wh, 0.0)
    sc["per_row_heavytail"] = float((int8_mm_dequant(qXh, sXh, qWh, sWh) - refh).norm() / base_scale)
    out["scaling_compare"] = sc

    # ---- 4. throughput: INT8 TC vs FP16 (kernel-only + end-to-end) ----------------
    tp = {}
    for n in [2048, 4096]:
        A16 = torch.randn(n, n, device=dev, dtype=torch.float16)
        B16 = torch.randn(n, n, device=dev, dtype=torch.float16)
        A8 = torch.randint(-127, 128, (n, n), device=dev, dtype=torch.int8)
        B8 = torch.randint(-127, 128, (n, n), device=dev, dtype=torch.int8)

        t_fp16 = _bench(lambda: A16 @ B16)
        t_int8 = _bench(lambda: torch._int_mm(A8, B8))
        Xf = torch.randn(n, n, device=dev)
        Wf = torch.randn(n, n, device=dev)
        t_e2e8 = _bench(lambda: llm_int8_matmul(Xf, Wf, 0.0))
        flops = 2 * n**3
        tp[str(n)] = {
            "fp16_kernel_ms": t_fp16 * 1e3,
            "fp16_tflops": flops / t_fp16 / 1e12,
            "int8_kernel_ms": t_int8 * 1e3,
            "int8_top_int32": flops / t_int8 / 1e12,
            "int8_e2e_ms": t_e2e8 * 1e3,
            "int8_e2e_top": flops / t_e2e8 / 1e12,
            "ratio_kernel": t_fp16 / t_int8,
            "ratio_e2e": t_fp16 / t_e2e8,
        }
        del A16, B16, A8, B8
        torch.cuda.empty_cache()
    out["throughput"] = tp

    # ---- 4b. layout attribution: cuBLAS int8 TC fast path needs TN layout -------
    # Finding: default row-major _int_mm lands on the SLOW int8 path; column-major
    # B engages the TN IMMA path (1.9x faster). This is WHY bnb's igemmlt uses a
    # custom weight layout [source] csrc layout conversion. Measured [local].
    A8r = torch.randint(-127, 128, (2048, 2048), device=dev, dtype=torch.int8)
    B8r = torch.randint(-127, 128, (2048, 2048), device=dev, dtype=torch.int8)
    B8c = torch.randint(-127, 128, (2048, 2048), device=dev, dtype=torch.int8).t()  # col-major (K,N)
    A8c = torch.randint(-127, 128, (2048, 2048), device=dev, dtype=torch.int8).t()
    lay = {}
    for name, a, b in [("rowA_rowB", A8r, B8r), ("rowA_colB", A8r, B8c), ("colA_rowB", A8c, B8r), ("colA_colB", A8c, B8c)]:
        t = _bench(lambda: torch._int_mm(a, b), warmup=10, reps=30)
        lay[name] = {"ms": t * 1e3, "tops": 2 * 2048**3 / t / 1e12}
    lay["fp16_reference_ms"] = tp["2048"]["fp16_kernel_ms"]
    out["layout_attribution"] = lay

    # gate G2: INT8 kernel-only >= 1.2x fp16 at 2048 (Turing 2x peak minus overheads)
    g2 = tp["2048"]["ratio_kernel"] >= 1.2
    out["gate_int8_ratio"] = bool(g2)
    out["gate_int8_revision"] = {
        "pre_registered": "int8 kernel-only >= 1.2x fp16 (row-major default)",
        "result": f"FAILED at {tp['2048']['ratio_kernel']:.2f}x: default layout lands on cuBLAS int8 SLOW path (NT)",
        "revised_finding": "with column-major B (TN layout) _int_mm hits the IMMA path: "
        + f"{lay['rowA_colB']['tops']:.1f} TOPS = {lay['rowA_rowB']['ms']/lay['rowA_colB']['ms']:.2f}x vs default, "
        + f"but only {lay['rowA_colB']['tops']/tp['2048']['fp16_tflops']:.2f}x of fp16 — sm_75 torch._int_mm "
        "does not reach the 2x paper peak; bnb's custom igemmlt layouts exist precisely for this reason",
    }

    gates = {
        "g2_int8_kernel_ge_1p2x_fp16": bool(g2),
        "g_threshold_sweep_recorded": len(sweep) >= 5,
        "g_uniform_err_sane": 0.001 < out["uniform"]["rel_err_int8"] < 0.2,
    }
    out["gates"] = gates
    jdump("e2_llmint8.json", out)
    print(json.dumps({"uniform": out["uniform"], "sweep": {k: round(v["rel_err"], 5) for k, v in sweep.items()},
                      "scaling": {k: round(v, 5) for k, v in sc.items()},
                      "tp2048": tp["2048"], "layout": {k: (round(v["ms"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in lay.items()},
                      "gates": gates}, indent=1))
    assert gates["g_threshold_sweep_recorded"] and gates["g_uniform_err_sane"]
    if not g2:
        print("[E2][HONEST] gate G2 (int8 >= 1.2x fp16) FAILED — recorded for revision analysis")
    else:
        print("[E2] all gates PASS")


# ------------------------------------------------------------------ E3

def _train_gpt(opt_mode: str, steps=800, record_every=10):
    """Train nanoGPT 4L-256 (~3M) with fp32-AdamW or our Adam8bit reimplementation.

    opt_mode: 'fp32' (torch.optim.AdamW) | 'int8' (quant_ops.Adam8bitStates).
    Same seed, data, lr. Returns (loss_curve, step_ms_median, optim_state_bytes, final_loss).
    """
    from quant_ops import Adam8bitStates

    tokens = np.load(DATA / "tokens.npy")
    vocab = int(tokens.max()) + 1
    dev = "cuda"
    torch.manual_seed(SEED)
    tok_gpu = torch.from_numpy(tokens.astype(np.int64)).to(dev)

    sys.path.insert(0, NANOGPT)
    from model import GPT, GPTConfig  # nanoGPT upstream, zero modification (AR006-proven)

    cfg = GPTConfig(vocab_size=vocab, block_size=256, n_layer=4, n_embd=256, n_head=4, dropout=0.0)
    model = GPT(cfg).to(dev)
    model.train()

    g = torch.Generator(device=dev).manual_seed(SEED)
    ctx = torch.arange(256, device=dev)

    if opt_mode == "fp32":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        states = None
    else:
        states = None  # created inside the memory measurement block below

    def get_batch(bs=32):
        pos = torch.randint(0, len(tokens) - 257, (bs,), generator=g, device=dev)
        xb = tok_gpu[pos.unsqueeze(1) + ctx]
        yb = tok_gpu[pos.unsqueeze(1) + ctx + 1]
        return xb, yb

    # measure optimizer-state bytes (memory_allocated delta across state creation)
    if opt_mode == "fp32":
        torch.cuda.synchronize()
        mem0 = torch.cuda.memory_allocated()
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        model.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        mem1 = torch.cuda.memory_allocated()
        # analytic state bytes from opt.state (measured delta is contaminated by
        # backward intermediates — recorded separately for honesty)
        optim_state_bytes = sum(t.numel() * t.element_size() for st in opt.state.values() for t in st.values())
        measured_delta = mem1 - mem0
    else:
        torch.cuda.synchronize()
        mem0 = torch.cuda.memory_allocated()
        states = [Adam8bitStates(p.shape, dev, block=256) for p in model.parameters()]
        torch.cuda.synchronize()
        mem1 = torch.cuda.memory_allocated()
        optim_state_bytes = sum(st.state_bytes() for st in states)
        measured_delta = mem1 - mem0

    losses, times = [], []
    for i in range(steps):
        xb, yb = get_batch()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, loss = model(xb, yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if opt_mode == "fp32":
            opt.step()
        else:
            for p, st in zip(model.parameters(), states):
                if p.grad is not None:
                    st.step(p, p.grad, lr=1e-3, weight_decay=0.01)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if i % record_every == 0:
            losses.append(float(loss))
    return {
        "loss_curve": losses,
        "step_ms_median": float(np.median(times)) * 1e3,
        "optim_state_bytes": int(optim_state_bytes),
        "measured_delta_bytes": int(measured_delta),
        "final_loss": losses[-1],
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def e3():
    """8-bit Adam: blockwise int8 optimizer states vs fp32 states — convergence + memory."""
    out = {"exp": "e3_adam8bit", "env": env_block()}
    r_fp32 = _train_gpt("fp32")
    r_int8 = _train_gpt("int8")
    out["fp32"] = r_fp32
    out["int8"] = r_int8
    diff = abs(r_fp32["final_loss"] - r_int8["final_loss"])
    out["final_loss_diff"] = diff
    out["optim_mem_ratio_int8_vs_fp32"] = r_int8["optim_state_bytes"] / r_fp32["optim_state_bytes"]

    gates = {
        "g3_loss_diff_lt_0.05": bool(diff < 0.05),
        "g3_mem_ratio_near_quarter": bool(0.2 < out["optim_mem_ratio_int8_vs_fp32"] < 0.3),
    }
    out["gates"] = gates
    jdump("e3_adam8bit.json", out)
    print(json.dumps({"fp32_final": r_fp32["final_loss"], "int8_final": r_int8["final_loss"], "diff": round(diff, 4),
                      "fp32_step_ms": round(r_fp32["step_ms_median"], 1), "int8_step_ms": round(r_int8["step_ms_median"], 1),
                      "mem_fp32_MB": r_fp32["optim_state_bytes"] / 1e6, "mem_int8_MB": r_int8["optim_state_bytes"] / 1e6,
                      "mem_ratio": round(out["optim_mem_ratio_int8_vs_fp32"], 3), "gates": gates}, indent=1))
    assert gates["g3_loss_diff_lt_0.05"], "G3 failed: int8-Adam loss diverged from fp32 by >= 0.05"
    if not gates["g3_mem_ratio_near_quarter"]:
        print("[E3][HONEST] memory ratio outside [0.2, 0.3] — recorded for analysis")
    print("[E3] gates PASS")


# ------------------------------------------------------------------ E5

def e5():
    """Memory ledger: measured bytes/param for training combos + 7.5B projection."""
    set_seed()
    from quant_ops import blockwise_quant, nf4_levels

    dev = "cuda"
    out = {"exp": "e5_ledger", "env": env_block()}
    n = 4_000_000  # params per combo probe

    def alloc_delta(mk):
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        t = mk()
        torch.cuda.synchronize()
        m1 = torch.cuda.memory_allocated()
        b = m1 - m0
        del t
        return b

    f32 = torch.float32
    combos = {}

    # fp32 training: P+G+M+V all fp32
    combos["fp32_training"] = alloc_delta(lambda: [torch.empty(n, device=dev, dtype=f32) for _ in range(4)])
    # AMP mixed: P fp32 master + G fp16 + M + V fp32
    combos["amp_mixed"] = alloc_delta(lambda: [torch.empty(n, device=dev, dtype=f32)] + [torch.empty(n, device=dev, dtype=torch.float16)] + [torch.empty(n, device=dev, dtype=f32) for _ in range(2)])
    # 8-bit optimizer states (uint8 m/v + fp32 absmax per 256-block), params fp32, grads fp16
    nblocks = -(-n // 256)
    combos["int8_optim_states"] = alloc_delta(lambda: [torch.empty(n, device=dev, dtype=torch.uint8) for _ in range(2)] + [torch.empty(nblocks, device=dev, dtype=f32) for _ in range(2)])
    combos["int8_optim_full_P32_G16"] = combos["int8_optim_states"] + n * (4 + 2)
    # 4-bit NF4 params: PACKED two indices per byte (n/2 uint8) + fp16 absmax per
    # 64-block; double quantization of the absmax itself would shave ~0.03 more
    # (recorded as projection only). Packing matches bnb's two-per-byte layout.
    nb64 = -(-n // 64)
    combos["nf4_params_storage"] = alloc_delta(lambda: [torch.empty(n // 2, device=dev, dtype=torch.uint8), torch.empty(nb64, device=dev, dtype=torch.float16)])

    per_param = {k: v / n for k, v in combos.items()}
    theory = {
        "fp32_training": 16.0,
        "amp_mixed": 14.0,
        "int8_optim_full_P32_G16": 4 + 2 + 2 + 2 * 4 / 256,
        "nf4_params_storage": 0.5 + 2 / 64,
    }
    consistency = {k: float(per_param[k] / theory[k]) for k in theory}
    out["measured_bytes"] = combos
    out["bytes_per_param"] = per_param
    out["theory_bytes_per_param"] = theory
    out["consistency_ratio"] = consistency

    # 7.5B projection (formulas + substitution, no allocation obviously)
    b = 7.5e9
    proj = {
        "fp32_training_GB": per_param["fp32_training"] * b / 1e9,
        "amp_mixed_GB": per_param["amp_mixed"] * b / 1e9,
        "int8_optim_GB": per_param["int8_optim_full_P32_G16"] * b / 1e9,
        "qlora_base_nf4_GB": per_param["nf4_params_storage"] * b / 1e9,
        "qlora_total_example_GB": per_param["nf4_params_storage"] * b / 1e9 + (0.01 * b * (2 + 2 + 2.03)) / 1e9,
        "note": "qlora_total_example assumes 1% trainable LoRA params (fp16 P+G + 8bit m/v)",
    }
    out["projection_7p5B"] = proj

    gates = {"g5_consistency_within_10pct": all(0.9 <= r <= 1.1 for r in consistency.values())}
    out["gates"] = gates
    jdump("e5_ledger.json", out)
    print(json.dumps({"per_param": {k: round(v, 4) for k, v in per_param.items()},
                      "consistency": {k: round(v, 4) for k, v in consistency.items()},
                      "proj_7p5B": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in proj.items()},
                      "gates": gates}, indent=1))
    assert gates["g5_consistency_within_10pct"], "G5 failed: measured vs theory off by >10%"
    print("[E5] gates PASS")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "e0", "e1", "e2", "e3", "e5", "all"])
    args = ap.parse_args()
    if args.cmd == "prep":
        subprocess.run([sys.executable, "-X", "utf8", str(LAB / "corpus_prep.py")], check=True)
        return
    if args.cmd == "all":
        # subprocess isolation: e0's tinygrad CUDA realize attempt crashes inside
        # hcq2 and leaves the process CUDA context unusable for torch (any later
        # kernel launch fails with "invalid argument") — measured, so each
        # experiment runs in its own process.
        for cmd in ["e0", "e1", "e2", "e3", "e5"]:
            subprocess.run([sys.executable, "-X", "utf8", __file__, cmd], check=True)
        return
    if args.cmd == "e0":
        e0()
    if args.cmd == "e1":
        e1()
    if args.cmd == "e2":
        e2()
    if args.cmd == "e3":
        e3()
    if args.cmd == "e5":
        e5()


if __name__ == "__main__":
    main()
