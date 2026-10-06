"""bench_infer.py — AR005 E1..E5 experiment driver.

Timing discipline (inherited from AR002-004):
  session-level 2s FP32 GEMM burn-in -> per-config 3x warmup -> adaptive reps
  (3..10) -> CUDA Event median. All data device-resident.
Every experiment ends with hard gates (assert) and a single JSON save.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import charlm
import quant_gguf as qg

LAB = Path(__file__).parent
RES = LAB / "results"
RES.mkdir(exist_ok=True)
DEV = "cuda"
GGUF_PATHS = {q: RES / f"charlm_{q}.gguf" for q in ["f16", "q8_0", "q4_K"]}
HBM_GBPS = 448.0  # Quadro RTX 5000 spec


def save_json(name, obj):
    (RES / name).write_text(json.dumps(obj, indent=1, default=float))
    print(f"[saved] {name}")


def burn(seconds=2.0):
    a = torch.randn(1024, 1024, device=DEV)
    t0 = time.time()
    while time.time() - t0 < seconds:
        a = a @ a * 0.001
    torch.cuda.synchronize()


def bench(fn, warmup=3, reps_min=3, reps_max=10, target_s=1.0):
    """Median time in ms via CUDA events, adaptive reps."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    fn()
    e1.record()
    torch.cuda.synchronize()
    t1 = e0.elapsed_time(e1) / 1e3
    reps = max(reps_min, min(reps_max, int(target_s / max(t1, 1e-4))))
    ts = []
    for _ in range(reps):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return float(np.median(ts)), reps


# ================================================================ env
def exp_env(model, tok, meta):
    p = torch.cuda.get_device_properties(0)
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": p.name,
        "sm": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "vram_gb": p.total_memory / 2**30,
        "hbm_spec_gbps": HBM_GBPS,
        "corpus_chars": meta.get("corpus_chars"),
        "vocab": tok.vocab_size,
        "n_params": meta["n_params"],
        "train_time_s": meta["train_time_s"],
        "final_train_loss": meta["final_loss"],
        "gguf_py": str((LAB.parent / "llama.cpp" / "gguf-py")),
    }
    save_json("env.json", env)
    return env


# ================================================================ E1
def exp_e1(model):
    """Quantization roundtrip on real trained weights."""
    out = {"exp": "e1_quant_roundtrip"}
    w2d = {k: v for k, v in model.state_dict().items() if v.dim() == 2}
    w1d = {k: v for k, v in model.state_dict().items() if v.dim() == 1}
    res = {}
    hist = {}
    elems_per_block = {34: 32, 144: 256}
    for qname, qfn, dqfn, bs in [("q8_0", qg.quantize_q8_0, qg.dequantize_q8_0, 34),
                                  ("q4_K", qg.quantize_q4_K, qg.dequantize_q4_K, 144)]:
        per, allerr = {}, []
        for k, w in w2d.items():
            wc = w.detach().float().cpu()
            b = qfn(wc)
            # block-size accounting gate: total bytes == n_blocks * bs, all weights covered
            assert b.numel() == (wc.numel() // elems_per_block[bs]) * bs, k
            x = dqfn(b, wc.shape)
            err = (x - wc)
            per[k] = {
                "rel": float(err.norm() / wc.norm()),
                "maxabs_norm": float(err.abs().max() / wc.abs().max()),
                "n_blocks": int(b.numel() // bs),
            }
            allerr.append((err / (wc.abs().max() + 1e-12)).flatten())
        e = torch.cat(allerr)
        res[qname] = {
            "per_tensor": per,
            "rel_overall": float(
                np.sqrt(sum((p["rel"] * float(w2d[k].norm())) ** 2 for k, p in per.items()))
                / float(sum(float(w.norm()) for w in w2d.values()))),
            "maxabs_norm_overall": float(max(p["maxabs_norm"] for p in per.values())),
            "bpw": {"q8_0": 34 / 32, "q4_K": 144 / 256}[qname],
        }
        hist[qname] = torch.histc(e, bins=32).tolist()
        hist[qname + "_edges"] = torch.linspace(e.min(), e.max(), 33).tolist()
    # gates
    assert res["q8_0"]["maxabs_norm_overall"] <= 1 / 128, "Q8_0 half-step gate"
    assert res["q4_K"]["rel_overall"] <= 0.25, "Q4_K rel gate"
    ls = torch.randint(0, 64, (64, 8))
    lm = torch.randint(0, 64, (64, 8))
    ls2, lm2 = qg._unpack_scales6(qg._pack_scales6(ls, lm))
    assert torch.equal(ls, ls2) and torch.equal(lm, lm2), "6-bit pack roundtrip"
    out.update(res)
    out["hist"] = hist
    out["norm_tensors_unquantized"] = list(w1d.keys())
    out["gates"] = "q8_0 maxabs<=1/128 PASS; q4_K rel<=0.25 PASS; 6bit pack exact PASS"
    save_json("e1.json", out)
    return out


# ================================================================ E2
def exp_e2(model, tok, prompt="The model "):
    """GGUF export/import roundtrip for f16/q8_0/q4_K."""
    greedy = lambda lg: lg.argmax(-1)
    pid = tok.encode(prompt)
    out = {"exp": "e2_gguf_roundtrip", "prompt": prompt}
    src = model.state_dict()
    # export all three
    for quant, name in [(None, "f16"), ("q8_0", "q8_0"), ("q4_K", "q4_K")]:
        if not GGUF_PATHS[name].exists():
            qg.save_gguf(model, tok, GGUF_PATHS[name], quant=quant)
    # reference generation (fp32 source model)
    ref = model.generate(pid, 60, greedy, use_cache=True)[0]
    # teacher-forced top-1 agreement on val text
    val_text = _val_text(tok)
    agree_windows = 10
    for name in ["f16", "q8_0", "q4_K"]:
        m2, t2 = qg.model_from_gguf(GGUF_PATHS[name], dtype=torch.float32)
        sd2 = m2.state_dict()
        per = {}
        worst_rel = 0.0
        for k, w in src.items():
            d = (sd2[k].float() - w.float())
            per[k] = float(d.norm() / w.norm())
            worst_rel = max(worst_rel, per[k])
        gen = m2.generate(pid, 60, greedy, use_cache=True)[0]
        agree = float((gen == ref).float().mean())
        top1 = _top1_agreement(model, m2, val_text, agree_windows)
        out[name] = {
            "file_kb": os.path.getsize(GGUF_PATHS[name]) / 1024,
            "worst_tensor_rel": worst_rel,
            "greedy_gen_agreement_vs_fp16": agree,
            "gen_text": t2.decode(gen.tolist())[:80],
            "teacher_forced_top1_agreement": top1,
            "per_tensor_rel": per,
        }
        m2 = None
        torch.cuda.empty_cache()
    # metadata roundtrip
    import gguf
    r = gguf.GGUFReader(GGUF_PATHS["q4_K"])
    meta = {}
    for k in ["llama.embedding_length", "llama.block_count", "llama.attention.head_count",
              "llama.attention.head_count_kv", "llama.context_length"]:
        meta[k] = r.fields[k].contents()
    ntok = len(list(r.fields["tokenizer.ggml.tokens"].contents()))
    assert meta["llama.embedding_length"] == 256 and meta["llama.block_count"] == 4
    assert ntok == tok.vocab_size, "tokenizer roundtrip"
    out["metadata_roundtrip"] = {k: int(v) for k, v in meta.items()}
    out["tokenizer_roundtrip_vocab"] = ntok
    # gates: f16 exact
    assert out["f16"]["worst_tensor_rel"] == 0.0, "f16 exact roundtrip"
    assert out["q8_0"]["worst_tensor_rel"] <= 0.02, "q8_0 tensor gate"
    assert out["q4_K"]["worst_tensor_rel"] <= 0.25, "q4_K tensor gate"
    assert out["q8_0"]["teacher_forced_top1_agreement"] >= 0.95, "q8_0 top1 agreement gate"
    assert out["q4_K"]["teacher_forced_top1_agreement"] >= 0.50, "q4_K structure gate"
    out["gates"] = ("f16 exact PASS; q8_0 rel<=0.02 & top1>=0.95 PASS; "
                    "q4_K rel<=0.25 & top1>=0.50 PASS; tokenizer/metadata exact PASS")
    save_json("e2.json", out)
    return out


def _val_text(tok):
    ck = torch.load(RES / "checkpoint.pt", map_location="cpu", weights_only=False)
    return ck["val_text"]


def _top1_agreement(m_ref, m_test, val_text, windows, ctx=128):
    """Fraction of positions where test model argmax == reference argmax."""
    return _top1_agreement_impl(m_ref, m_test, val_text, windows, ctx)


def _top1_agreement_impl(m_ref, m_test, val_text, windows, ctx):
    # encode with the REFERENCE tokenizer (char ids are shared via vocab order)
    chars = sorted(set(val_text))
    # both models share the same char<->id map because we exported it; encode via
    # the reference tokenizer stored on the checkpoint
    tok = _load_tok()
    ids = torch.tensor(tok.encode(val_text[: windows * ctx]), device=DEV)
    agree = []
    with torch.no_grad():
        for i in range(windows):
            x = ids[i * ctx:(i + 1) * ctx][None]
            a = m_ref(x)[0].argmax(-1)
            b = m_test(x)[0].argmax(-1)
            agree.append((a == b).float().mean().item())
    return float(np.mean(agree))


_TOK = None
def _load_tok():
    global _TOK
    if _TOK is None:
        ck = torch.load(RES / "checkpoint.pt", map_location="cpu", weights_only=False)
        _TOK = charlm.CharTokenizer("".join(ck["chars"]))
    return _TOK


# ================================================================ E3
def exp_e3(model, tok):
    """Quantized inference quality (ppl) + speed (dequant cost decomposition)."""
    out = {"exp": "e3_quant_inference"}
    val_text = _val_text(tok)
    tokR = _load_tok()
    ids = tokR.encode(val_text[:20 * 128])
    xs = torch.tensor([ids[i * 128:(i + 1) * 128] for i in range(20)], device=DEV)

    # ---- quality: perplexity with dequantized weights, fp32 compute
    def ppl(m):
        with torch.no_grad():
            nll = []
            for i in range(xs.shape[0]):
                lg = m(xs[i:i + 1])[0]
                nll.append(F.cross_entropy(lg[:-1], xs[i, 1:]).item())
        return float(np.exp(np.mean(nll)))

    weights_variants = {}
    m16 = charlm.load_model(dtype=torch.float32)[0]
    weights_variants["fp16"] = (m16, ppl(m16))
    for name in ["q8_0", "q4_K"]:
        m2, _ = qg.model_from_gguf(GGUF_PATHS[name], dtype=torch.float32)
        weights_variants[name] = (m2, ppl(m2))
    for k, (m, p) in weights_variants.items():
        out[f"ppl_{k}"] = p
    out["ppl_delta_q8_0"] = out["ppl_q8_0"] - out["ppl_fp16"]
    out["ppl_delta_q4_K"] = out["ppl_q4_K"] - out["ppl_fp16"]
    assert out["ppl_delta_q8_0"] <= 0.05, "Q8_0 dppl gate"

    # ---- dequant kernel: naive per-block vs vectorized (same data, GPU)
    burn()
    synth = torch.randn(16384 * 256) * 0.05   # 16384 super-blocks of 256 elems
    b4 = qg.quantize_q4_K(synth.view(-1, 256)).view(-1, 144).to(DEV)

    def deq4_naive():
        outs = []
        for i in range(4096):  # subset for wall-time sanity; same subset both impls
            b = b4[i]
            d = b[0:2].view(torch.float16)[0].float()
            dmin = b[2:4].view(torch.float16)[0].float()
            ls, lm = qg._unpack_scales6(b[4:16][None])
            qs = b[16:].view(4, 32)
            lo = (qs & 0xF).float()
            hi = (qs >> 4).float()
            lsf = ls.float().view(8); lmf = lm.float().view(8)
            x = torch.empty(8, 32, device=DEV)
            x[0::2] = d * lsf[0::2, None] * lo - dmin * lmf[0::2, None]
            x[1::2] = d * lsf[1::2, None] * hi - dmin * lmf[1::2, None]
            outs.append(x.view(-1))
        return torch.stack(outs)

    def deq4_vec():
        return qg.dequantize_q4_K(b4[:4096], (4096, 256))

    t_naive, _ = bench(deq4_naive, warmup=1, reps_min=3, reps_max=3)
    t_vec, _ = bench(deq4_vec)
    n_elem = 4096 * 256
    out["dequant_q4_K_4096blocks"] = {
        "naive_ms": t_naive, "vec_ms": t_vec,
        "speedup": t_naive / t_vec,
        "vec_gbps_achieved": (4096 * 144 + n_elem * 4) / (t_vec / 1e3) / 1e9,
        "note": "4096 super-blocks = 1.05M weights; launch-bound at this size",
    }
    assert out["dequant_q4_K_4096blocks"]["speedup"] >= 10, "vector speedup gate"

    # ---- large-scale dequant: torch multi-op (launch-bound) vs triton fused
    import kernels_dequant as kd
    big = torch.randn(4096, 4096) * 0.05
    burn(0.5)
    for qname, qfn, dqfn, tfn, bs in [("q8_0", qg.quantize_q8_0, qg.dequantize_q8_0, kd.dequant_q8_0_triton, 34),
                                      ("q4_K", qg.quantize_q4_K, qg.dequantize_q4_K, kd.dequant_q4_k_triton, 144)]:
        bb = qfn(big).view(-1, bs).to(DEV)
        bytes_in = bb.numel()
        # correctness of triton path on this tensor
        ref = dqfn(bb[:512], (512 * (32 if bs == 34 else 256),)).flatten()
        got = tfn(bb[:512]).float()
        assert (got - ref).abs().max() < 1e-4, f"{qname} triton large gate"
        for tag, fn, mult in [("torch_vec_fp32", lambda: dqfn(bb, big.shape), 4),
                              ("torch_vec_fp16", lambda: dqfn(bb, big.shape).to(torch.float16), 2),
                              ("triton_fp32", lambda: tfn(bb, out_dtype=torch.float32), 4),
                              ("triton_fp16", lambda: tfn(bb, out_dtype=torch.float16), 2)]:
            # sub-ms kernels under WDDM: 20 reps, record min AND median; gate on
            # min (noise only adds latency; min = least-interfered estimate)
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            ts = []
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            for _ in range(20):
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            t_med, t_min = float(np.median(ts)), float(np.min(ts))
            traffic = bytes_in + big.numel() * mult
            out[f"dequant_{qname}_{tag}"] = {
                "ms_median": t_med, "ms_min": t_min,
                "gbps_median": traffic / (t_med / 1e3) / 1e9,
                "gbps_min": traffic / (t_min / 1e3) / 1e9,
                "pct_hbm_min": traffic / (t_min / 1e3) / 1e9 / HBM_GBPS * 100,
            }
            print(f"  {qname} {tag:16s} min {t_min:7.3f} ms -> {out[f'dequant_{qname}_{tag}']['gbps_min']:7.1f} GB/s (med {t_med:6.3f})")
        del bb
        torch.cuda.empty_cache()
    out["dequant_finding"] = ("torch multi-op dequant is launch-bound under WDDM (dozens of "
                              "kernel launches dominate); the fused triton kernel approaches "
                              "memory bandwidth - the same reason llama.cpp fuses dequant into "
                              "mmvq/mmq GEMM kernels; sub-ms kernel gates use min-of-20")
    assert out["dequant_q4_K_triton_fp32"]["gbps_min"] >= 200, "fused dequant bandwidth gate"

    # ---- decode speed: fp16 vs on-the-fly dequant vs pre-dequant (batch 1)
    m16h = charlm.load_model(dtype=torch.float16)[0]
    tokr = _load_tok()
    pid = tokr.encode("The model ")
    greedy = lambda lg: lg.argmax(-1)

    def gen_tokens_s(m):
        def run():
            m.generate(pid, 64, greedy, use_cache=True)
        t, _ = bench(run, warmup=2, reps_min=3, reps_max=5, target_s=0.5)
        return 64 / (t / 1e3)

    out["decode_fp16_tokens_per_s"] = gen_tokens_s(m16h)

    # on-the-fly q4_K: dequant all weights every step, then fp16 forward
    blocks_gpu = {k: qg.quantize_q4_K(v).view(-1, 144).to(DEV)
                  for k, v in m16h.state_dict().items() if v.dim() == 2}
    shapes = {k: tuple(v.shape) for k, v in m16h.state_dict().items()}
    m4 = charlm.load_model(dtype=torch.float16)[0]

    def dequant_all():
        sd = m4.state_dict()
        for k, bb in blocks_gpu.items():
            sd[k].data = qg.dequantize_q4_K(bb, shapes[k]).view(shapes[k]).to(torch.float16)

    t_deq, _ = bench(dequant_all, warmup=2, reps_min=3, reps_max=5, target_s=0.5)
    out["dequant_all_weights_ms"] = t_deq
    dequant_all()
    out["decode_q4K_predequant_tokens_per_s"] = gen_tokens_s(m4)
    # compose on-the-fly estimate and verify with one real run
    out["decode_q4K_onthefly_est_tokens_per_s"] = 64 / (64 / out["decode_q4K_predequant_tokens_per_s"] + 64 * t_deq / 1e3)
    # triton fused per-tensor dequant: kernel efficient but still one launch per
    # weight tensor -> launch-bound at 33-tensor model scale (the mmvq lesson:
    # production fuses dequant INTO the GEMM, never materializing fp16 weights)
    def dequant_all_triton():
        sd = m4.state_dict()
        for k, bb in blocks_gpu.items():
            shp = shapes[k]
            sd[k].data = kd.dequant_q4_k_triton(bb.view(-1, 144), out_dtype=torch.float16).view(shp)
    t_deq_t, _ = bench(dequant_all_triton, warmup=2, reps_min=3, reps_max=5, target_s=0.5)
    out["dequant_all_weights_triton_ms"] = t_deq_t
    out["decode_q4K_onthefly_triton_est_tokens_per_s"] = 64 / (
        64 / out["decode_q4K_predequant_tokens_per_s"] + 64 * t_deq_t / 1e3)
    dequant_all_triton()
    with torch.no_grad():
        t0 = time.time()
        for _ in range(3):
            dequant_all()
        torch.cuda.synchronize()
        out["dequant_all_weights_ms_check"] = (time.time() - t0) / 3 * 1e3

    save_json("e3.json", out)
    return out


# ================================================================ E4
def exp_e4(model, tok):
    """Decode throughput vs batch + KV cache formula vs measured + prefill/decode."""
    out = {"exp": "e4_decode_kv"}
    burn()
    m = charlm.load_model(dtype=torch.float16)[0]
    tokr = _load_tok()
    pid = tokr.encode("The model ")
    greedy = lambda lg: lg.argmax(-1)

    # ---- tokens/s vs batch
    curve = []
    for B in [1, 2, 4, 8, 16, 32, 64]:
        def run():
            m.generate(pid, 48, greedy, batch=B, use_cache=True)
        t, reps = bench(run, warmup=2, reps_min=3, reps_max=5, target_s=0.6)
        curve.append({"batch": B, "time_ms": t, "reps": reps,
                      "tokens_per_s_total": B * 48 / (t / 1e3),
                      "tokens_per_s_per_seq": 48 / (t / 1e3)})
        print(f"  B={B:3d}  {t:8.2f} ms  {curve[-1]['tokens_per_s_total']:9.0f} t/s total")
    out["decode_vs_batch"] = curve
    # weight-streaming floor for context: fp16 weights = 6.4MB
    out["weight_bytes_fp16"] = sum(v.numel() * 2 for v in m.state_dict().values())
    out["single_seq_floor_tokens_per_s"] = HBM_GBPS * 1e9 / out["weight_bytes_fp16"]

    # ---- KV cache: formula vs measured (fp16 cache)
    kv = []
    for B in [1, 16]:
        for ctx_max in [128, 256, 512, 1024]:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            c = charlm.KVCache(B, 4, 2, 64, ctx_max, DEV, torch.float16)
            torch.cuda.synchronize()
            meas = torch.cuda.memory_allocated() - base
            formula = 2 * 4 * 2 * ctx_max * 64 * 2 * B  # 2(KV) x layers x kvh x seq x hd x 2B
            kv.append({"batch": B, "ctx": ctx_max, "measured_bytes": int(meas),
                       "formula_bytes": formula, "err_pct": abs(meas - formula) / formula * 100})
            del c
    out["kv_cache_check"] = kv
    assert max(k["err_pct"] for k in kv) <= 5.0, "KV formula gate"

    # ---- prefill vs decode (same 256 tokens), fp16
    ctx_ids = torch.randint(0, tokr.vocab_size, (1, 256), device=DEV)
    with torch.no_grad():
        def prefill():
            c = charlm.KVCache(1, 4, 2, 64, 256, DEV, torch.float16)
            m(ctx_ids, c)
        t_pre, _ = bench(prefill, warmup=3, reps_min=5, reps_max=10)
        def decode_one():
            c = charlm.KVCache(1, 4, 2, 64, 256, DEV, torch.float16)
            m(ctx_ids[:, :1], c)
            for i in range(255):
                m(ctx_ids[:, i + 1:i + 2], c)
        t_dec, _ = bench(decode_one, warmup=2, reps_min=3, reps_max=4)
    n_par = sum(p.numel() for p in m.parameters())
    out["prefill_vs_decode"] = {
        "n_tokens": 256,
        "prefill_ms": t_pre, "decode_ms": t_dec, "ratio_decode_over_prefill": t_dec / t_pre,
        "prefill_us_per_token": t_pre / 256 * 1e3, "decode_us_per_token": t_dec / 256 * 1e3,
        "flops_per_token": 2 * n_par,
        "prefill_tflops": 2 * n_par * 256 / (t_pre / 1e3) / 1e12,
        "decode_tflops": 2 * n_par * 256 / (t_dec / 1e3) / 1e12,
        "prefill_arith_intensity": 256, "decode_arith_intensity": 1,
    }
    save_json("e4.json", out)
    return out


# ================================================================ E5
def exp_e5(model, tok):
    """Sampling strategies: correctness gates + behavior + timing."""
    out = {"exp": "e5_sampling"}
    from scipy.stats import chi2 as chi2_dist

    # ---- implementations
    def sample_greedy(logits):
        return logits.argmax(-1)

    def sample_top_k(logits, k):
        v, ix = torch.topk(logits, k, dim=-1)
        pick = torch.multinomial(v.softmax(-1), 1)
        return ix.gather(-1, pick)

    def sample_top_p(logits, p):
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1)
        cum = probs.cumsum(-1)
        keep = (cum - probs) < p          # prefix property: boundary token kept
        mask = keep.gather(-1, torch.argsort(sorted_idx, dim=-1))
        masked = logits.masked_fill(~mask, float("-inf"))
        return torch.multinomial(masked.softmax(-1), 1).squeeze(-1)

    # ---- correctness on constructed distribution
    base = torch.log(torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05], device=DEV))
    lg = torch.cat([base, torch.full((114,), -20.0, device=DEV)])[None]
    for p, expect_n in [(0.6, 2), (0.9, 4)]:
        torch.manual_seed(0)
        picks = torch.stack([sample_top_p(lg, p) for _ in range(2000)]).flatten()
        kept = sorted(set(picks.tolist()))
        assert max(kept) < expect_n, f"top-p {p} kept token beyond prefix"
        assert len(kept) == expect_n, f"top-p {p} expected {expect_n} kept, got {kept}"
    out["topp_prefix_gate"] = "PASS (p=0.6 -> 2 tokens, p=0.9 -> 4 tokens)"

    # ---- distribution consistency vs torch.multinomial (chi-square, N=10k)
    torch.manual_seed(1)
    logits = torch.randn(119, device=DEV) * 3
    probs_ref = logits.softmax(-1)
    N = 10000
    a = torch.multinomial(probs_ref, N, replacement=True)
    b = torch.stack([sample_top_p(logits[None], 1.0) for _ in range(N)]).flatten()
    # integer-aligned bins so observed and expected use identical slices
    edges = np.array(list(range(0, 120, 6)) + [119])
    obs_a = np.histogram(a.cpu().numpy(), bins=edges)[0]
    obs_b = np.histogram(b.cpu().numpy(), bins=edges)[0]
    exp_a = np.array([float(probs_ref[edges[i]:edges[i + 1]].sum()) * N
                      for i in range(len(edges) - 1)])
    valid = exp_a > 5
    chi_a = float(((obs_a[valid] - exp_a[valid]) ** 2 / exp_a[valid]).sum())
    chi_b = float(((obs_b[valid] - exp_a[valid]) ** 2 / exp_a[valid]).sum())
    dof = int(valid.sum()) - 1
    out["chi2_multinomial"] = {"stat": chi_a, "dof": dof, "p_value": float(chi2_dist.sf(chi_a, dof))}
    out["chi2_topp_p1"] = {"stat": chi_b, "dof": dof, "p_value": float(chi2_dist.sf(chi_b, dof))}
    assert out["chi2_multinomial"]["p_value"] > 0.01 and out["chi2_topp_p1"]["p_value"] > 0.01

    # ---- behavior on real model logits: temperature -> top-p cutoff set size
    val_text = _val_text(tok)
    tokr = _load_tok()
    m32 = charlm.load_model(dtype=torch.float32)[0]
    ids = torch.tensor([tokr.encode(val_text[:4096])], device=DEV)
    with torch.no_grad():
        logits_bank = []
        for i in range(20):
            x = ids[:, i * 196:i * 196 + 128]
            logits_bank.append(m32(x)[0, -1])
        logits_bank = torch.stack(logits_bank)   # (20, 119)
    out["real_logits_top_prob_mean"] = float(logits_bank.softmax(-1).max(-1).values.mean())
    beh = []
    for T in [0.5, 0.7, 1.0, 1.5]:
        for p in [0.9]:
            scaled = logits_bank / T
            sorted_p = scaled.softmax(-1).sort(-1, descending=True).values
            cum = sorted_p.cumsum(-1)
            n_keep = (cum < p).sum(-1).float() + 1
            beh.append({"T": T, "p": p, "mean_cutoff_set": float(n_keep.mean()),
                        "std": float(n_keep.std()), "min": int(n_keep.min()), "max": int(n_keep.max())})
    out["temperature_vs_topp_cutoff"] = beh

    # top-k vs top-p tail retention on the same logits (T=1)
    p_sorted = logits_bank.softmax(-1).sort(-1, descending=True).values
    cum = p_sorted.cumsum(-1)
    nk_p = (cum < 0.9).sum(-1).float() + 1
    tail_p = torch.stack([p_sorted[i, int(nk_p[i]):].sum() for i in range(20)])
    out["topp_vs_topk"] = {
        "topp09_mean_set": float(nk_p.mean()),
        "topk8_kept_mass": float(p_sorted[:, :8].sum(-1).mean()),
        "topp09_tail_mass_dropped": float(1 - p_sorted[:, :int(nk_p.mean())].sum(-1).mean()),
    }

    # ---- timing: sampling kernel cost at char vocab vs LLM vocab (32k)
    burn()
    for V, tag in [(119, "char_vocab"), (32000, "llm_vocab")]:
        lx = torch.randn(32, V, device=DEV)
        t_g, _ = bench(lambda: sample_greedy(lx))
        t_k, _ = bench(lambda: sample_top_k(lx, 40))
        t_p, _ = bench(lambda: sample_top_p(lx, 0.9))
        out[f"sampling_ms_{tag}"] = {"greedy": t_g, "topk40": t_k, "topp0.9": t_p,
                                     "batch": 32}
    save_json("e5.json", out)
    return out


# ================================================================ main
def main(which="all"):
    torch.manual_seed(0)
    model, tok = charlm.load_model(dtype=torch.float32)
    meta = json.loads((RES / "e0_train.json").read_text())
    meta["corpus_chars"] = len(charlm.load_corpus())
    burn()
    exps = {"e1": lambda: exp_e1(model),
            "e2": lambda: exp_e2(model, tok),
            "e3": lambda: exp_e3(model, tok),
            "e4": lambda: exp_e4(model, tok),
            "e5": lambda: exp_e5(model, tok)}
    if which == "all":
        exp_env(model, tok, meta)
        for k in ["e1", "e2", "e3", "e4", "e5"]:
            print(f"=== {k} ===")
            exps[k]()
    else:
        exps[which]()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
