"""charlm.py — tiny llama-architecture char-level LM (RoPE + GQA + SwiGLU + RMSNorm).

AR005 F1 deliverable. Trained on llama.cpp docs corpus on GPU (~1 min).
Supports: full forward, incremental decode with paged-free simple KV cache,
batched generation, fp32 (correctness) / fp16 (speed) execution.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LAB = Path(__file__).parent
CORPUS_DIR = LAB.parent / "llama.cpp" / "docs"


# ---------------------------------------------------------------- tokenizer
class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.chars)


def load_corpus():
    texts = []
    for p in sorted(CORPUS_DIR.glob("**/*.md")):
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(texts)


# ---------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_cache(seq_len: int, head_dim: int, device, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    ang = torch.outer(t, freqs)  # (T, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); cos/sin: (T, hd/2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = y1
    out[..., 1::2] = y2
    return out


class Attention(nn.Module):
    def __init__(self, d: int, n_qh: int, n_kvh: int):
        super().__init__()
        self.n_qh, self.n_kvh = n_qh, n_kvh
        self.hd = d // n_qh
        self.q_proj = nn.Linear(d, n_qh * self.hd, bias=False)
        self.k_proj = nn.Linear(d, n_kvh * self.hd, bias=False)
        self.v_proj = nn.Linear(d, n_kvh * self.hd, bias=False)
        self.o_proj = nn.Linear(n_qh * self.hd, d, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_qh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kvh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kvh, self.hd).transpose(1, 2)

        pos = 0 if cache is None else cache.pos
        cos, sin = rope_cache(pos + T, self.hd, x.device)
        cos, sin = cos[pos : pos + T], sin[pos : pos + T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            cache.k[self.layer][:, :, pos : pos + T] = k
            cache.v[self.layer][:, :, pos : pos + T] = v
            k = cache.k[self.layer][:, :, : pos + T]
            v = cache.v[self.layer][:, :, : pos + T]

        rep = self.n_qh // self.n_kvh
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None or T > 1))
        # note: when writing into cache with T>1 (prefill) the causal mask over the
        # first T positions is still correct because cache was empty before pos.
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, d: int, n_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, n_ff, bias=False)
        self.up_proj = nn.Linear(d, n_ff, bias=False)
        self.down_proj = nn.Linear(n_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d, n_qh, n_kvh, n_ff):
        super().__init__()
        self.attn_norm = RMSNorm(d)
        self.attn = Attention(d, n_qh, n_kvh)
        self.ffn_norm = RMSNorm(d)
        self.mlp = MLP(d, n_ff)

    def forward(self, x, cache=None):
        x = x + self.attn(self.attn_norm(x), cache)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class KVCache:
    """Preallocated (B, n_kvh, ctx_max, hd) per layer."""

    def __init__(self, b, n_layers, n_kvh, hd, ctx_max, device, dtype=torch.float32):
        self.k = [torch.zeros(b, n_kvh, ctx_max, hd, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(b, n_kvh, ctx_max, hd, device=device, dtype=dtype) for _ in range(n_layers)]
        self.n = 0      # sequence length seen so far (all layers share this)
        self.pos = 0    # start position of the current forward pass (read once)

    def bytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


class CharLM(nn.Module):
    def __init__(self, vocab, d=256, n_layers=4, n_qh=4, n_kvh=2, n_ff=768, ctx=256):
        super().__init__()
        self.ctx = ctx
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_qh, n_kvh, n_ff) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        for i, blk in enumerate(self.blocks):
            blk.attn.layer = i

    def forward(self, ids, cache=None):
        if cache is not None:
            cache.pos = cache.n  # fixed start position for every layer in this pass
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x, cache)
        if cache is not None:
            cache.n = cache.pos + ids.shape[1]
        return self.lm_head(self.norm(x))

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, prompt_ids, n_new, sampler, batch=None, use_cache=True):
        """Returns list of generated id sequences (prompt excluded)."""
        dev = next(self.parameters()).device
        if batch is None:
            ids = torch.tensor([prompt_ids], device=dev)
        else:
            ids = torch.tensor([prompt_ids] * batch, device=dev)
        B = ids.shape[0]
        out = []
        if use_cache:
            cache = KVCache(B, len(self.blocks), self.blocks[0].attn.n_kvh,
                            self.blocks[0].attn.hd, self.ctx, dev, next(self.parameters()).dtype)
            logits = self(ids, cache)  # prefill
            nxt = sampler(logits[:, -1])
            out.append(nxt)
            for _ in range(n_new - 1):
                logits = self(nxt[:, None], cache)
                nxt = sampler(logits[:, -1])
                out.append(nxt)
        else:
            cur = ids
            for _ in range(n_new):
                logits = self(cur)
                nxt = sampler(logits[:, -1])
                out.append(nxt)
                cur = torch.cat([cur, nxt[:, None]], dim=1)
        return torch.stack(out, dim=1)  # (B, n_new)


# ---------------------------------------------------------------- training
def train(steps=3000, batch=64, ctx=128, lr=3e-4, seed=0, out_dir=LAB / "results"):
    torch.manual_seed(seed)
    dev = "cuda"
    text = load_corpus()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = max(1, int(len(data) * 0.05))
    val_data = data[-n_val:]
    tr_data = data[:-n_val]
    print(f"corpus: {len(text)} chars, vocab {tok.vocab_size}, train {len(tr_data)} val {len(val_data)}")

    model = CharLM(tok.vocab_size).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.1)
    losses = []
    val_curve = {"steps": [], "vals": []}
    t0 = time.time()
    for step in range(steps):
        ix = torch.randint(0, len(tr_data) - ctx - 1, (batch,))
        offs = torch.arange(ctx)
        x = tr_data[ix[:, None] + offs].to(dev)
        y = tr_data[ix[:, None] + offs + 1].to(dev)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 300 == 0 or step == steps - 1:
            with torch.no_grad():
                model.eval()
                vl = []
                for i in range(0, min(len(val_data) - ctx - 1, 4096), ctx):
                    vx = val_data[i : i + ctx].to(dev)
                    vy = val_data[i + 1 : i + ctx + 1].to(dev)
                    vl.append(F.cross_entropy(model(vx[None])[0], vy).item())
                model.train()
            val_curve["steps"].append(step)
            val_curve["vals"].append(sum(vl) / len(vl))
            print(f"step {step}: loss {loss.item():.3f} val {val_curve['vals'][-1]:.3f} ({time.time()-t0:.1f}s)")

    out_dir.mkdir(exist_ok=True)
    torch.save({
        "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
        "config": {"vocab": tok.vocab_size, "d": 256, "n_layers": 4, "n_qh": 4,
                    "n_kvh": 2, "n_ff": 768, "ctx": 256},
        "chars": tok.chars,
        "val_text": text[-n_val:],
    }, out_dir / "checkpoint.pt")
    meta = {"steps": steps, "batch": batch, "ctx": ctx, "lr": lr, "seed": seed,
            "n_params": n_params, "train_time_s": time.time() - t0,
            "final_loss": losses[-1], "loss_curve": losses[::10],
            "val_curve": val_curve}
    (out_dir / "e0_train.json").write_text(json.dumps(meta))
    return model, tok, meta


def load_model(path=LAB / "results" / "checkpoint.pt", dev="cuda", dtype=torch.float32):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = CharLM(cfg["vocab"], cfg["d"], cfg["n_layers"], cfg["n_qh"], cfg["n_kvh"], cfg["n_ff"], cfg["ctx"])
    model.load_state_dict({k: v.to(dtype) for k, v in ck["state_dict"].items()})
    model.to(dev).to(dtype).eval()
    tok = CharTokenizer("".join(ck["chars"]))
    return model, tok


# llama.cpp tensor naming for GGUF export
def llama_names(state_dict):
    m = {}
    for k in state_dict:
        if k == "embed.weight":
            m[k] = "token_embd.weight"
        elif k == "norm.weight":
            m[k] = "output_norm.weight"
        elif k == "lm_head.weight":
            m[k] = "output.weight"
        else:
            m[k] = k.replace("blocks.", "blk.").replace("attn.q_proj", "attn_q") \
                    .replace("attn.k_proj", "attn_k").replace("attn.v_proj", "attn_v") \
                    .replace("attn.o_proj", "attn_output").replace("attn_norm", "attn_norm") \
                    .replace("mlp.gate_proj", "ffn_gate").replace("mlp.up_proj", "ffn_up") \
                    .replace("mlp.down_proj", "ffn_down").replace("ffn_norm", "ffn_norm")
    return m


# ------------------------------------------------------------ cache gate (F1)
@torch.no_grad()
def verify_cache(prompt="The llama", n_new=48, batch=4, tol=1e-4,
                 out_path=LAB / "results" / "cache_check.json"):
    """F1 acceptance gate: incremental decode (KV cache) must match full recompute.

    Compares per-step logits (cache path vs no-cache recompute) and greedy id
    sequences; also checks fixed-seed determinism and batch replication.
    Writes results/cache_check.json and asserts the gate.
    """
    model, tok = load_model(dtype=torch.float32)
    dev = next(model.parameters()).device
    prompt_ids = tok.encode(prompt)
    ids = torch.tensor([prompt_ids] * batch, device=dev)
    attn = model.blocks[0].attn
    cache = KVCache(batch, len(model.blocks), attn.n_kvh, attn.hd, model.ctx, dev, torch.float32)

    logits_c = model(ids, cache)                     # prefill via cache
    logits_f = model(ids)                            # full recompute
    prefill_maxdiff = (logits_c - logits_f).abs().max().item()

    cur, decode_maxdiff, ids_equal = ids, 0.0, True
    for _ in range(n_new - 1):
        nxt = logits_c[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
        logits_c = model(nxt, cache)                 # incremental step
        logits_f = model(cur)                        # full recompute
        decode_maxdiff = max(decode_maxdiff, (logits_c - logits_f[:, -1:]).abs().max().item())
        ids_equal = ids_equal and bool(torch.equal(logits_c[:, -1].argmax(-1), logits_f[:, -1].argmax(-1)))

    def greedy(seq):
        torch.manual_seed(0)
        return model.generate(seq, n_new, lambda lg: lg.argmax(-1), batch=batch, use_cache=True)
    determinism = bool(torch.equal(greedy(prompt_ids), greedy(prompt_ids)))

    rows_equal = True  # same prompt across batch rows must give identical outputs
    g = greedy(prompt_ids)
    rows_equal = all(torch.equal(g[0], g[i]) for i in range(1, batch))

    res = {"exp": "cache_consistency_gate", "prompt": prompt, "n_new": n_new, "batch": batch,
           "prefill_maxdiff": prefill_maxdiff, "decode_maxdiff": decode_maxdiff,
           "greedy_ids_equal_cache_vs_full": ids_equal, "fixed_seed_deterministic": determinism,
           "batch_rows_identical": rows_equal, "gate": {"maxdiff_tol": tol},
           "pass": prefill_maxdiff <= tol and decode_maxdiff <= tol and ids_equal and determinism and rows_equal}
    out_path.write_text(json.dumps(res))
    print(json.dumps(res))
    assert res["pass"], "cache consistency gate FAILED"
    return res


if __name__ == "__main__":
    import sys
    if "--verify-cache" in sys.argv:
        verify_cache()
    else:
        train()
