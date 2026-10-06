"""quant_gguf.py — Q8_0 / Q4_K quantization (faithful vectorized port of
llama.cpp ggml/src/ggml-quants.c) + GGUF export/import via gguf-py.

Reference mapping (ggml-quants.c, llama.cpp @ 05-inference clone):
  quantize_row_q8_0_ref   L276   -> quantize_q8_0
  dequantize_row_q8_0     L553   -> dequantize_q8_0
  nearest_int             L621   -> torch.round  (2^23 magic trick == round-half-to-even)
  make_qkx2_quants        L799   -> _make_qkx2_quants (vectorized over all sub-blocks)
  get_scale_min_k4        L880   -> pack/unpack in _pack_scales6/_unpack_scales6
  quantize_row_q4_K_ref   L1458  -> quantize_q4_K
  dequantize_row_q4_K     L1530  -> dequantize_q4_K

Honesty note: float summation order differs from the C reference (vectorized
reductions), so quantized values may differ from llama.cpp bit-exact output by
+-1 quantization level in rare cases. Block layout, packing bit order and the
dequantization formula are exact per the C source above.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).parent
sys.path.insert(0, str(LAB.parent / "llama.cpp" / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType  # noqa: E402

QK8_0 = 32
QK_K = 256
Q4_K_BLOCK_BYTES = 144  # 2B d + 2B dmin + 12B scales + 128B qs
Q8_0_BLOCK_BYTES = 34   # 2B d + 32B qs


# ================================================================ Q8_0
@torch.no_grad()
def quantize_q8_0(w: torch.Tensor):
    """w: (out, in), in % 32 == 0. Returns uint8 (out, in/32, 34) block bytes."""
    assert w.shape[1] % QK8_0 == 0
    x = w.detach().float().cpu()
    out, n = x.shape
    xb = x.view(out, n // QK8_0, QK8_0)
    amax = xb.abs().amax(-1)                          # (out, nb)
    d = amax / 127.0
    iden = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = torch.round(xb * iden[..., None]).clamp(-127, 127).to(torch.int8)
    d16 = d.to(torch.float16)                          # stored scale (fp16)
    blocks = torch.empty(out, n // QK8_0, Q8_0_BLOCK_BYTES, dtype=torch.uint8)
    blocks[..., :2] = d16.view(torch.uint8).view(out, -1, 2)
    blocks[..., 2:] = q.view(torch.uint8)
    return blocks


@torch.no_grad()
def dequantize_q8_0(blocks: torch.Tensor, shape):
    """blocks: uint8 (..., nb, 34) -> float32 tensor of `shape`."""
    d = blocks[..., :2].view(torch.float16)[..., 0].float()   # (..., nb)
    q = blocks[..., 2:].view(torch.int8).float()              # (..., nb, 32)
    x = (d[..., None] * q).reshape(*blocks.shape[:-2], -1)
    return x.reshape(shape)


# ================================================================ Q4_K
def _make_qkx2_quants(x, wt, nmax=15, rmin=-1.0, rdelta=0.1, nstep=20):
    """Vectorized port of make_qkx2_quants (n=32, use_mad=false).

    x, wt: (R, 32). Returns (scale (R,), the_min (R,), L (R, 32) uint8).
    """
    R = x.shape[0]
    mn = x.min(-1).values
    mx = x.max(-1).values
    mn = torch.minimum(mn, torch.zeros_like(mn))     # if (min > 0) min = 0
    degenerate = mx == mn
    span = torch.where(degenerate, torch.ones_like(mx), mx - mn)

    sum_w = wt.sum(-1)
    sum_x = (wt * x).sum(-1)

    iscale0 = nmax / span
    L = torch.round(iscale0[..., None] * (x - mn[..., None])).clamp(0, nmax)
    scale0 = 1.0 / iscale0
    scale = torch.where(degenerate, torch.zeros_like(scale0), scale0)
    off = torch.where(degenerate, mn, mn)            # affine offset (min)
    err0 = (wt * (scale[..., None] * L + off[..., None] - x) ** 2).sum(-1)
    best_err = torch.where(degenerate, torch.zeros_like(err0), err0)

    for is_ in range(nstep + 1):
        iscale = (rmin + rdelta * is_ + nmax) / span
        l = torch.round(iscale[..., None] * (x - mn[..., None])).clamp(0, nmax)
        sum_l = (wt * l).sum(-1)
        sum_l2 = (wt * l * l).sum(-1)
        sum_xl = (wt * l * x).sum(-1)
        D = sum_w * sum_l2 - sum_l * sum_l
        valid = (D > 0) & ~degenerate
        Dsafe = torch.where(valid, D, torch.ones_like(D))
        tscale = (sum_w * sum_xl - sum_x * sum_l) / Dsafe
        tmin = (sum_l2 * sum_x - sum_l * sum_xl) / Dsafe
        pos_min = tmin > 0
        l2safe = torch.where(sum_l2 > 0, sum_l2, torch.ones_like(sum_l2))
        tscale = torch.where(pos_min & valid, sum_xl / l2safe, tscale)
        tmin = torch.where(pos_min, torch.zeros_like(tmin), tmin)
        cur_err = (wt * (tscale[..., None] * l + tmin[..., None] - x) ** 2).sum(-1)
        better = valid & (cur_err < best_err)
        L = torch.where(better[..., None], l, L)
        scale = torch.where(better, tscale, scale)
        off = torch.where(better, tmin, off)
        best_err = torch.where(better, cur_err, best_err)

    the_min = -off                                    # *the_min = -min
    return scale, the_min, L.clamp(0, nmax).to(torch.uint8)


def _pack_scales6(ls, lm):
    """Pack 8x 6-bit (scale, min) pairs into 12 bytes — exact port of
    quantize_row_q4_K_ref L1490-1502. ls, lm: (R, 8) uint8."""
    R = ls.shape[0]
    sc = torch.zeros(R, 12, dtype=torch.uint8)
    for j in range(8):
        if j < 4:
            sc[:, j] = ls[:, j]
            sc[:, j + 4] = lm[:, j]
        else:
            sc[:, j + 4] = (ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)
            sc[:, j - 4] |= (ls[:, j] >> 4) << 6
            sc[:, j] |= (lm[:, j] >> 4) << 6
    return sc


def _unpack_scales6(sc):
    """Inverse of _pack_scales6 == get_scale_min_k4 (L880). sc: (R, 12) uint8."""
    ls = torch.zeros(sc.shape[0], 8, dtype=torch.uint8, device=sc.device)
    lm = torch.zeros_like(ls)
    for j in range(8):
        if j < 4:
            ls[:, j] = sc[:, j] & 63
            lm[:, j] = sc[:, j + 4] & 63
        else:
            ls[:, j] = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
            lm[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
    return ls, lm


@torch.no_grad()
def quantize_q4_K(w: torch.Tensor):
    """w: (out, in), in % 256 == 0. Returns uint8 (out, in/256, 144) block bytes.

    Faithful port of quantize_row_q4_K_ref: weighted qkx2 scale/min search per
    32-elem sub-block, 6-bit super-block packing, fp16(d)/fp16(dmin) requant.
    """
    assert w.shape[1] % QK_K == 0
    x = w.detach().float().cpu()
    out, n = x.shape
    nb = n // QK_K
    xs = x.view(out, nb, 8, 32)                       # (out, nb, sub, 32)

    # weights: av_x + |x| (L1473-1476)
    sum_x2 = (xs * xs).sum(-1)
    av_x = torch.sqrt(sum_x2 / 32)
    wt = av_x[..., None] + xs.abs()

    # per-sub-block optimal scale/min (L1477)
    R = out * nb * 8
    scale, the_min, _L = _make_qkx2_quants(xs.reshape(R, 32), wt.reshape(R, 32))
    scale = scale.view(out, nb, 8)
    the_min = the_min.view(out, nb, 8)

    # super-block d/dmin (L1488-1505)
    max_scale = scale.amax(-1)                        # (out, nb)
    max_min = the_min.amax(-1)
    inv_scale = torch.where(max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale))
    inv_min = torch.where(max_min > 0, 63.0 / max_min, torch.zeros_like(max_min))
    ls = (torch.round(inv_scale[..., None] * scale)).clamp(0, 63).to(torch.uint8)
    lm = (torch.round(inv_min[..., None] * the_min)).clamp(0, 63).to(torch.uint8)
    d = (max_scale / 63.0).to(torch.float16)          # fp16 stored
    dmin = (max_min / 63.0).to(torch.float16)
    sc_bytes = _pack_scales6(ls.reshape(-1, 8), lm.reshape(-1, 8)).view(out, nb, 12)

    # requantize with the fp16-round-tripped effective scales (L1507-1518)
    d_eff = d.float()[..., None] * ls.float()         # (out, nb, 8)
    dm_eff = dmin.float()[..., None] * lm.float()
    lq = torch.round((xs + dm_eff[..., None]) / d_eff[..., None]).clamp(0, 15)
    lq = torch.where(d_eff[..., None] > 0, lq, torch.zeros_like(lq)).to(torch.uint8)

    # nibble pack: byte = even_sub(low) | odd_sub(high)<<4 (L1520-1524)
    lo = lq[..., 0::2, :]                             # sub-blocks 0,2,4,6
    hi = lq[..., 1::2, :]                             # sub-blocks 1,3,5,7
    qs = (lo | (hi << 4)).view(out, nb, 128)

    blocks = torch.cat([
        d.view(torch.uint8).view(out, nb, 2),
        dmin.view(torch.uint8).view(out, nb, 2),
        sc_bytes,
        qs,
    ], dim=-1)
    assert blocks.shape[-1] == Q4_K_BLOCK_BYTES
    return blocks


@torch.no_grad()
def dequantize_q4_K(blocks: torch.Tensor, shape, device=None):
    """blocks: uint8 (..., nb, 144) -> float32 tensor of `shape`.

    Exact port of dequantize_row_q4_K: y = d*sc*q - dmin*m, nibble low = even
    sub-block, high = odd sub-block.
    """
    d = blocks[..., 0:2].view(torch.float16)[..., 0].float()
    dmin = blocks[..., 2:4].view(torch.float16)[..., 0].float()
    ls, lm = _unpack_scales6(blocks[..., 4:16].reshape(-1, 12))
    ls = ls.float().view(*blocks.shape[:-1], 8)       # (..., nb, 8)
    lm = lm.float().view(*blocks.shape[:-1], 8)
    qs = blocks[..., 16:].view(*blocks.shape[:-1], 4, 32)

    lo = (qs & 0xF).float()                           # even sub-blocks
    hi = (qs >> 4).float()                            # odd sub-blocks
    d_eff = d[..., None] * ls                         # (..., nb, 8)
    dm_eff = dmin[..., None] * lm
    x = torch.empty(*blocks.shape[:-1], 8, 32, device=blocks.device)
    x[..., 0::2, :] = d_eff[..., 0::2, None] * lo - dm_eff[..., 0::2, None]
    x[..., 1::2, :] = d_eff[..., 1::2, None] * hi - dm_eff[..., 1::2, None]
    x = x.reshape(*blocks.shape[:-1], -1)
    if device is not None:
        x = x.to(device)
    return x.reshape(shape)


# ================================================================ GGUF I/O
def quantize_tensor(w, quant):
    if quant is None or w.dim() < 2:
        # 1D norm weights are never block-quantized (llama.cpp convention)
        return w.detach().cpu().to(torch.float16).numpy(), None
    if quant == "q8_0":
        b = quantize_q8_0(w)
        return b.numpy(), GGMLQuantizationType.Q8_0
    if quant == "q4_K":
        b = quantize_q4_K(w)
        return b.numpy(), GGMLQuantizationType.Q4_K
    raise ValueError(quant)


def dequantize_tensor(data, ttype, shape, device=None):
    if ttype == GGMLQuantizationType.F16:
        t = torch.from_numpy(np.asarray(data).copy()).float()
    elif ttype == GGMLQuantizationType.Q8_0:
        arr = torch.from_numpy(np.asarray(data).copy())
        blocks = arr.reshape(shape[0], -1, Q8_0_BLOCK_BYTES)
        t = dequantize_q8_0(blocks, shape)
    elif ttype == GGMLQuantizationType.Q4_K:
        arr = torch.from_numpy(np.asarray(data).copy())
        blocks = arr.reshape(shape[0], -1, Q4_K_BLOCK_BYTES)
        t = dequantize_q4_K(blocks, shape)
    else:
        raise ValueError(ttype)
    return t.to(device) if device else t


def save_gguf(model, tok, path, quant=None):
    """Export model as GGUF with llama-arch metadata. quant in {None,q8_0,q4_K}."""
    w = GGUFWriter(path, "llama")
    w.add_string("general.name", "charlm-ar005")
    w.add_string("general.description", "tiny llama-arch char LM trained in gguf-lab (AR005)")
    w.add_uint32("llama.embedding_length", 256)
    w.add_uint32("llama.block_count", 4)
    w.add_uint32("llama.attention.head_count", 4)
    w.add_uint32("llama.attention.head_count_kv", 2)
    w.add_uint32("llama.rope.dimension_count", 64)
    w.add_uint32("llama.context_length", model.ctx)
    w.add_string("tokenizer.ggml.model", "char")
    w.add_array("tokenizer.ggml.tokens", list(tok.chars))
    import charlm
    names = charlm.llama_names(model.state_dict())
    for k, v in model.state_dict().items():
        data, ttype = quantize_tensor(v, quant)
        if ttype is None:
            w.add_tensor(names[k], data)
        else:
            data = data.reshape(data.shape[0], -1)      # (out, row_bytes)
            w.add_tensor(names[k], data, raw_shape=data.shape, raw_dtype=ttype)
    w.write_header_to_file(path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    return path


def charlm_llama_names():
    return {
        "embed.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }


def llama_to_charlm_names():
    m = charlm_llama_names()
    m = {v: k for k, v in m.items()}
    for l in range(4):
        for a, b in [("attn_q", "attn.q_proj"), ("attn_k", "attn.k_proj"),
                     ("attn_v", "attn.v_proj"), ("attn_output", "attn.o_proj"),
                     ("ffn_gate", "mlp.gate_proj"), ("ffn_up", "mlp.up_proj"),
                     ("ffn_down", "mlp.down_proj")]:
            m[f"blk.{l}.{a}.weight"] = f"blocks.{l}.{b}.weight"
        m[f"blk.{l}.attn_norm.weight"] = f"blocks.{l}.attn_norm.weight"
        m[f"blk.{l}.ffn_norm.weight"] = f"blocks.{l}.ffn_norm.weight"
    return m


def load_gguf(path, device=None):
    """Read GGUF -> (state_dict float32, config dict, tokenizer chars list)."""
    r = GGUFReader(path)
    cfg = {}
    for key in ["llama.embedding_length", "llama.block_count", "llama.attention.head_count",
                "llama.attention.head_count_kv", "llama.rope.dimension_count",
                "llama.context_length"]:
        fld = r.fields.get(key)
        if fld is not None:
            cfg[key] = fld.parts[-1].tolist()[0]
    chars = list(r.fields["tokenizer.ggml.tokens"].contents())
    # bytes -> str: gguf-py returns np.bytes_ for string arrays
    chars = [c.decode() if isinstance(c, bytes) else str(c) for c in chars]
    name_map = llama_to_charlm_names()
    sd = {}
    for t in r.tensors:
        logical = tuple(int(x) for x in reversed(t.shape))  # ggml ne order -> numpy
        sd[name_map[t.name]] = dequantize_tensor(t.data, t.tensor_type, logical, device)
    cfg2 = {"vocab": len(chars), "d": cfg.get("llama.embedding_length", 256),
            "n_layers": cfg.get("llama.block_count", 4),
            "n_qh": cfg.get("llama.attention.head_count", 4),
            "n_kvh": cfg.get("llama.attention.head_count_kv", 2),
            "n_ff": sd["blocks.0.mlp.gate_proj.weight"].shape[0],
            "ctx": cfg.get("llama.context_length", 256)}
    return sd, cfg2, chars


def model_from_gguf(path, device="cuda", dtype=torch.float32):
    import charlm
    sd, cfg, chars = load_gguf(path)
    model = charlm.CharLM(cfg["vocab"], cfg["d"], cfg["n_layers"], cfg["n_qh"],
                          cfg["n_kvh"], cfg["n_ff"], cfg["ctx"])
    model.load_state_dict(sd)
    model.to(device).to(dtype).eval()
    tok = charlm.CharTokenizer("".join(chars))
    return model, tok


if __name__ == "__main__":
    # self-test: quantize -> dequantize roundtrip error gates
    torch.manual_seed(0)
    w = torch.randn(64, 512) * 0.05
    b8 = quantize_q8_0(w)
    x8 = dequantize_q8_0(b8, w.shape)
    e8 = (x8 - w).norm() / w.norm()                       # data-dependent (gauss ~5e-3)
    m8 = (x8 - w).abs().max() / w.abs().max()             # max-abs normalized
    b4 = quantize_q4_K(w)
    x4 = dequantize_q4_K(b4, w.shape)
    e4 = (x4 - w).norm() / w.norm()
    print(f"q8_0 blocks {tuple(b8.shape)} rel-err {e8.item():.2e} max-abs-norm {m8.item():.2e} (gate 1/128={1/128:.2e})")
    print(f"q4_K blocks {tuple(b4.shape)} rel-err {e4.item():.2e} (gate 0.25)")
    assert m8.item() <= 1 / 128, "Q8_0 half-step gate failed"
    assert e4.item() <= 0.25, "Q4_K gate failed"
    # scale packing roundtrip
    ls = torch.randint(0, 64, (10, 8))
    lm = torch.randint(0, 64, (10, 8))
    ls2, lm2 = _unpack_scales6(_pack_scales6(ls, lm))
    assert torch.equal(ls, ls2) and torch.equal(lm, lm2), "6-bit pack roundtrip failed"
    print("6-bit scale pack/unpack roundtrip: exact")
