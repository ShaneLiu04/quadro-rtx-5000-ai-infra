"""quant_ops.py — pure-torch re-implementations of bnb quantization algorithms (AR007).

Aligned with bitsandbytes 0.50.3.dev0 semantics, cited as file:line [source]:
- NF4 quantile map:    functional.py:170-196; bucketize via midpoints backends/default/ops.py:194
- blockwise quantize:  backends/default/ops.py:180-196
- vectorwise int8:     backends/default/ops.py:142-177; functional.py:1655
- int8 mm dequant:     backends/default/ops.py:38-61 (6.200124e-05 == 1/127^2)
- 8-bit Adam states:   functional.py quantize_blockwise (blocksize 4096 default),
                       backends/cpu/ops.py:469-580 reference semantics
All math is torch-only; runs on CPU and CUDA. 4-bit payloads are stored as
uint8 index tensors (two half-bytes per element is a packing detail we do NOT
emulate — we measure quantization error, not packing density).
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------- codebooks

# Official NF4 table (QLoRA paper; ships in bnb get_4bit_type('nf4'), verified
# [local] via clone import — NOT the raw ndtri quantiles: endpoints pinned to +-1,
# inner levels are the paper's fitted values). [source] functional.py:772-806
NF4_OFFICIAL = [
    -1.0, -0.696192801, -0.5250730515, -0.3949174881, -0.2844413817, -0.1847734302,
    -0.0910500363, 0.0, 0.0795802996, 0.1609302014, 0.2461123019, 0.3379152417,
    0.4407098293, 0.5626170039, 0.7229568362, 1.0,
]


def nf4_levels(device=None, dtype=torch.float32) -> torch.Tensor:
    """Official NF4 16-level codebook [source] bnb functional.py:772-806 + QLoRA paper."""
    return torch.tensor(NF4_OFFICIAL, device=device, dtype=dtype)


def nf4_raw_levels(device=None, dtype=torch.float32) -> torch.Tensor:
    """Raw quantile-matched levels ndtri((2k+1)/32) — the 'textbook' variant WITHOUT the
    paper's endpoint pinning. Kept as a comparator: official table != raw quantiles
    (measured gap is an E1 teaching point)."""
    ps = torch.tensor([(2 * k + 1) / 32 for k in range(16)], dtype=torch.float64)
    return torch.special.ndtri(ps).to(device=device, dtype=dtype)


# e2m1 magnitudes (bnb get_4bit_type('fp4'): 1 sign, 2 exp, 1 mantissa) [source] functional.py:807-825
_FP4_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def fp4_levels(device=None, dtype=torch.float32) -> torch.Tensor:
    m = torch.tensor(_FP4_MAGS, dtype=torch.float64)
    lv = torch.cat([m.flip(0).neg()[:-1], m])  # symmetric: -6..-0.5,0,0.5..6
    return lv.to(device=device, dtype=dtype)


def int4_levels(device=None, dtype=torch.float32) -> torch.Tensor:
    """16 uniform levels over [-1, 1] — the 'Int4' comparator in the QLoRA paper (NOT two's complement)."""
    return torch.linspace(-1.0, 1.0, 16, device=device, dtype=dtype)


def get_levels(name: str, device=None, dtype=torch.float32) -> torch.Tensor:
    fns = {"nf4": nf4_levels, "nf4_raw": nf4_raw_levels, "fp4": fp4_levels, "int4": int4_levels}
    return fns[name](device, dtype)


# ------------------------------------------------------- blockwise 4-bit

@torch.no_grad()
def blockwise_quant(W: torch.Tensor, levels: torch.Tensor, block: int):
    """Blockwise absmax + codebook bucketize [source] backends/default/ops.py:180-196,233-260.

    Returns dict with idx (uint8 per element), absmax (fp32 per block), pad,
    dequantized tensor and error metrics. rel_rmse = ||W-What||_2 / ||W||_2
    computed on the UNPADDED region.
    """
    flat = W.reshape(-1).float()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block)
    absmax = blocks.abs().amax(dim=1)
    absmax = absmax.clamp(min=1e-12)
    normed = blocks / absmax.unsqueeze(1)

    bounds = (levels[:-1] + levels[1:]) / 2  # midpoint boundaries [source] default/ops.py:194
    idx = torch.bucketize(normed, bounds).to(torch.uint8)
    deq_blocks = levels[idx.long()] * absmax.unsqueeze(1)
    deq = deq_blocks.reshape(-1)[: W.numel()].reshape(W.shape)

    err = (deq - W).float()
    denom = W.float().pow(2).sum().clamp(min=1e-30)
    rel_rmse = float(torch.sqrt((err**2).sum() / denom))
    return {
        "idx": idx,
        "absmax": absmax,
        "pad": pad,
        "dequant": deq,
        "rel_rmse": rel_rmse,
        "max_abs_err": float(err.abs().max()),
    }


# ----------------------------------------------------- vectorwise int8

@torch.no_grad()
def vw_quant_rows(A: torch.Tensor, threshold: float = 0.0):
    """Row-wise absmax int8 quantization with optional outlier-column split.

    [source] backends/default/ops.py:142-177; functional.py:1655.
    Returns (q int8 (M,K), row_stats fp32 (M,), outlier_cols or None, A_orig).
    Values >= threshold are removed before quantization; the columns holding
    them are reported for the fp16 side-path (LLM.int8 sparse decomposition).
    """
    Ac = A.clone()
    outlier_cols = None
    if threshold > 0.0:
        mask = Ac.abs() >= threshold
        if mask.any():
            outlier_cols = torch.argwhere(mask.any(dim=0)).view(-1)
            Ac[mask] = 0.0
    stats = Ac.abs().amax(dim=1).clamp(min=1e-12)
    q = torch.round(Ac * (127.0 / stats.unsqueeze(1))).clamp_(-127, 127).to(torch.int8)
    if outlier_cols is not None and Ac.shape[0] > 1:
        q[:, outlier_cols] = 0
    return q, stats, outlier_cols, A


@torch.no_grad()
def int8_mm_dequant(qA, sA, qB, sB, out_dtype=torch.float32):
    """int8 mm + dual-scale dequant. qA (M,K) int8 rowscaled by sA (M,);
    qB (N,K) int8 rowscaled by sB (N,). C = (qA @ qB^T) * (sA x sB) / 127^2.

    [source] default/ops.py:57 uses 6.200124e-05 == 1/127^2 == 1/16129.
    Uses torch._int_mm on CUDA when available (INT8 tensor cores), else fp32 emulation.
    """
    if qA.is_cuda:
        ci = torch._int_mm(qA, qB.t().contiguous())  # (M,N) int32
    else:
        ci = (qA.float() @ qB.t().float()).to(torch.int32)
    out = ci.float() * (sA.unsqueeze(1) * sB.unsqueeze(0)) * (1.0 / (127.0 * 127.0))
    return out.to(out_dtype)


@torch.no_grad()
def llm_int8_matmul(X: torch.Tensor, W: torch.Tensor, threshold: float = 0.0):
    """Full LLM.int8 path: vectorwise int8 + optional fp16 outlier side-path.

    [source] functional.py:1590-1671; default/ops.py:64-100 (int8_mixed_scaled_mm).
    X (M,K) activations, W (N,K) weights. Returns (out (M,N), meta dict).
    """
    qx, sx, ocols, _ = vw_quant_rows(X, threshold)
    qw, sw, _, _ = vw_quant_rows(W, 0.0)
    out = int8_mm_dequant(qx, sx, qw, sw, out_dtype=torch.float32)
    meta = {"outlier_cols": None, "outlier_frac": 0.0, "kept_int8_cols": X.shape[1]}
    if ocols is not None and ocols.numel():
        sub_x = X[:, ocols].to(torch.float16)  # (M, K_sub)
        sub_w = W[:, ocols].t().to(torch.float16)  # (K_sub, N)
        out = out + (sub_x @ sub_w).float()
        meta["outlier_cols"] = int(ocols.numel())
        meta["outlier_frac"] = float(ocols.numel() / X.shape[1])
        meta["kept_int8_cols"] = int(X.shape[1] - ocols.numel())
    return out, meta


# ------------------------------------------------------- 8-bit Adam states

def dynamic_map(signed: bool = True) -> torch.Tensor:
    """bnb 'dynamic' 8-bit type: decade-based floating codebook over 10^-7..1
    (7 decades of dynamic range — the anti-underflow armor for optimizer states).
    [source] functional.py:296-348 (create_dynamic_map), exact formula:
    fraction_items = 2^(i + non_sign_bits - max_exp) + 1        (signed)
                   = 2^(i + non_sign_bits - max_exp + 1) + 1    (unsigned)
    v states use unsigned (255 nonzero levels); m states signed (255 too).
    """
    total_bits, max_exp = 8, 7
    non_sign_bits = total_bits - 1
    data = []
    for i in range(max_exp):
        frac_items = int(2 ** (i + non_sign_bits - max_exp) + 1) if signed else \
            int(2 ** (i + non_sign_bits - max_exp + 1) + 1)
        b = torch.linspace(0.1, 1, frac_items, dtype=torch.float32)
        means = (b[:-1] + b[1:]) / 2.0
        data += ((10 ** (-(max_exp - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exp - 1) + i)) * means).tolist()
    data.append(0.0)
    data.append(1.0)
    assert len(data) == 2**total_bits, len(data)
    data.sort()
    return torch.tensor(data, dtype=torch.float32)


class Adam8bitStates:
    """Blockwise 8-bit dynamic-type Adam m/v states, aligned with bnb semantics:
    - blocksize 256 [source] backends/cpu/ops.py:490
    - m: signed dynamic map; v: unsigned dynamic map [source] optim/optimizer.py:519-522
    - update arrangement [source] backends/cpu/ops.py:507-517:
        denom = (v.sqrt()/sqrt(1-b2^t)).add_(eps); p -= lr/(1-b1^t) * m/denom
    Storage: m_q/v_q uint8 codes, m_abs/v_abs fp32 per block.
    """

    M_MAP = None
    V_MAP = None

    def __init__(self, shape, device, block=256, b1=0.9, b2=0.999, eps=1e-8):
        if Adam8bitStates.M_MAP is None:
            Adam8bitStates.M_MAP = dynamic_map(True)
            Adam8bitStates.V_MAP = dynamic_map(False)
        self.shape, self.device, self.block = tuple(shape), device, block
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m_map = Adam8bitStates.M_MAP.to(device)
        self.v_map = Adam8bitStates.V_MAP.to(device)
        self.m_bounds = (self.m_map[:-1] + self.m_map[1:]) / 2
        self.v_bounds = (self.v_map[:-1] + self.v_map[1:]) / 2
        n = 1
        for s in shape:
            n *= s
        self.m_q = torch.zeros(shape, device=device, dtype=torch.uint8)
        self.v_q = torch.zeros(shape, device=device, dtype=torch.uint8)
        pad = (-n) % block
        self.pad = pad
        self.nblocks = (n + pad) // block
        self.m_abs = torch.zeros(self.nblocks, device=device, dtype=torch.float32)
        self.v_abs = torch.zeros(self.nblocks, device=device, dtype=torch.float32)
        self.t = 0

    def _deq(self, q, ab, mp):
        flat = q.reshape(-1).long()
        if self.pad:
            flat = torch.cat([flat, flat.new_zeros(self.pad)])
        codes = mp[flat]
        return (codes.reshape(-1, self.block) * ab.unsqueeze(1)).reshape(-1)[: q.numel()].reshape(self.shape)

    def _requant(self, x, q, ab, bounds):
        flat = x.reshape(-1).float()
        if self.pad:
            flat = torch.cat([flat, flat.new_zeros(self.pad)])
        blocks = flat.reshape(-1, self.block)
        am = blocks.abs().amax(dim=1).clamp(min=1e-12)
        normed = blocks / am.unsqueeze(1)
        idx = torch.bucketize(normed, bounds).clamp(0, 255).to(torch.uint8)
        q.copy_(idx.reshape(-1)[: x.numel()].reshape(self.shape))
        ab.copy_(am)

    def state_bytes(self):
        it = torch.tensor([], dtype=torch.uint8).element_size()
        ft = torch.tensor([], dtype=torch.float32).element_size()
        return self.m_q.numel() * it + self.v_q.numel() * it + (self.m_abs.numel() + self.v_abs.numel()) * ft

    @torch.no_grad()
    def step(self, param: torch.Tensor, grad: torch.Tensor, lr: float, weight_decay: float = 0.0):
        self.t += 1
        m = self._deq(self.m_q, self.m_abs, self.m_map)
        v = self._deq(self.v_q, self.v_abs, self.v_map)
        m.mul_(self.b1).add_(grad, alpha=1 - self.b1)
        v.mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)
        self._requant(m, self.m_q, self.m_abs, self.m_bounds)
        self._requant(v, self.v_q, self.v_abs, self.v_bounds)
        # [source] cpu/ops.py:511-517 arrangement
        c1 = 1 - self.b1**self.t
        c2 = (1 - self.b2**self.t) ** 0.5
        denom = (v.sqrt() / c2).add_(self.eps)
        if weight_decay:
            param.mul_(1 - lr * weight_decay)
        param.addcdiv_(m, denom, value=-lr / c1)
