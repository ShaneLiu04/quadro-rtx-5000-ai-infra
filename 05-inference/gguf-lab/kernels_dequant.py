"""kernels_dequant.py — fused single-kernel Q4_K/Q8_0 dequantization (triton).

Motivation (measured in E3): the multi-op torch dequant is launch-bound under
WDDM (dozens of kernel launches dominate) and never approaches HBM bandwidth;
a per-block triton kernel is CTA-scheduling bound (one tiny program per
32/256 elements). The flat design below gives each program a contiguous chunk
of OUTPUT elements and does every access as a vector gather - this is the
same reason llama.cpp fuses dequantization into its GEMM kernels
(mmvq.cu/mmq.cu): one pass over the packed bytes, bandwidth-limited.

Reference formulas: ggml-quants.c dequantize_row_q4_K (L1530) / q8_0 (L553).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_q4_k_kernel(b8_ptr, b16_ptr, out_ptr, n_elem, E: tl.constexpr):
    """Flat: program covers E consecutive output elements.

    x = d * sc * q - dmin * m; 6-bit scales unpacked per element
    (get_scale_min_k4 bit layout); nibble low = even sub-block, high = odd.
    b16_ptr = uint16 view of the same block buffer (for fp16 d/dmin loads).
    """
    pid = tl.program_id(0)
    offs = pid * E + tl.arange(0, E)
    mask = offs < n_elem

    sb = offs // 256                       # super-block index
    within = offs % 256
    sub = within // 32                     # sub-block 0..7
    l = within % 32
    pair = sub // 2
    par = sub % 2

    # fp16 d / dmin via uint16 view (144 B = 72 uint16 per super-block)
    d = tl.load(b16_ptr + sb * 72, mask=mask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
    dmin = tl.load(b16_ptr + sb * 72 + 1, mask=mask, other=0).to(tl.float16, bitcast=True).to(tl.float32)

    # 6-bit scale/min unpack, all vector gathers (byte offsets within block):
    # j<4 : sc = byte j & 63            m = byte j+4 & 63
    # j>=4: sc = (byte j+4 & 0xF) | ((byte j-4 >> 6) << 4)
    #       m  = (byte j+4 >> 4)  | ((byte j   >> 6) << 4)
    sc_lo = tl.load(b8_ptr + sb * 144 + 4 + tl.where(sub < 4, sub, sub + 4), mask=mask, other=0)
    sc_hi = tl.load(b8_ptr + sb * 144 + 4 + tl.where(sub < 4, sub, sub - 4), mask=mask, other=0)
    m_src = tl.load(b8_ptr + sb * 144 + 4 + tl.where(sub < 4, sub + 4, sub), mask=mask, other=0)
    sc6 = tl.where(sub < 4, sc_lo & 63, (sc_lo & 15) | ((sc_hi >> 6) << 4)).to(tl.float32)
    mn6 = tl.where(sub < 4, m_src & 63, (sc_lo >> 4) | ((m_src >> 6) << 4)).to(tl.float32)

    # weight nibble: byte 16 + pair*32 + l, low = even sub-block, high = odd
    q = tl.load(b8_ptr + sb * 144 + 16 + pair * 32 + l, mask=mask, other=0)
    nib = tl.where(par == 0, q & 15, q >> 4).to(tl.float32)

    x = d * sc6 * nib - dmin * mn6
    tl.store(out_ptr + offs, x.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dequant_q8_0_kernel(b8_ptr, b16_ptr, out_ptr, n_elem, E: tl.constexpr):
    """Flat: x = fp16(d) * int8(q), d via uint16 view (34 B = 17 uint16)."""
    pid = tl.program_id(0)
    offs = pid * E + tl.arange(0, E)
    mask = offs < n_elem
    blk = offs // 32
    d = tl.load(b16_ptr + blk * 17, mask=mask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
    q = tl.load(b8_ptr + blk * 34 + 2 + offs % 32, mask=mask, other=0).to(tl.float32)
    q = tl.where(q > 127, q - 256.0, q)    # uint8 -> int8
    tl.store(out_ptr + offs, (d * q).to(out_ptr.dtype.element_ty), mask=mask)


def dequant_q4_k_triton(blocks: torch.Tensor, out_dtype=torch.float32, E=2048, warps=4):
    """blocks: uint8 (n_blocks, 144) on GPU -> (n_blocks*256,) dequantized."""
    assert blocks.is_cuda and blocks.shape[-1] == 144
    b = blocks.reshape(-1, 144).contiguous()
    n_elem = b.shape[0] * 256
    out = torch.empty(n_elem, device=b.device, dtype=out_dtype)
    grid = ((n_elem + E - 1) // E,)
    _dequant_q4_k_kernel[grid](b, b.view(torch.uint16), out, n_elem, E=E, num_warps=warps)
    return out


def dequant_q8_0_triton(blocks: torch.Tensor, out_dtype=torch.float32, E=4096, warps=4):
    assert blocks.is_cuda and blocks.shape[-1] == 34
    b = blocks.reshape(-1, 34).contiguous()
    n_elem = b.shape[0] * 32
    out = torch.empty(n_elem, device=b.device, dtype=out_dtype)
    grid = ((n_elem + E - 1) // E,)
    _dequant_q8_0_kernel[grid](b, b.view(torch.uint16), out, n_elem, E=E, num_warps=warps)
    return out


if __name__ == "__main__":
    import quant_gguf as qg
    torch.manual_seed(0)
    w = torch.randn(128, 512) * 0.05       # 128*512 = 65536 = 256*256 elements
    # Q4_K
    b4 = qg.quantize_q4_K(w).view(-1, 144).cuda()
    ref = qg.dequantize_q4_K(b4.cpu(), w.shape).flatten().cuda()
    got = dequant_q4_k_triton(b4)
    err = (got - ref).abs().max().item()
    print(f"q4_K triton vs cpu-ref maxdiff {err:.2e}")
    assert err < 1e-5, "q4_K triton mismatch"
    # odd size (mask path)
    w2 = torch.randn(7, 512) * 0.05
    b4b = qg.quantize_q4_K(w2).view(-1, 144).cuda()
    refb = qg.dequantize_q4_K(b4b.cpu(), w2.shape).flatten().cuda()
    gotb = dequant_q4_k_triton(b4b)
    assert (gotb - refb).abs().max().item() < 1e-5, "q4_K mask path mismatch"
    # Q8_0
    b8 = qg.quantize_q8_0(w).view(-1, 34).cuda()
    ref8 = qg.dequantize_q8_0(b8.cpu(), w.shape).flatten().cuda()
    got8 = dequant_q8_0_triton(b8)
    err8 = (got8 - ref8).abs().max().item()
    print(f"q8_0 triton vs cpu-ref maxdiff {err8:.2e}")
    assert err8 < 1e-5, "q8_0 triton mismatch"
    print("triton dequant kernels: exact match vs reference port (incl. mask path)")
