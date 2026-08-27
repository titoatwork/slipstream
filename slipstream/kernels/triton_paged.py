"""Triton reshape_and_cache + paged decode. Callers fall back if this raises."""

from __future__ import annotations

from typing import Any

import torch

from slipstream.kernels.attention_ref import _flatten_kv

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except ImportError:
    triton = None
    tl = None

TRITON_AVAILABLE: bool = triton is not None
_reshape_and_cache_kernel: Any = None
_paged_attention_decode_kernel: Any = None

_MAX_HEAD_DIM = 256
_MAX_BLOCK_SIZE = 128


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _require_triton() -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")


if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _reshape_and_cache_kernel(  # type: ignore[no-untyped-def]
        k_ptr,
        v_ptr,
        k_cache_ptr,
        v_cache_ptr,
        slot_ptr,
        n_tok,
        head_dim,
        block_size,
        stride_k_tok,
        stride_k_h,
        stride_k_d,
        stride_v_tok,
        stride_v_h,
        stride_v_d,
        stride_ck_blk,
        stride_ck_off,
        stride_ck_h,
        stride_ck_d,
        stride_cv_blk,
        stride_cv_off,
        stride_cv_h,
        stride_cv_d,
        BLOCK_D: tl.constexpr,
    ) -> None:
        tok = tl.program_id(0)
        h = tl.program_id(1)
        if tok >= n_tok:
            return
        slot = tl.load(slot_ptr + tok)
        if slot < 0:
            return
        block_id = (slot // block_size).to(tl.int64)
        offset = (slot % block_size).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < head_dim
        k = tl.load(k_ptr + tok * stride_k_tok + h * stride_k_h + offs_d * stride_k_d, mask=mask)
        v = tl.load(v_ptr + tok * stride_v_tok + h * stride_v_h + offs_d * stride_v_d, mask=mask)
        tl.store(
            k_cache_ptr
            + block_id * stride_ck_blk
            + offset * stride_ck_off
            + h * stride_ck_h
            + offs_d * stride_ck_d,
            k,
            mask=mask,
        )
        tl.store(
            v_cache_ptr
            + block_id * stride_cv_blk
            + offset * stride_cv_off
            + h * stride_cv_h
            + offs_d * stride_cv_d,
            v,
            mask=mask,
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def _paged_attention_decode_kernel(  # type: ignore[no-untyped-def]
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        out_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        scale,
        n_q,
        n_kv,
        head_dim,
        tq,
        block_size,
        max_blocks,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_bt0,
        stride_bt1,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        q_head = tl.program_id(1)
        seq_len = tl.load(seq_lens_ptr + batch_idx).to(tl.int32)
        n_rep = n_q // n_kv
        kv_head = q_head // n_rep
        n_blocks = tl.minimum(tl.cdiv(seq_len, block_size), max_blocks)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        offs_t = tl.arange(0, BLOCK_M)

        for tq_i in range(tq):
            q = tl.load(
                q_ptr
                + batch_idx * stride_qb
                + q_head * stride_qh
                + tq_i * stride_qt
                + offs_d * stride_qd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            # packed causal: query tq_i sits at absolute index seq_len - tq + tq_i
            kv_limit = seq_len - tq + tq_i + 1
            m_i = tl.full((), float("-inf"), dtype=tl.float32)
            l_i = tl.zeros((), dtype=tl.float32)
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)

            for blk in range(n_blocks):
                phys = tl.load(block_tables_ptr + batch_idx * stride_bt0 + blk * stride_bt1)
                start = blk * block_size
                phys_ok = phys >= 0
                phys_i = tl.where(phys_ok, phys, 0).to(tl.int64)
                pos = start + offs_t
                valid = phys_ok & (offs_t < block_size) & (pos < kv_limit)
                k = tl.load(
                    k_cache_ptr
                    + phys_i * stride_kb
                    + offs_t[:, None] * stride_kt
                    + kv_head * stride_kh
                    + offs_d[None, :] * stride_kd,
                    mask=valid[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                v = tl.load(
                    v_cache_ptr
                    + phys_i * stride_vb
                    + offs_t[:, None] * stride_vt
                    + kv_head * stride_vh
                    + offs_d[None, :] * stride_vd,
                    mask=valid[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                qk = tl.sum(q[None, :] * k, axis=1) * scale
                qk = tl.where(valid, qk, float("-inf"))
                m_ij = tl.max(qk)
                if m_ij > float("-inf"):
                    m_new = tl.maximum(m_i, m_ij)
                    alpha = tl.exp(m_i - m_new)
                    p = tl.exp(qk - m_new)
                    p = tl.where(valid, p, 0.0)
                    l_i = l_i * alpha + tl.sum(p)
                    acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                    m_i = m_new

            denom = tl.where(l_i > 0, l_i, 1.0)
            out = (acc / denom).to(out_ptr.dtype.element_ty)
            tl.store(
                out_ptr
                + batch_idx * stride_ob
                + tq_i * stride_ot
                + q_head * stride_oh
                + offs_d * stride_od,
                out,
                mask=mask_d,
            )


def reshape_and_cache_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    layer_idx: int,
    block_size: int,
) -> None:
    """Scatter flattened K/V into `kv_cache[layer, 0/1, block, offset]`."""
    _require_triton()
    if k.device.type != "cuda" or kv_cache.device.type != "cuda":
        raise RuntimeError("reshape_and_cache_triton requires CUDA tensors")
    k, v = _flatten_kv(k, v)
    n_tok, n_kv, head_dim = k.shape
    if slot_mapping.numel() != n_tok:
        raise ValueError(f"slot_mapping {slot_mapping.numel()} != tokens {n_tok}")
    if n_tok == 0 or n_kv == 0 or head_dim == 0:
        return
    if head_dim > _MAX_HEAD_DIM:
        raise RuntimeError(f"head_dim {head_dim} exceeds Triton tile {_MAX_HEAD_DIM}")
    slot_mapping = slot_mapping.contiguous()
    k_cache = kv_cache[layer_idx, 0]
    v_cache = kv_cache[layer_idx, 1]
    block_d = _next_power_of_two(head_dim)
    _reshape_and_cache_kernel[(n_tok, n_kv)](
        k,
        v,
        k_cache,
        v_cache,
        slot_mapping,
        n_tok,
        head_dim,
        block_size,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        BLOCK_D=block_d,
        num_warps=2,
    )


def paged_attention_decode_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    layer_idx: int,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """Paged decode. Launch grid is `(batch, n_q_heads)` only."""
    _require_triton()
    if q.device.type != "cuda" or kv_cache.device.type != "cuda":
        raise RuntimeError("paged_attention_decode_triton requires CUDA tensors")
    if q.ndim != 4:
        raise ValueError(f"q must be [B, n_q, Tq, D], got {tuple(q.shape)}")
    batch, n_q, tq, head_dim = q.shape
    n_kv = int(kv_cache.shape[4])
    cache_dim = int(kv_cache.shape[5])
    if head_dim != cache_dim:
        raise ValueError(f"q head_dim {head_dim} != cache {cache_dim}")
    if n_kv == 0 or n_q % n_kv != 0:
        raise ValueError(f"GQA requires n_q % n_kv == 0, got n_q={n_q} n_kv={n_kv}")
    if head_dim > _MAX_HEAD_DIM or block_size > _MAX_BLOCK_SIZE or block_size < 1:
        raise RuntimeError(f"unsupported decode tile head_dim={head_dim} block_size={block_size}")
    if batch == 0 or n_q == 0 or tq == 0:
        return q.new_empty((batch, tq, n_q, head_dim))

    q = q.contiguous()
    block_tables = block_tables.contiguous()
    seq_lens = seq_lens.contiguous()
    k_cache = kv_cache[layer_idx, 0]
    v_cache = kv_cache[layer_idx, 1]
    out = torch.empty((batch, tq, n_q, head_dim), device=q.device, dtype=q.dtype)
    max_blocks = int(block_tables.shape[1])
    block_m = _next_power_of_two(block_size)
    block_d = _next_power_of_two(head_dim)
    # Grid depends only on (B, n_q). seq_lens==0 masks unused batch slots.
    grid = (batch, n_q)
    _paged_attention_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        out,
        block_tables,
        seq_lens,
        float(scale),
        n_q,
        n_kv,
        head_dim,
        tq,
        block_size,
        max_blocks,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out
