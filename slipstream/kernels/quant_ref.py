"""PyTorch reference for W4A16 quantization. Source of truth for kernel parity.

S9 (Phase 5). The numeric oracle for weight-only INT4 quantization, mirroring
``attention_ref.py``'s role for paged attention: the Triton dequant-fused GEMM
must match ``w4a16_gemm_ref`` within tolerance, and this module defines the
weight *format* both share.

Scheme — **group-wise asymmetric (affine) INT4**, the AWQ/GPTQ family. A weight
matrix ``W`` of shape ``[out, in]`` (``nn.Linear.weight`` layout) is quantized
along ``in`` in contiguous groups of ``group_size``. Per (row, group):

    scale = (wmax - wmin) / 15                      # 4 bits → levels 0..15
    zero  = round(-wmin / scale)          (int 0..15, the affine zero-point)
    q     = clamp(round(w / scale) + zero, 0, 15)
    w_hat = (q - zero) * scale                      # dequantization

Storage is genuinely 4-bit — the whole point of W4A16 is **memory traffic**, not
FLOPs (MASTERPLAN §S9). ``q`` is packed 8 nibbles per int32 along ``in``:
``qweight[o, j]`` holds ``q[o, 8j .. 8j+7]`` with value ``n`` at bit ``4n``.
Scales are fp per (row, group); zero-points are int per (row, group).

**Critical framing (for the report):** W4A16 reduces bytes moved, giving ~4×
decode gain only in the memory-bound regime where dequant hides behind HBM
latency — and ~zero prefill gain, where dequant lands on the compute critical
path. This reference exists to prove *correctness*; the speedup asymmetry is a
GPU measurement (benchmarks/, T1/A100).
"""

from __future__ import annotations

import torch

_NIBBLE = 0xF
_U32 = 0xFFFFFFFF
_LEVELS = 15  # 2**4 - 1


def _check_shapes(in_features: int, group_size: int) -> None:
    if group_size % 8 != 0:
        raise ValueError(f"group_size must be a multiple of 8, got {group_size}")
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack ``[out, in]`` nibbles (int, 0..15) → ``[out, in//8]`` int32, 8 per word.

    Bit ``4n`` of word ``j`` holds ``q[:, 8j+n]``. Packing is done in int64 (the
    high nibble at shift 28 overflows signed int32) then cast so the 2's-complement
    bit pattern is preserved; ``unpack_int4`` recovers it via an unsigned mask.
    """
    if q.ndim != 2:
        raise ValueError(f"q must be 2-D [out, in], got {tuple(q.shape)}")
    out, in_features = q.shape
    if in_features % 8 != 0:
        raise ValueError(f"in_features {in_features} not divisible by 8")
    q64 = q.to(torch.int64).reshape(out, in_features // 8, 8)
    shifts = (torch.arange(8, device=q.device, dtype=torch.int64) * 4).view(1, 1, 8)
    packed = (q64 << shifts).sum(dim=-1)
    return packed.to(torch.int32)


def unpack_int4(qweight: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4`: ``[out, in//8]`` int32 → ``[out, in]`` int64 (0..15)."""
    out = qweight.shape[0]
    p = qweight.to(torch.int64) & _U32  # unsigned interpretation
    shifts = (torch.arange(8, device=qweight.device, dtype=torch.int64) * 4).view(1, 1, 8)
    nibbles = (p.unsqueeze(-1) >> shifts) & _NIBBLE
    return nibbles.reshape(out, in_features)


def quantize_w4a16(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` ``[out, in]`` → ``(qweight, scales, qzeros)``.

    Returns packed int32 weights ``[out, in//8]``, fp ``scales`` ``[out, in//group_size]``
    (same dtype as ``weight``), and int32 ``qzeros`` ``[out, in//group_size]`` (0..15).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [out, in], got {tuple(weight.shape)}")
    out, in_features = weight.shape
    _check_shapes(in_features, group_size)
    n_groups = in_features // group_size

    w = weight.reshape(out, n_groups, group_size).float()
    # Include 0 in each group's range so the integer zero-point is always
    # representable — this keeps a constant group offset from 0 exact and is a
    # no-op for the zero-straddling groups typical of trained weights.
    zeros_t = torch.zeros_like(w[..., :1])
    wmin = torch.minimum(w.amin(dim=-1, keepdim=True), zeros_t)
    wmax = torch.maximum(w.amax(dim=-1, keepdim=True), zeros_t)
    scale = (wmax - wmin).clamp_min(1e-12) / _LEVELS
    zero = torch.round(-wmin / scale).clamp_(0, _LEVELS)
    q = torch.clamp(torch.round(w / scale) + zero, 0, _LEVELS).reshape(out, in_features)

    qweight = pack_int4(q.to(torch.int64))
    scales = scale.squeeze(-1).to(weight.dtype)
    qzeros = zero.squeeze(-1).to(torch.int32)
    return qweight, scales, qzeros


def dequantize_w4a16(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Reconstruct the fp weight ``[out, in]`` from the packed W4A16 format."""
    out, n_groups = scales.shape
    in_features = n_groups * group_size
    q = unpack_int4(qweight, in_features).reshape(out, n_groups, group_size).float()
    scale = scales.reshape(out, n_groups, 1).float()
    zero = qzeros.reshape(out, n_groups, 1).float()
    w = (q - zero) * scale
    return w.reshape(out, in_features).to(scales.dtype)


def w4a16_gemm_ref(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Reference W4A16 linear: ``x @ dequantize(W).T``, matching ``nn.Linear(bias=False)``.

    ``x`` is ``[..., in]``; the result is ``[..., out]`` in ``x``'s dtype. This
    dequantizes the full weight then does a dense matmul — the *correctness*
    oracle, not the perf path (a fused kernel never materializes ``W``).
    """
    w = dequantize_w4a16(qweight, scales, qzeros, group_size).to(x.dtype)
    return x @ w.t()
