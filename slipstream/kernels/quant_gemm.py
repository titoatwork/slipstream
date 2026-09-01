"""W4A16 dequant-fused GEMM. Phase 5 (A3, S9).

Public entry point for weight-only INT4 linear. The **weight format** and the
**numeric oracle** live in ``quant_ref`` (``quantize_w4a16`` / ``dequantize_w4a16``
/ ``w4a16_gemm_ref``); this module is what the engine calls.

Current implementation is the correct, device-agnostic PyTorch path (unpack →
dequantize → dense matmul), identical in result to ``w4a16_gemm_ref``. The
performance win — a Triton kernel that **fuses dequant into the GEMM epilogue so
the fp weight is never materialized in HBM** — is the GPU follow-up (T0/A100);
until then this path is numerically the source of truth and safe on any device.
The speedup it targets is a *memory-traffic* win (~4× decode, ~0 prefill;
MASTERPLAN §S9), measured in benchmarks/, not asserted here.
"""

from __future__ import annotations

import torch

from slipstream.kernels.quant_ref import w4a16_gemm_ref


def w4a16_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """W4A16 linear ``x @ dequantize(W).T`` (no bias).

    Args:
        x: Activations ``[..., in]`` (fp16/bf16/fp32).
        qweight: Packed int4 weights ``[out, in//8]`` int32 (see ``quant_ref``).
        scales: Group scales ``[out, in//group_size]``.
        qzeros: Group zero-points ``[out, in//group_size]`` int32 (0..15).
        group_size: Quantization group size (multiple of 8; divides ``in``).

    Returns:
        ``[..., out]`` in ``x``'s dtype.
    """
    return w4a16_gemm_ref(x, qweight, scales, qzeros, group_size)
