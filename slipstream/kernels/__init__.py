"""Triton kernels (A3). Implementations land in Phases 2–5.

P0: paged_attention_decode, paged_attention_prefill, reshape_and_cache
P1: fused_rmsnorm, fused_rope, fused_swiglu, copy/swap_blocks
P2: sampling, quant_gemm

Fused P1 ops stay eager unless SLIPSTREAM_FUSED=1 (Triton, CUDA only).
"""
