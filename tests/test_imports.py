"""Gate 0: every agent can import every interface they depend on."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "slipstream",
    "slipstream.core",
    "slipstream.core.types",
    "slipstream.core.config",
    "slipstream.core.sampling_params",
    "slipstream.core.sequence",
    "slipstream.core.debug",
    "slipstream.memory",
    "slipstream.memory.block_manager",
    "slipstream.memory.block_table",
    "slipstream.memory.prefix_cache",
    "slipstream.memory.swap",
    "slipstream.kernels",
    "slipstream.kernels.paged_attention",
    "slipstream.kernels.reshape_and_cache",
    "slipstream.kernels.fused_rmsnorm",
    "slipstream.kernels.fused_rope",
    "slipstream.kernels.fused_swiglu",
    "slipstream.kernels.quant_gemm",
    "slipstream.scheduler",
    "slipstream.scheduler.scheduler",
    "slipstream.scheduler.policies",
    "slipstream.scheduler.policies.base",
    "slipstream.scheduler.policies.fcfs",
    "slipstream.scheduler.policies.horizon",
    "slipstream.scheduler.policies.oracle",
    "slipstream.scheduler.predictor",
    "slipstream.scheduler.predictor.features",
    "slipstream.scheduler.predictor.length_model",
    "slipstream.models",
    "slipstream.models.llama",
    "slipstream.models.qwen",
    "slipstream.models.loader",
    "slipstream.models.layers",
    "slipstream.engine",
    "slipstream.engine.engine_core",
    "slipstream.engine.model_runner",
    "slipstream.engine.llm_engine",
    "slipstream.engine.cuda_graph",
    "slipstream.engine.isolated",
    "slipstream.engine.sampler",
    "slipstream.entrypoints",
    "slipstream.entrypoints.api_server",
    "slipstream.entrypoints.openai_protocol",
    "slipstream.observability",
    "slipstream.observability.goodput",
    "slipstream.observability.metrics",
    "slipstream.observability.ws_stream",
    "slipstream.distributed",
    "slipstream.distributed.tensor_parallel",
    "slipstream.distributed.communication",
    "slipstream.distributed.disaggregated",
    "slipstream.speculative",
]


@pytest.mark.parametrize("mod", MODULES)
def test_import(mod: str) -> None:
    importlib.import_module(mod)
