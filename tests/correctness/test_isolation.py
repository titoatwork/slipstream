"""EngineCore isolation: in-process wrapper + stub-child protocol. No weights."""

from __future__ import annotations

import ast
import os
import queue
import threading
from pathlib import Path

import pytest
from slipstream.core.config import EngineConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import Request, SchedulerOutput, Sequence, SequenceStatus
from slipstream.engine.isolated import (
    CMD_ABORT,
    CMD_ADD,
    CMD_OK,
    CMD_SHUTDOWN,
    CMD_STEP,
    IsolatedEngine,
    IsolationError,
    isolation_requested,
    make_engine_core,
    serve_commands,
    tokenize,
)


def _cfg() -> EngineConfig:
    return EngineConfig.for_model("dummy")


def _seq(seq_id: int = 1, tokens: list[int] | None = None, max_tokens: int = 2) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=tokens if tokens is not None else [1, 2, 3],
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )


class _FakeCore:
    """Minimal EngineCore stand-in for the command protocol."""

    def __init__(self) -> None:
        self._seq: Sequence | None = None

    def add_request(self, seq: Sequence) -> None:
        if not seq.prompt_token_ids:
            raise ValueError("Sequence needs prompt_token_ids")
        self._seq = seq
        seq.status = SequenceStatus.WAITING

    def step(self) -> SchedulerOutput:
        seq = self._seq
        if seq is None or seq.is_finished:
            return SchedulerOutput(
                scheduled_seqs=[],
                num_batched_tokens=0,
                blocks_to_swap_in={},
                blocks_to_swap_out={},
                blocks_to_copy={},
                is_prefill_chunk={},
            )
        seq.status = SequenceStatus.RUNNING
        seq.num_computed_tokens = seq.num_prompt_tokens
        seq.append_token(9)
        if seq.num_output_tokens >= seq.sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH
        return SchedulerOutput(
            scheduled_seqs=[seq],
            num_batched_tokens=1,
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
            is_prefill_chunk={seq.seq_id: False},
        )

    def abort(self, seq_id: int) -> None:
        if self._seq is not None and self._seq.seq_id == seq_id:
            self._seq.status = SequenceStatus.FINISHED_ABORTED


class _FakeEngine:
    """Duck-typed engine for real EngineCore.step without loading weights."""

    def __init__(self) -> None:
        self.config = _cfg()
        self.config.model.max_model_len = 64

    def make_cache(self, max_len: int) -> object:
        return object()

    def prefill(self, token_ids: list[int], cache: object) -> object:
        return "prefill"

    def decode(self, token_id: int, cache: object) -> object:
        return "decode"

    def append_sampled(
        self, seq: Sequence, logits: object, *, max_output: int | None = None
    ) -> int:
        seq.append_token(3)
        cap = max_output if max_output is not None else seq.sampling_params.max_tokens
        if seq.num_output_tokens >= cap:
            seq.status = SequenceStatus.FINISHED_LENGTH
        return 3

    def tokenize(self, request: Request) -> list[int]:
        if request.prompt_token_ids is None:
            raise ValueError("fake engine expects prompt_token_ids")
        return list(request.prompt_token_ids)


def _stub_child(cmd_q: object, resp_q: object, ready: object, config: EngineConfig) -> None:
    ready.set()  # type: ignore[union-attr]
    serve_commands(_FakeCore(), cmd_q, resp_q)


def _dead_child(cmd_q: object, resp_q: object, ready: object, config: EngineConfig) -> None:
    os._exit(1)


def test_isolation_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLIPSTREAM_ISOLATE", raising=False)
    assert isolation_requested(None) is False
    assert isolation_requested(False) is False
    assert isolation_requested(True) is True


def test_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLIPSTREAM_ISOLATE", "1")
    assert isolation_requested(None) is True
    monkeypatch.setenv("SLIPSTREAM_ISOLATE", "0")
    assert isolation_requested(None) is False


def test_inprocess_factory_wraps_core() -> None:
    wrapped = make_engine_core(_cfg(), isolate=False)
    assert isinstance(wrapped, IsolatedEngine)
    assert wrapped.isolated is False
    seq = _seq()
    wrapped.add_request(seq)
    assert seq.status is SequenceStatus.WAITING
    wrapped.abort(1)
    assert seq.status is SequenceStatus.FINISHED_ABORTED
    wrapped.shutdown()


def test_inprocess_add_step_abort() -> None:
    fake = _FakeEngine()
    eng = make_engine_core(fake.config, engine=fake, isolate=False)
    seq = _seq(max_tokens=2)
    eng.add_request(seq)
    out1 = eng.step()
    assert out1.scheduled_seqs[0].output_token_ids == [3]
    eng.step()
    assert seq.status is SequenceStatus.FINISHED_LENGTH
    seq2 = _seq(seq_id=2, tokens=[4, 5], max_tokens=4)
    eng.add_request(seq2)
    eng.abort(2)
    assert seq2.status is SequenceStatus.FINISHED_ABORTED
    empty = eng.step()
    assert empty.scheduled_seqs == []
    eng.shutdown()


def test_tokenize_stays_in_parent() -> None:
    req = Request(
        request_id="x",
        prompt=None,
        prompt_token_ids=[9, 8, 7],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        arrival_ts=0.0,
    )
    assert tokenize(req) == [9, 8, 7]
    fake = _FakeEngine()
    eng = make_engine_core(fake.config, engine=fake, isolate=False)
    assert eng.tokenize(req) == [9, 8, 7]


def test_command_protocol_stub_core() -> None:
    core = _FakeCore()
    cmd: queue.Queue[object] = queue.Queue()
    resp: queue.Queue[object] = queue.Queue()
    thread = threading.Thread(target=serve_commands, args=(core, cmd, resp), daemon=True)
    thread.start()
    seq = _seq(max_tokens=2)
    cmd.put((CMD_ADD, seq))
    assert resp.get(timeout=2.0)[0] == CMD_OK
    cmd.put((CMD_STEP, None))
    op, out = resp.get(timeout=2.0)
    assert op == CMD_STEP
    assert isinstance(out, SchedulerOutput)
    assert out.scheduled_seqs[0].output_token_ids == [9]
    cmd.put((CMD_ABORT, seq.seq_id))
    assert resp.get(timeout=2.0)[0] == CMD_OK
    assert seq.status is SequenceStatus.FINISHED_ABORTED
    cmd.put((CMD_SHUTDOWN, None))
    assert resp.get(timeout=2.0)[0] == CMD_OK
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_subprocess_stub_child_add_step_abort() -> None:
    eng = make_engine_core(_cfg(), isolate=True, child_target=_stub_child, ready_timeout_s=20.0)
    try:
        assert isinstance(eng, IsolatedEngine)
        assert eng.isolated is True
        seq = _seq(seq_id=7, max_tokens=2)
        eng.add_request(seq)
        assert seq.status is SequenceStatus.WAITING
        out = eng.step()
        assert seq.output_token_ids == [9]
        assert out.scheduled_seqs[0] is seq
        eng.abort(7)
        assert seq.status is SequenceStatus.FINISHED_ABORTED
    finally:
        eng.shutdown()


def test_isolate_with_engine_refuses_double_load() -> None:
    fake = _FakeEngine()
    with pytest.raises(IsolationError, match="twice"):
        make_engine_core(fake.config, engine=fake, isolate=True)


def test_spawn_failure_raises_no_fallback() -> None:
    with pytest.raises(IsolationError, match="SLIPSTREAM_ISOLATE"):
        make_engine_core(_cfg(), isolate=True, child_target=_dead_child, ready_timeout_s=4.0)


def test_isolated_module_body_avoids_cuda_imports() -> None:
    path = Path(__file__).resolve().parents[2] / "slipstream" / "engine" / "isolated.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"torch", "slipstream.engine.llm_engine"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] != "torch"
                assert alias.name != "slipstream.engine.llm_engine"
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            root = node.module.split(".", 1)[0]
            assert root != "torch"
            assert node.module not in forbidden
