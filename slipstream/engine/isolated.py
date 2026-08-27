"""Optional EngineCore process isolation.

Default is in-process. T0 is an RTX 3050 6GB + 7.4 GiB RAM: loading the
checkpoint twice OOMs. Isolation is opt-in via SLIPSTREAM_ISOLATE=1.

Tokenize stays in the parent. The GPU child only runs ADD/STEP/ABORT.
LLMEngine is imported only inside the child target.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from slipstream.core.config import EngineConfig
from slipstream.core.types import Request, SchedulerOutput, Sequence, SequenceStatus
from slipstream.engine.engine_core import EngineCore

CMD_ADD = "ADD"
CMD_STEP = "STEP"
CMD_ABORT = "ABORT"
CMD_SHUTDOWN = "SHUTDOWN"
CMD_OK = "OK"
CMD_ERR = "ERR"

# T0 (docs/environment.md): 6144 MiB VRAM, 7.4 GiB host RAM.
# Isolation stays opt-in until a larger host measures a step-loop CPU win
# (MASTERPLAN §8.5). Do not flip the default on this machine.
ISOLATION_JUSTIFICATION = (
    "T0 (RTX 3050 6GB + 7.4 GiB RAM) OOMs if the checkpoint is loaded twice. "
    "Default is in-process with the same add/step/abort API. "
    "Set SLIPSTREAM_ISOLATE=1 only on a host that can hold two copies, and "
    "report step-loop CPU occupancy before/after."
)

ChildMain = Callable[[Any, Any, Any, EngineConfig], None]


class IsolationError(RuntimeError):
    """Isolation was requested and the GPU child did not start."""


@runtime_checkable
class IsolatedEngineAPI(Protocol):
    """EngineCore surface plus parent-side tokenize/shutdown."""

    def add_request(self, seq: Sequence) -> None: ...

    def step(self) -> SchedulerOutput: ...

    def abort(self, seq_id: int) -> None: ...

    def shutdown(self) -> None: ...

    def tokenize(self, request: Request) -> list[int]: ...


def isolation_requested(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("SLIPSTREAM_ISOLATE", "").strip().lower() in {"1", "true", "yes"}


def tokenize(
    request: Request,
    *,
    engine: Any | None = None,
    tokenizer: Any | None = None,
    config: EngineConfig | None = None,
) -> list[int]:
    """Parent-side tokenize. Never call this from the GPU child loop."""
    if engine is not None:
        return list(engine.tokenize(request))
    if request.prompt_token_ids is not None:
        return list(request.prompt_token_ids)
    if request.prompt is None:
        raise ValueError("Request needs prompt or prompt_token_ids")
    if tokenizer is None:
        if config is None:
            raise ValueError("tokenize needs prompt_token_ids, tokenizer, or config")
        tokenizer = _parent_tokenizer(config)
    ids = list(tokenizer.encode(request.prompt, add_special_tokens=False))
    add_bos = bool(getattr(tokenizer, "add_bos_token", False))
    bos = config.model.bos_token_id if config is not None else None
    if bos is None:
        bos = getattr(tokenizer, "bos_token_id", None)
    if add_bos and bos is not None and (not ids or ids[0] != bos):
        ids = [int(bos)] + ids
    return ids


_tokenize = tokenize


def empty_scheduler_output() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_seqs=[],
        num_batched_tokens=0,
        blocks_to_swap_in={},
        blocks_to_swap_out={},
        blocks_to_copy={},
        is_prefill_chunk={},
    )


def serve_commands(core: Any, cmd_q: Any, resp_q: Any) -> None:
    """Child command loop. `core` is EngineCore or a stub with the same API."""
    while True:
        msg = cmd_q.get()
        if not isinstance(msg, tuple) or not msg:
            resp_q.put((CMD_ERR, "malformed command"))
            continue
        op = msg[0]
        payload = msg[1] if len(msg) > 1 else None
        try:
            if op == CMD_ADD:
                core.add_request(payload)
                resp_q.put((CMD_OK, None))
            elif op == CMD_STEP:
                resp_q.put((CMD_STEP, core.step()))
            elif op == CMD_ABORT:
                if payload is None:
                    raise ValueError("ABORT missing seq_id")
                core.abort(int(payload))
                resp_q.put((CMD_OK, None))
            elif op == CMD_SHUTDOWN:
                resp_q.put((CMD_OK, None))
                return
            else:
                resp_q.put((CMD_ERR, f"unknown command {op!r}"))
        except Exception as exc:
            resp_q.put((CMD_ERR, f"{type(exc).__name__}: {exc}"))


def _parent_tokenizer(config: EngineConfig) -> Any:
    from slipstream.models.hf_config import resolve_model_path
    from slipstream.models.tokenizer import Tokenizer

    snapshot = resolve_model_path(config.model.model_id, revision=config.model.revision)
    return Tokenizer(snapshot, model_type=config.model.model_type)


def _sync_seq(src: Sequence, dst: Sequence) -> None:
    dst.prompt_token_ids = src.prompt_token_ids
    dst.output_token_ids = src.output_token_ids
    dst.block_table = src.block_table
    dst.status = src.status
    dst.num_cached_tokens = src.num_cached_tokens
    dst.num_computed_tokens = src.num_computed_tokens
    dst.predicted_remaining = src.predicted_remaining
    dst.first_token_ts = src.first_token_ts
    dst.oracle_output_len = src.oracle_output_len


def _gpu_child_main(cmd_q: Any, resp_q: Any, ready: Any, config: EngineConfig) -> None:
    """Spawn target. Constructs LLMEngine in this process only."""
    try:
        from slipstream.engine.llm_engine import LLMEngine

        engine = LLMEngine(config)
        core: Any = engine.core
    except Exception as exc:
        with contextlib.suppress(Exception):
            resp_q.put((CMD_ERR, f"child init failed: {type(exc).__name__}: {exc}"))
        return
    ready.set()
    serve_commands(core, cmd_q, resp_q)


class SubprocessEngine:
    """Parent-side proxy. Does not import torch or construct LLMEngine."""

    isolated = True

    def __init__(
        self,
        config: EngineConfig,
        *,
        child_target: ChildMain | None = None,
        ready_timeout_s: float = 120.0,
        rpc_timeout_s: float = 30.0,
    ) -> None:
        self.config = config
        self._seqs: dict[int, Sequence] = {}
        self._proc: Any | None = None
        self._cmd_q: Any | None = None
        self._resp_q: Any | None = None
        self._ready: Any | None = None
        self._closed = False
        self._rpc_timeout_s = rpc_timeout_s
        target = child_target if child_target is not None else _gpu_child_main
        try:
            ctx = mp.get_context("spawn")
            self._cmd_q = ctx.Queue()
            self._resp_q = ctx.Queue()
            self._ready = ctx.Event()
            self._proc = ctx.Process(
                target=target,
                args=(self._cmd_q, self._resp_q, self._ready, config),
                name="slipstream-engine-core",
                daemon=True,
            )
            self._proc.start()
            self._wait_ready(ready_timeout_s)
        except IsolationError:
            self.shutdown()
            raise
        except Exception as exc:
            self.shutdown()
            raise IsolationError(f"SLIPSTREAM_ISOLATE requested but spawn failed: {exc}") from exc

    def add_request(self, seq: Sequence) -> None:
        self._seqs[seq.seq_id] = seq
        self._rpc(CMD_ADD, seq)
        seq.status = SequenceStatus.WAITING

    def step(self) -> SchedulerOutput:
        out = self._rpc(CMD_STEP)
        if not isinstance(out, SchedulerOutput):
            raise IsolationError(f"child STEP returned {type(out).__name__}")
        merged: list[Sequence] = []
        for child_seq in out.scheduled_seqs:
            parent = self._seqs.get(child_seq.seq_id)
            if parent is None:
                self._seqs[child_seq.seq_id] = child_seq
                merged.append(child_seq)
            else:
                _sync_seq(child_seq, parent)
                merged.append(parent)
        out.scheduled_seqs = merged
        return out

    def abort(self, seq_id: int) -> None:
        self._rpc(CMD_ABORT, seq_id)
        local = self._seqs.get(seq_id)
        if local is not None:
            local.status = SequenceStatus.FINISHED_ABORTED

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        try:
            if proc is not None and proc.is_alive() and self._cmd_q is not None:
                try:
                    self._cmd_q.put((CMD_SHUTDOWN, None), timeout=0.5)
                    if self._resp_q is not None:
                        self._resp_q.get(timeout=2.0)
                except Exception:
                    pass
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            if proc is not None and proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        finally:
            self._proc = None

    def _wait_ready(self, timeout_s: float) -> None:
        assert self._ready is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._ready.is_set():
                return
            if self._proc is not None and not self._proc.is_alive():
                err = self._drain_err()
                raise IsolationError(
                    "SLIPSTREAM_ISOLATE requested but EngineCore child died during start"
                    + (f": {err}" if err else "")
                )
            time.sleep(0.01)
        err = self._drain_err()
        raise IsolationError(
            "SLIPSTREAM_ISOLATE requested but EngineCore child did not become ready"
            + (f": {err}" if err else "")
        )

    def _rpc(self, op: str, payload: object = None) -> object:
        if self._proc is None or not self._proc.is_alive() or self._cmd_q is None:
            raise IsolationError("EngineCore child is not running")
        assert self._resp_q is not None
        self._cmd_q.put((op, payload))
        try:
            reply = self._resp_q.get(timeout=self._rpc_timeout_s)
        except queue.Empty as exc:
            raise IsolationError(f"EngineCore child timed out on {op}") from exc
        if not isinstance(reply, tuple) or not reply:
            raise IsolationError(f"bad child reply: {reply!r}")
        if reply[0] == CMD_ERR:
            raise IsolationError(str(reply[1] if len(reply) > 1 else "child error"))
        return reply[1] if len(reply) > 1 else None

    def _drain_err(self) -> str | None:
        if self._resp_q is None:
            return None
        try:
            reply = self._resp_q.get_nowait()
        except Exception:
            return None
        if isinstance(reply, tuple) and reply:
            if len(reply) > 1:
                return str(reply[1])
            return str(reply[0])
        return str(reply)


class IsolatedEngine:
    """Same add/step/abort API as EngineCore. Tokenize stays in this process."""

    def __init__(
        self,
        backend: Any,
        *,
        engine: Any | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self._backend = backend
        self._engine = engine
        self.config = config if config is not None else getattr(backend, "config", None)

    @property
    def isolated(self) -> bool:
        return bool(getattr(self._backend, "isolated", False))

    def add_request(self, seq: Sequence) -> None:
        self._backend.add_request(seq)

    def step(self) -> SchedulerOutput:
        return self._backend.step()  # type: ignore[no-any-return]

    def abort(self, seq_id: int) -> None:
        self._backend.abort(seq_id)

    def shutdown(self) -> None:
        close = getattr(self._backend, "shutdown", None)
        if close is not None:
            close()

    def tokenize(self, request: Request) -> list[int]:
        return _tokenize(request, engine=self._engine, config=self.config)

    def __enter__(self) -> IsolatedEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def make_engine_core(
    config: EngineConfig,
    engine: Any | None = None,
    isolate: bool | None = None,
    *,
    child_target: ChildMain | None = None,
    ready_timeout_s: float = 120.0,
) -> IsolatedEngine:
    """In-process EngineCore wrapper, or a spawned GPU child if isolate is on.

    `isolate=None` reads SLIPSTREAM_ISOLATE. Default is in-process.
    Spawn failure raises IsolationError; it does not fall back.
    """
    if isolation_requested(isolate):
        if engine is not None:
            raise IsolationError(
                "isolate=True cannot take an in-process engine; that loads the model twice"
            )
        backend = SubprocessEngine(
            config,
            child_target=child_target,
            ready_timeout_s=ready_timeout_s,
        )
        return IsolatedEngine(backend, config=config)
    return IsolatedEngine(EngineCore(config, engine=engine), engine=engine, config=config)
