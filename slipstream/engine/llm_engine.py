"""In-process facade used by tests and the API server."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from slipstream.core.config import EngineConfig
from slipstream.core.types import AllocStatus, Request, Sequence, SequenceStatus
from slipstream.engine.engine_core import EngineCore
from slipstream.engine.model_runner import ModelRunner
from slipstream.engine.sampler import Sampler
from slipstream.memory.block_manager import BlockManagerImpl
from slipstream.memory.contiguous_cache import NaiveKVCache
from slipstream.memory.paged_cache import (
    allocate_kv_cache,
    build_paged_forward,
    estimate_num_gpu_blocks,
)
from slipstream.memory.prefix_cache import RadixPrefixCache
from slipstream.memory.swap import CpuSwapSpace
from slipstream.models.hf_config import apply_hf_config, load_hf_config, resolve_model_path
from slipstream.models.loader import load_model
from slipstream.models.tokenizer import Tokenizer
from slipstream.observability.metrics import MetricsSnapshot, RequestTrace, StepMetrics
from slipstream.scheduler.policies import get_policy
from slipstream.scheduler.replay import kv_uncomputed, needs_replay, replay_token_ids
from slipstream.scheduler.scheduler import Scheduler

_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def absolute_arrival(ts: float, now: float) -> float:
    """Relative offsets (W4-style) become wall-clock so starvation wait is real."""
    if ts >= 1_000_000_000.0:
        return ts
    return now + ts


def take_ready(pending: list[Sequence], now: float) -> list[Sequence]:
    """Pop sequences whose `arrival_ts` is due. `pending` stays sorted by arrival."""
    ready: list[Sequence] = []
    i = 0
    while i < len(pending) and pending[i].arrival_ts <= now:
        ready.append(pending[i])
        i += 1
    del pending[:i]
    return ready


def capped_new_tokens(prompt_len: int, max_tokens: int, max_model_len: int) -> int:
    """How many new tokens fit under the context window.

    Prevents NaiveKVCache overflow when ``prompt + max_tokens > max_model_len``.
    """
    if max_tokens < 1:
        return 0
    return max(0, min(max_tokens, max_model_len - prompt_len))


class LLMEngine:
    """Load a local checkpoint and run single-sequence prefill + decode."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        snapshot = resolve_model_path(config.model.model_id, revision=config.model.revision)
        loaded = load_hf_config(snapshot, model_id=config.model.model_id)
        apply_hf_config(config.model, loaded)
        self.torch_dtype = _DTYPE[config.model.dtype]
        self.tokenizer = Tokenizer(snapshot, model_type=config.model.model_type)
        self.model = load_model(snapshot, config.model, device=self.device, dtype=self.torch_dtype)
        self.model.eval()
        self.sampler = Sampler()
        self.runner = ModelRunner(
            config, model=self.model, device=self.device, dtype=self.torch_dtype
        )
        self.core = EngineCore(config, engine=self)
        self.block_manager: BlockManagerImpl | None = None
        self.kv_cache: torch.Tensor | None = None
        self.scheduler: Scheduler | None = None
        self.prefix_cache: RadixPrefixCache | None = None
        self.swap_space: CpuSwapSpace | None = None
        self.last_itl_s: list[float] = []
        self.last_ttft_s: list[float] = []
        self.last_request_traces: list[RequestTrace] = []
        self._traces: dict[int, RequestTrace] = {}
        self._last_emit: dict[int, float] = {}
        self._started_at: dict[int, float] = {}
        self.step_index = 0
        self.metrics = MetricsSnapshot()
        if self.config.cache.enable_paging:
            self._setup_paging()

    def generate(self, request: Request) -> list[int]:
        """Blocking greedy-or-sampled generation. Returns output token ids only."""
        with torch.inference_mode():
            if self.config.cache.enable_paging:
                return self._generate_paged(request)
            return self._generate(request)

    def generate_batch(self, requests: list[Request], *, inject: str = "all") -> list[list[int]]:
        """Continuous-batch several requests. Requires paging.

        `inject="all"` enqueues every request immediately (closed burst).
        `inject="arrival"` releases a request only when wall time reaches
        `arrival_ts` (open loop). Tests keep the default.
        """
        if inject not in {"all", "arrival"}:
            raise ValueError("inject must be 'all' or 'arrival'")
        if not self.config.cache.enable_paging or self.scheduler is None:
            return [self.generate(req) for req in requests]
        with torch.inference_mode():
            return self._generate_batch(requests, inject=inject)

    def stream(self, request: Request) -> Iterator[int]:
        """Token iterator. Same tokens as generate()."""
        if not self.config.cache.enable_paging:
            yield from self.generate(request)
            return
        with torch.inference_mode():
            yield from self._generate_paged_iter(request)

    def _generate(self, request: Request) -> list[int]:
        prompt_ids = self.tokenize(request)
        params = request.sampling_params
        max_model_len = self.config.model.max_model_len
        if not prompt_ids:
            raise ValueError("empty prompt")
        if len(prompt_ids) >= max_model_len:
            raise ValueError(
                f"prompt length {len(prompt_ids)} exceeds max_model_len {max_model_len}"
            )

        max_new = capped_new_tokens(len(prompt_ids), params.max_tokens, max_model_len)
        if max_new < 1:
            raise ValueError(
                f"prompt length {len(prompt_ids)} leaves no room under "
                f"max_model_len {max_model_len}"
            )
        max_len = len(prompt_ids) + max_new
        cache = self.make_cache(max_len)
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=prompt_ids,
            output_token_ids=[],
            status=SequenceStatus.RUNNING,
            sampling_params=params,
            arrival_ts=request.arrival_ts,
            request_id=request.request_id,
            slo_ttft_ms=request.slo_ttft_ms,
            slo_tpot_ms=request.slo_tpot_ms,
            num_computed_tokens=0,
        )

        logits = self.prefill(prompt_ids, cache)
        seq.num_computed_tokens = len(prompt_ids)
        self.append_sampled(seq, logits, max_output=max_new)
        while not seq.is_finished:
            logits = self.decode(seq.output_token_ids[-1], cache)
            seq.num_computed_tokens = seq.num_tokens
            self.append_sampled(seq, logits, max_output=max_new)
        return list(seq.output_token_ids)

    def tokenize(self, request: Request) -> list[int]:
        if request.prompt_token_ids is not None:
            return list(request.prompt_token_ids)
        if request.prompt is None:
            raise ValueError("Request needs prompt or prompt_token_ids")
        ids = self.tokenizer.encode(request.prompt, add_special_tokens=False)
        if self.tokenizer.add_bos_token:
            bos = self.config.model.bos_token_id
            if bos is not None and (not ids or ids[0] != bos):
                ids = [bos] + ids
        return ids

    def make_cache(self, max_len: int) -> NaiveKVCache:
        cfg = self.config.model
        if cfg.num_layers is None or cfg.num_kv_heads is None or cfg.head_dim is None:
            raise ValueError("ModelConfig is missing layer/head fields")
        return NaiveKVCache(
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            max_batch=1,
            max_len=max_len,
            dtype=self.torch_dtype,
            device=self.device,
        )

    def prefill(self, token_ids: list[int], cache: NaiveKVCache) -> torch.Tensor:
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        return self.runner.prefill(ids, cache)

    def decode(self, token_id: int, cache: NaiveKVCache) -> torch.Tensor:
        ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        return self.runner.decode(ids, cache)

    def append_sampled(
        self, seq: Sequence, logits: torch.Tensor, *, max_output: int | None = None
    ) -> int:
        token = self.sample_token(logits[:, -1, :], seq)
        seq.append_token(token)
        limit = max_output if max_output is not None else seq.sampling_params.max_tokens
        if self.is_stop(seq, token):
            seq.status = SequenceStatus.FINISHED_STOPPED
        elif seq.num_output_tokens >= limit or seq.num_tokens >= self.config.model.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH
        self._mark_emit(seq)
        return token

    def sample_token(self, logits: torch.Tensor, seq: Sequence) -> int:
        history = seq.prompt_token_ids + seq.output_token_ids
        token_ids = torch.tensor([history], dtype=torch.long, device=logits.device)
        sampled = self.sampler.sample(logits, seq.sampling_params, token_ids=token_ids)
        return int(sampled[0].item())

    def is_stop(self, seq: Sequence, token: int) -> bool:
        params = seq.sampling_params
        if token in params.stop_token_ids:
            return True
        eos = self.config.model.eos_token_id
        if not params.ignore_eos and eos is not None and token == eos:
            return True
        if params.stop_strings:
            text = self.tokenizer.decode(seq.output_token_ids)
            if any(stop and stop in text for stop in params.stop_strings):
                return True
        return False

    def _setup_paging(self) -> None:
        cfg = self.config.model
        if cfg.num_layers is None or cfg.num_kv_heads is None or cfg.head_dim is None:
            raise ValueError("ModelConfig is missing layer/head fields")
        block_size = self.config.cache.block_size
        if self.config.cache.num_gpu_blocks is not None:
            n_blocks = self.config.cache.num_gpu_blocks
        elif self.device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(self.device)
            n_blocks = estimate_num_gpu_blocks(
                num_layers=cfg.num_layers,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                block_size=block_size,
                dtype_bytes=cfg.dtype_bytes,
                available_bytes=int(free),
                gpu_memory_utilization=self.config.cache.gpu_memory_utilization,
            )
        else:
            per_seq = (cfg.max_model_len + block_size - 1) // block_size
            n_blocks = per_seq * min(self.config.scheduler.max_num_seqs, 16)
        self.block_manager = BlockManagerImpl(n_blocks, block_size, max_model_len=cfg.max_model_len)
        self.kv_cache = allocate_kv_cache(
            cfg.num_layers,
            n_blocks,
            block_size,
            cfg.num_kv_heads,
            cfg.head_dim,
            dtype=self.torch_dtype,
            device=self.device,
        )
        self.block_manager.bind_kv(self.kv_cache)
        block_bytes = (
            2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * cfg.dtype_bytes * block_size
        )
        sched_cfg = self.config.scheduler
        self.scheduler = Scheduler(
            sched_cfg,
            self.block_manager,
            get_policy(
                sched_cfg.policy,
                safety_factor=sched_cfg.safety_factor,
                starvation_guard_ms=sched_cfg.starvation_guard_ms,
            ),
            block_size=block_size,
            kv_bytes_per_block=block_bytes,
        )
        if self.config.cache.enable_prefix_caching:
            self.prefix_cache = RadixPrefixCache(block_size)
            self.prefix_cache.bind(self.block_manager)
            self.block_manager.bind_prefix(self.prefix_cache)
        n_cpu = self.config.cache.num_cpu_blocks
        if n_cpu is None:
            n_cpu = max(8, n_blocks // 4)
        self.swap_space = CpuSwapSpace(
            n_cpu,
            block_size,
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
            dtype=self.torch_dtype,
        )
        self.block_manager.bind_swap(self.swap_space)

    def _generate_paged(self, request: Request) -> list[int]:
        return list(self._generate_paged_iter(request))

    def _generate_paged_iter(self, request: Request) -> Iterator[int]:
        if self.block_manager is None or self.kv_cache is None:
            raise RuntimeError("paging is not initialized")
        prompt_ids = self.tokenize(request)
        params = request.sampling_params
        max_model_len = self.config.model.max_model_len
        if not prompt_ids:
            raise ValueError("empty prompt")
        if len(prompt_ids) >= max_model_len:
            raise ValueError(
                f"prompt length {len(prompt_ids)} exceeds max_model_len {max_model_len}"
            )
        max_new = capped_new_tokens(len(prompt_ids), params.max_tokens, max_model_len)
        if max_new < 1:
            raise ValueError("no room under max_model_len")

        seq = Sequence(
            seq_id=0,
            prompt_token_ids=prompt_ids,
            output_token_ids=[],
            status=SequenceStatus.RUNNING,
            sampling_params=params,
            arrival_ts=request.arrival_ts,
            request_id=request.request_id,
            slo_ttft_ms=request.slo_ttft_ms,
            slo_tpot_ms=request.slo_tpot_ms,
            num_computed_tokens=0,
        )
        self._apply_prefix(seq)
        if self.block_manager.can_allocate(seq) is not AllocStatus.OK:
            raise RuntimeError("not enough KV blocks for this request")
        self.block_manager.allocate(seq)

        try:
            uncached = prompt_ids[seq.num_computed_tokens :]
            if uncached:
                slots = [self.block_manager.append_slot(seq) for _ in uncached]
                ctx = build_paged_forward(
                    self.kv_cache, self.block_manager, [seq], [slots], self.device
                )
                ids = torch.tensor([uncached], dtype=torch.long, device=self.device)
                start = seq.num_computed_tokens
                positions = torch.arange(
                    start, start + len(uncached), device=self.device
                ).unsqueeze(0)
                logits = self.runner.run(ids, positions, ctx)
            else:
                logits = self._logits_from_cached_prompt(seq, prompt_ids)
            seq.num_computed_tokens = len(prompt_ids)
            self.append_sampled(seq, logits, max_output=max_new)
            yield seq.output_token_ids[-1]
            while not seq.is_finished:
                pair = self.block_manager.append_slot(seq)
                ctx = build_paged_forward(
                    self.kv_cache, self.block_manager, [seq], [[pair]], self.device
                )
                token = seq.output_token_ids[-1]
                ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
                pos = torch.tensor([[seq.num_tokens - 1]], dtype=torch.long, device=self.device)
                logits = self.runner.run(ids, pos, ctx)
                seq.num_computed_tokens = seq.num_tokens
                self.append_sampled(seq, logits, max_output=max_new)
                yield seq.output_token_ids[-1]
        finally:
            self._publish_prefix(seq)
            self.block_manager.free(seq)

    def _generate_batch(self, requests: list[Request], *, inject: str = "all") -> list[list[int]]:
        assert self.scheduler is not None and self.block_manager is not None
        assert self.kv_cache is not None
        seqs: list[Sequence] = []
        import time as _time

        batch_now = _time.time()
        batch_perf = _time.perf_counter()
        self.last_itl_s.clear()
        self.last_ttft_s.clear()
        self._traces.clear()
        self.last_request_traces = []
        pending: list[Sequence] = []
        for req in requests:
            prompt_ids = self.tokenize(req)
            max_new = capped_new_tokens(
                len(prompt_ids), req.sampling_params.max_tokens, self.config.model.max_model_len
            )
            arrival = absolute_arrival(req.arrival_ts, batch_now)
            seq = Sequence(
                seq_id=self.scheduler.next_seq_id(),
                prompt_token_ids=prompt_ids,
                output_token_ids=[],
                status=SequenceStatus.WAITING,
                sampling_params=req.sampling_params,
                arrival_ts=arrival,
                request_id=req.request_id,
                slo_ttft_ms=req.slo_ttft_ms,
                slo_tpot_ms=req.slo_tpot_ms,
                num_computed_tokens=0,
            )
            # Harness sets this to the true length; default is the output cap.
            seq.oracle_output_len = max_new
            self._apply_prefix(seq)
            # Closed burst: TTFT starts at enqueue. Open loop: at arrival.
            if inject == "all":
                self._started_at[seq.seq_id] = batch_perf
            else:
                self._started_at[seq.seq_id] = batch_perf + (arrival - batch_now)
            self._traces[seq.seq_id] = RequestTrace(
                request_id=req.request_id,
                seq_id=seq.seq_id,
                arrival_ts=arrival,
            )
            if inject == "all":
                self.scheduler.add_seq(seq)
            else:
                pending.append(seq)
            seqs.append(seq)
        pending.sort(key=lambda s: (s.arrival_ts, s.seq_id))

        steps = 0
        # Recompute may replay prompt+outputs more than once under pressure.
        limit = 1 + 8 * sum(
            len(s.prompt_token_ids) + (s.oracle_output_len or s.sampling_params.max_tokens)
            for s in seqs
        )
        while (
            self.scheduler.running or self.scheduler.waiting or self.scheduler.swapped or pending
        ) and steps < limit:
            now = _time.time()
            for seq in take_ready(pending, now):
                self.scheduler.add_seq(seq)
            if (
                not self.scheduler.running
                and not self.scheduler.waiting
                and not self.scheduler.swapped
                and pending
            ):
                delay = pending[0].arrival_ts - _time.time()
                if delay > 0:
                    _time.sleep(min(delay, 0.02))
                continue
            steps += 1
            planned = self.scheduler.schedule()
            if planned.scheduled_seqs:
                admit_now = _time.perf_counter()
                for sched_seq in planned.scheduled_seqs:
                    tr = self._traces.get(sched_seq.seq_id)
                    if tr is None or tr.admit_s is not None:
                        continue
                    start = self._started_at.get(sched_seq.seq_id, admit_now)
                    tr.admit_s = max(0.0, admit_now - start)
            if not planned.scheduled_seqs:
                finished_any = False
                for seq in list(self.scheduler.running):
                    if seq.is_finished:
                        self._publish_prefix(seq)
                        self.scheduler.finish(seq)
                        finished_any = True
                if finished_any:
                    continue
                if pending:
                    delay = pending[0].arrival_ts - _time.time()
                    if delay > 0:
                        _time.sleep(min(delay, 0.02))
                    continue
                if (
                    self.scheduler.waiting
                    and not self.scheduler.running
                    and not self.scheduler.swapped
                ):
                    for stuck in list(self.scheduler.waiting):
                        stuck.status = SequenceStatus.FINISHED_ABORTED
                        self.scheduler.finish(stuck)
                break
            decodes = [s for s in planned.scheduled_seqs if not needs_replay(s)]
            prefills = [s for s in planned.scheduled_seqs if needs_replay(s)]

            if decodes:
                self._run_paged_decodes(decodes)
            for seq in prefills:
                take = self.scheduler.last_take.get(seq.seq_id, kv_uncomputed(seq))
                take = max(1, min(take, kv_uncomputed(seq)))
                self._run_paged_prefill_chunk(seq, take)
            for seq in list(self.scheduler.running):
                if seq.is_finished:
                    self._publish_prefix(seq)
                    self.scheduler.finish(seq)
            self._record_step(
                planned.num_batched_tokens,
                len(decodes),
                sum(self.scheduler.last_take.get(s.seq_id, 0) for s in prefills),
            )

        done = _time.time()
        traces: list[RequestTrace] = []
        for seq in seqs:
            tr = self._traces.get(seq.seq_id)
            if tr is None:
                continue
            tr.output_len = seq.num_output_tokens
            tr.aborted = seq.status is SequenceStatus.FINISHED_ABORTED
            tr.finish_ts = done
            traces.append(tr)
        self.last_request_traces = traces
        return [list(s.output_token_ids) for s in seqs]

    def _run_paged_prefill_chunk(self, seq: Sequence, take: int) -> None:
        assert self.block_manager is not None and self.kv_cache is not None
        start = seq.num_computed_tokens
        replay = replay_token_ids(seq)
        tokens = replay[start : start + take]
        if not tokens:
            return
        slots: list[tuple[int, int]] = []
        for _ in tokens:
            pair = self._reserve_slot(seq)
            if pair is None:
                return
            slots.append(pair)
        kv_len = start + len(tokens)
        ctx = build_paged_forward(
            self.kv_cache,
            self.block_manager,
            [seq],
            [slots],
            self.device,
            seq_lens=[kv_len],
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(start, start + len(tokens), device=self.device).unsqueeze(0)
        logits = self.runner.run(ids, positions, ctx)
        seq.num_computed_tokens = start + len(tokens)
        # Only sample after the original prompt; a recompute catch-up of
        # existing outputs must resume via decode of the last output token.
        if seq.num_output_tokens == 0 and not needs_replay(seq):
            cap = seq.oracle_output_len or seq.sampling_params.max_tokens
            self.append_sampled(seq, logits, max_output=cap)

    def _run_paged_decodes(self, seqs: list[Sequence]) -> None:
        assert self.block_manager is not None and self.kv_cache is not None
        live: list[Sequence] = []
        new_slots: list[list[tuple[int, int]]] = []
        tokens: list[int] = []
        positions: list[int] = []
        protected: set[int] = set()
        for seq in seqs:
            if self.scheduler is not None and seq not in self.scheduler.running:
                continue
            pair = self._reserve_slot(seq, protected)
            if pair is None:
                continue
            protected.add(seq.seq_id)
            live.append(seq)
            new_slots.append([pair])
            tokens.append(seq.output_token_ids[-1])
            positions.append(seq.num_tokens - 1)
        if not live:
            return
        ctx = build_paged_forward(self.kv_cache, self.block_manager, live, new_slots, self.device)
        ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(-1)
        pos = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(-1)
        logits = self.runner.run(ids, pos, ctx)
        for i, seq in enumerate(live):
            cap = seq.oracle_output_len or seq.sampling_params.max_tokens
            self.append_sampled(seq, logits[i : i + 1], max_output=cap)
            seq.num_computed_tokens = seq.num_tokens

    def _reserve_slot(
        self, seq: Sequence, protected: set[int] | None = None
    ) -> tuple[int, int] | None:
        if self.scheduler is not None:
            return self.scheduler.ensure_slot(seq, protected)
        assert self.block_manager is not None
        return self.block_manager.append_slot(seq)

    def _mark_emit(self, seq: Sequence) -> None:
        import time as _time

        now = _time.perf_counter()
        prev = self._last_emit.get(seq.seq_id)
        trace = self._traces.get(seq.seq_id)
        if prev is None:
            start = self._started_at.get(seq.seq_id, now)
            ttft = max(0.0, now - start)
            self.last_ttft_s.append(ttft)
            seq.first_token_ts = now
            if trace is not None:
                trace.ttft_s = ttft
                wait = trace.admit_s if trace.admit_s is not None else 0.0
                trace.wait_s = wait
                trace.prefill_s = max(0.0, ttft - wait)
        else:
            itl = now - prev
            self.last_itl_s.append(itl)
            if trace is not None:
                trace.itls.append(itl)
        self._last_emit[seq.seq_id] = now

    def _apply_prefix(self, seq: Sequence) -> None:
        if self.prefix_cache is None or self.block_manager is None:
            return
        blocks, n = self.prefix_cache.match(seq.prompt_token_ids)
        # Exact full-prompt hits still need last-token logits; skip those
        # and only reuse a proper uncached suffix (W3-style system prompts).
        if n <= 0 or n >= len(seq.prompt_token_ids):
            return
        self.block_manager.attach(seq, blocks)
        seq.num_cached_tokens = n
        seq.num_computed_tokens = n

    def _publish_prefix(self, seq: Sequence) -> None:
        if self.prefix_cache is None:
            return
        self.prefix_cache.insert(seq.prompt_token_ids, seq.block_table)

    def _logits_from_cached_prompt(self, seq: Sequence, prompt_ids: list[int]) -> torch.Tensor:
        assert self.block_manager is not None and self.kv_cache is not None
        from slipstream.memory.paged_cache import PagedForward, pad_block_tables

        tables = pad_block_tables([seq.block_table], max(len(seq.block_table), 1), self.device)
        ctx = PagedForward(
            kv=self.kv_cache,
            block_tables=tables,
            seq_lens=torch.tensor([len(prompt_ids)], dtype=torch.int32, device=self.device),
            slot_mapping=torch.zeros(0, dtype=torch.int64, device=self.device),
            query_lens=torch.tensor([1], dtype=torch.int32, device=self.device),
            block_size=self.block_manager.block_size,
            write_kv=False,
        )
        last = prompt_ids[-1]
        ids = torch.tensor([[last]], dtype=torch.long, device=self.device)
        positions = torch.tensor([[len(prompt_ids) - 1]], device=self.device)
        return self.runner.run(ids, positions, ctx)

    def _record_step(self, batched: int, n_decode: int, n_prefill: int) -> None:
        self.step_index += 1
        free = self.block_manager.get_num_free_blocks() if self.block_manager else 0
        total = getattr(self.block_manager, "num_gpu_blocks", 1) or 1
        hit = self.prefix_cache.hit_rate() if self.prefix_cache else 0.0
        step = StepMetrics(
            step=self.step_index,
            num_running=len(self.scheduler.running) if self.scheduler else 0,
            num_waiting=len(self.scheduler.waiting) if self.scheduler else 0,
            num_swapped=len(self.scheduler.swapped) if self.scheduler else 0,
            batch_size=n_decode + (1 if n_prefill else 0),
            num_batched_tokens=batched,
            kv_utilization=1.0 - (free / total),
            prefill_tokens=n_prefill,
            decode_tokens=n_decode,
            step_time_ms=(self.last_itl_s[-1] * 1000.0) if self.last_itl_s else 0.0,
            cache_hit_rate=hit,
            preemptions=self.scheduler.preemptions if self.scheduler else 0,
        )
        self.metrics.last_step = step
        self.metrics.kv_utilization = step.kv_utilization

    def block_snapshot(self) -> list[dict[str, int]]:
        if self.block_manager is None:
            return []
        out: list[dict[str, int]] = []
        for b in self.block_manager.blocks:
            out.append({"id": b.block_id, "ref": b.ref_count, "tokens": b.num_tokens})
        return out
