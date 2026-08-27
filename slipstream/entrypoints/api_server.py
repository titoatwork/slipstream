"""FastAPI OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from slipstream.core.config import EngineConfig
from slipstream.core.sampling_params import SamplingParams
from slipstream.core.types import Request as EngineRequest
from slipstream.engine.llm_engine import LLMEngine
from slipstream.entrypoints.openai_protocol import (
    ChatCompletionRequest,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Usage,
)

DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def _sampling(req: CompletionRequest | ChatCompletionRequest) -> SamplingParams:
    stops: tuple[str, ...] = ()
    if req.stop is None:
        stops = ()
    elif isinstance(req.stop, str):
        stops = (req.stop,)
    else:
        stops = tuple(req.stop)
    return SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        stop_strings=stops,
    )


def _chat_prompt(messages: list[Any]) -> str:
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{msg.role}: {msg.content}")
    parts.append("assistant:")
    return "\n".join(parts)


def create_app(engine: LLMEngine | None = None) -> FastAPI:
    app = FastAPI(title="Slipstream", version="0.0.0")
    app.state.engine = engine

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict[str, object]:
        eng: LLMEngine | None = app.state.engine
        model_id = eng.config.model.model_id if eng is not None else "slipstream"
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "slipstream"}],
        }

    @app.get("/metrics")
    def metrics() -> JSONResponse:
        eng: LLMEngine | None = app.state.engine
        if eng is None:
            return JSONResponse({"error": "no engine"}, status_code=503)
        snap = eng.metrics
        step = snap.last_step
        body = {
            "kv_utilization": snap.kv_utilization,
            "cache_hit_rate": step.cache_hit_rate if step else 0.0,
            "num_running": step.num_running if step else 0,
            "num_waiting": step.num_waiting if step else 0,
            "preemptions": step.preemptions if step else 0,
            "itl_s": list(eng.last_itl_s[-64:]),
        }
        return JSONResponse(body)

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, raw: Request) -> object:
        eng = _engine(app)
        prompt = body.prompt if isinstance(body.prompt, str) else None
        ids = body.prompt if isinstance(body.prompt, list) else None
        req = EngineRequest(
            request_id=f"cmpl-{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            prompt_token_ids=ids,
            sampling_params=_sampling(body),
            arrival_ts=time.time(),
        )
        if body.stream:
            return StreamingResponse(
                _stream_completion(eng, req, body.model, raw),
                media_type="text/event-stream",
            )
        tokens = await asyncio.to_thread(eng.generate, req)
        text = eng.tokenizer.decode(tokens)
        return CompletionResponse(
            id=req.request_id,
            model=body.model,
            choices=[CompletionChoice(text=text, finish_reason="stop")],
            usage=Usage(
                prompt_tokens=len(ids or []),
                completion_tokens=len(tokens),
                total_tokens=len(ids or []) + len(tokens),
            ),
        )

    @app.post("/v1/chat/completions")
    async def chat(body: ChatCompletionRequest, raw: Request) -> object:
        eng = _engine(app)
        prompt = _chat_prompt(body.messages)
        req = EngineRequest(
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=_sampling(body),
            arrival_ts=time.time(),
        )
        if body.stream:

            async def events() -> AsyncIterator[str]:
                async for line in _stream_chat(eng, req, body.model, raw):
                    yield line

            return StreamingResponse(events(), media_type="text/event-stream")
        tokens = await asyncio.to_thread(eng.generate, req)
        text = eng.tokenizer.decode(tokens)
        return {
            "id": req.request_id,
            "object": "chat.completion",
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(tokens),
                "total_tokens": len(tokens),
            },
        }

    @app.websocket("/ws/metrics")
    async def ws_metrics(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                eng: LLMEngine | None = app.state.engine
                payload: dict[str, object] = {"blocks": [], "metrics": {}}
                if eng is not None:
                    payload["blocks"] = eng.block_snapshot()
                    payload["metrics"] = {
                        "kv_utilization": eng.metrics.kv_utilization,
                        "hit_rate": (
                            eng.metrics.last_step.cache_hit_rate if eng.metrics.last_step else 0.0
                        ),
                        "running": (
                            eng.metrics.last_step.num_running if eng.metrics.last_step else 0
                        ),
                        "waiting": (
                            eng.metrics.last_step.num_waiting if eng.metrics.last_step else 0
                        ),
                    }
                await ws.send_json(payload)
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return

    if DASHBOARD_DIR.is_dir():
        app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dash")

    return app


def _engine(app: FastAPI) -> LLMEngine:
    eng = app.state.engine
    if not isinstance(eng, LLMEngine):
        raise RuntimeError("engine not attached")
    return eng


async def _stream_completion(
    engine: LLMEngine, req: EngineRequest, model: str, raw: Request
) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[int | None] = asyncio.Queue()

    def produce() -> None:
        try:
            for tok in engine.stream(req):
                loop.call_soon_threadsafe(queue.put_nowait, tok)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(asyncio.to_thread(produce))
    while True:
        item = await queue.get()
        if item is None:
            yield "data: [DONE]\n\n"
            return
        piece = engine.tokenizer.decode([item])
        payload = {
            "id": req.request_id,
            "object": "text_completion",
            "model": model,
            "choices": [{"index": 0, "text": piece, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"


async def _stream_chat(
    engine: LLMEngine, req: EngineRequest, model: str, raw: Request
) -> AsyncIterator[str]:
    async for line in _stream_completion(engine, req, model, raw):
        if line.startswith("data: ") and "[DONE]" not in line:
            data = json.loads(line[len("data: ") :])
            text = data["choices"][0]["text"]
            data["object"] = "chat.completion.chunk"
            data["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
            yield f"data: {json.dumps(data)}\n\n"
        else:
            yield line


def main() -> None:
    parser = argparse.ArgumentParser(description="Slipstream OpenAI-compatible server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    engine = LLMEngine(EngineConfig.for_model(args.model))
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
