"""OpenAI REST schema (completions + chat). Phase 3 (A5)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[int]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    seed: int | None = None
    stop: str | list[str] | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    seed: int | None = None
    stop: str | list[str] | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Field(...)
