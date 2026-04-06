"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Chat Completions ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    model: Optional[str] = Field(None, description="Ollama model name (uses default if omitted)")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    stream: bool = Field(False, description="Enable streaming response")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    options: Optional[dict[str, Any]] = Field(None, description="Additional Ollama options")


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo = UsageInfo()


# ── Generation (simple prompt) ────────────────────────────────────

class GenerateRequest(BaseModel):
    model: Optional[str] = Field(None, description="Ollama model name")
    prompt: str = Field(..., description="Text prompt")
    stream: bool = False
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    options: Optional[dict[str, Any]] = None


class GenerateResponse(BaseModel):
    model: str
    response: str
    done: bool = True
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None


# ── Model Management ─────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    size: Optional[int] = None
    digest: Optional[str] = None
    modified_at: Optional[str] = None
    details: Optional[dict[str, Any]] = None


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class PullModelRequest(BaseModel):
    model: str = Field(..., description="Model name to pull, e.g. llama3.2")


class PullModelResponse(BaseModel):
    status: str
    model: str


class DeleteModelRequest(BaseModel):
    model: str = Field(..., description="Model name to delete")


# ── Health ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    ollama_url: str
    available_models: list[str] = []
    version: Optional[str] = None
