"""Chat and generation endpoints with automatic fallback."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models.schemas import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    UsageInfo,
)
from app.services.fallback import fallback_chat, fallback_generate
from app.services.ollama import get_ollama_client

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    """OpenAI-compatible chat completions endpoint backed by Ollama.

    Automatically falls back to built-in responses when Ollama is unavailable.
    """
    settings = get_settings()
    client = get_ollama_client()
    model = req.model or settings.default_model

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    options = dict(req.options or {})
    if req.temperature is not None:
        options["temperature"] = req.temperature
    if req.top_p is not None:
        options["top_p"] = req.top_p
    if req.max_tokens is not None:
        options["num_predict"] = req.max_tokens

    # Try Ollama first, fall back if unavailable
    connected = await client.ping()
    if connected and not req.stream:
        try:
            data = await client.chat(model, messages, options=options or None)
        except Exception:
            data = fallback_chat(model, messages, options=options or None)
    elif connected and req.stream:
        return StreamingResponse(
            client.chat_stream(model, messages, options=options or None),
            media_type="text/event-stream",
        )
    else:
        data = fallback_chat(model, messages, options=options or None)

    assistant_msg = data.get("message", {})
    is_fallback = data.get("_fallback", False)

    return ChatResponse(
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role=assistant_msg.get("role", "assistant"),
                    content=assistant_msg.get("content", ""),
                ),
                finish_reason="fallback" if is_fallback else "stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        ),
    )


@router.post("/v1/generate", response_model=GenerateResponse, include_in_schema=False)
@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Simple text generation with automatic fallback."""
    settings = get_settings()
    client = get_ollama_client()
    model = req.model or settings.default_model

    options = dict(req.options or {})
    if req.temperature is not None:
        options["temperature"] = req.temperature
    if req.max_tokens is not None:
        options["num_predict"] = req.max_tokens

    connected = await client.ping()
    if connected:
        try:
            data = await client.generate(model, req.prompt, options=options or None)
        except Exception:
            data = fallback_generate(model, req.prompt, options=options or None)
    else:
        data = fallback_generate(model, req.prompt, options=options or None)

    return GenerateResponse(
        model=model,
        response=data.get("response", ""),
        done=data.get("done", True),
        total_duration=data.get("total_duration"),
        eval_count=data.get("eval_count"),
    )
