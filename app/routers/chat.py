"""Chat and generation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
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
from app.services.ollama import get_ollama_client

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    """OpenAI-compatible chat completions endpoint backed by Ollama."""
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

    # Streaming path
    if req.stream:
        return StreamingResponse(
            client.chat_stream(model, messages, options=options or None),
            media_type="text/event-stream",
        )

    # Non-streaming path
    try:
        data = await client.chat(model, messages, options=options or None)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc

    assistant_msg = data.get("message", {})
    return ChatResponse(
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role=assistant_msg.get("role", "assistant"),
                    content=assistant_msg.get("content", ""),
                ),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        ),
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Simple text generation endpoint."""
    settings = get_settings()
    client = get_ollama_client()
    model = req.model or settings.default_model

    options = dict(req.options or {})
    if req.temperature is not None:
        options["temperature"] = req.temperature
    if req.max_tokens is not None:
        options["num_predict"] = req.max_tokens

    try:
        data = await client.generate(model, req.prompt, options=options or None)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc

    return GenerateResponse(
        model=model,
        response=data.get("response", ""),
        done=data.get("done", True),
        total_duration=data.get("total_duration"),
        eval_count=data.get("eval_count"),
    )
