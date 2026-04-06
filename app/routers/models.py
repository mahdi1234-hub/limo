"""Model management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    DeleteModelRequest,
    ModelInfo,
    ModelListResponse,
    PullModelRequest,
    PullModelResponse,
)
from app.services.ollama import OLLAMA_MODELS, get_ollama_client

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all models available in the Ollama instance."""
    client = get_ollama_client()
    try:
        raw = await client.list_models()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc

    return ModelListResponse(
        models=[
            ModelInfo(
                name=m.get("name", ""),
                size=m.get("size"),
                digest=m.get("digest"),
                modified_at=m.get("modified_at"),
                details=m.get("details"),
            )
            for m in raw
        ]
    )


@router.post("/models/pull", response_model=PullModelResponse)
async def pull_model(req: PullModelRequest):
    """Pull a model from the Ollama registry."""
    client = get_ollama_client()
    try:
        await client.pull_model(req.model)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to pull {req.model}: {exc}") from exc

    return PullModelResponse(status="success", model=req.model)


@router.delete("/models", status_code=200)
async def delete_model(req: DeleteModelRequest):
    """Delete a model from the Ollama instance."""
    client = get_ollama_client()
    ok = await client.delete_model(req.model)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Model {req.model} not found")
    return {"status": "deleted", "model": req.model}


@router.post("/models/ensure")
async def ensure_all_models():
    """Pull all configured models that are not yet present."""
    client = get_ollama_client()
    try:
        results = await client.ensure_models(OLLAMA_MODELS)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"models": results}


@router.get("/models/{model_name}/info")
async def model_info(model_name: str):
    """Get detailed information about a specific model."""
    client = get_ollama_client()
    try:
        info = await client.show_model(model_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Model not found: {exc}") from exc
    return info
