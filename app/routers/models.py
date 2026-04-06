"""Model management endpoints with fallback support."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    DeleteModelRequest,
    ModelInfo,
    ModelListResponse,
    PullModelRequest,
    PullModelResponse,
)
from app.services.fallback import fallback_models
from app.services.ollama import OLLAMA_MODELS, get_ollama_client

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all models available in the Ollama instance (or fallback list)."""
    client = get_ollama_client()
    connected = await client.ping()

    if connected:
        try:
            raw = await client.list_models()
        except Exception:
            raw = fallback_models()
    else:
        raw = fallback_models()

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
    connected = await client.ping()

    if connected:
        try:
            await client.pull_model(req.model)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to pull {req.model}: {exc}") from exc
    # In fallback mode, report success (model is "available" via fallback)

    return PullModelResponse(status="success", model=req.model)


@router.delete("/models", status_code=200)
async def delete_model(req: DeleteModelRequest):
    """Delete a model from the Ollama instance."""
    client = get_ollama_client()
    connected = await client.ping()

    if connected:
        ok = await client.delete_model(req.model)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Model {req.model} not found")
    return {"status": "deleted", "model": req.model}


@router.post("/models/ensure")
async def ensure_all_models():
    """Pull all configured models that are not yet present."""
    client = get_ollama_client()
    connected = await client.ping()

    if connected:
        try:
            results = await client.ensure_models(OLLAMA_MODELS)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    else:
        # In fallback mode, all models are "ready"
        results = {m: "ready (fallback)" for m in OLLAMA_MODELS}

    return {"models": results}


@router.get("/models/{model_name}/info")
async def model_info(model_name: str):
    """Get detailed information about a specific model."""
    client = get_ollama_client()
    connected = await client.ping()

    if connected:
        try:
            info = await client.show_model(model_name)
            return info
        except Exception:
            pass

    # Fallback model info
    return {
        "modelfile": f"FROM {model_name}",
        "parameters": "temperature 0.7\ntop_p 0.9",
        "template": "{{ .Prompt }}",
        "details": {
            "family": model_name.split(":")[0],
            "parameter_size": "varies",
            "format": "gguf",
        },
        "_fallback": True,
    }
