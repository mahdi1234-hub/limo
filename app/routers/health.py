"""Health-check and status endpoints."""

from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.services.ollama import get_ollama_client

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health including Ollama connectivity."""
    client = get_ollama_client()
    connected = await client.ping()

    models: list[str] = []
    version = None
    if connected:
        try:
            raw = await client.list_models()
            models = [m.get("name", "") for m in raw]
        except Exception:
            pass
        version = await client.version()

    return HealthResponse(
        status="ok" if connected else "degraded",
        ollama_connected=connected,
        ollama_url=client.base_url,
        available_models=models,
        version=version,
    )


@router.get("/")
async def root():
    return {
        "service": "limo",
        "description": "FastAPI gateway for Ollama LLM models",
        "docs": "/docs",
    }
