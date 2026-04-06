"""Ollama HTTP client service.

Wraps all communication with the Ollama REST API so the rest of the
application never builds raw HTTP requests itself.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Comprehensive list of popular Ollama models to auto-pull
OLLAMA_MODELS = [
    "llama3.2",
    "llama3.2:1b",
    "mistral",
    "gemma2:2b",
    "qwen2.5:0.5b",
    "phi3:mini",
    "tinyllama",
    "codellama:7b",
    "deepseek-coder:1.3b",
    "nomic-embed-text",
]


class OllamaClient:
    """Async client for the Ollama HTTP API."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        self.base_url = (base_url or get_settings().ollama_base_url).rstrip("/")
        self.timeout = timeout

    # ── helpers ───────────────────────────────────────────────────

    def _client(self, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            **kwargs,
        )

    # ── health / version ─────────────────────────────────────────

    async def ping(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            async with self._client() as client:
                r = await client.get("/")
                return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def version(self) -> Optional[str]:
        try:
            async with self._client() as client:
                r = await client.get("/api/version")
                r.raise_for_status()
                return r.json().get("version")
        except Exception:
            return None

    # ── model management ─────────────────────────────────────────

    async def list_models(self) -> list[dict[str, Any]]:
        async with self._client() as client:
            r = await client.get("/api/tags")
            r.raise_for_status()
            return r.json().get("models", [])

    async def pull_model(self, model: str) -> dict[str, Any]:
        """Pull a model from the Ollama library."""
        logger.info("Pulling model %s ...", model)
        async with self._client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            r = await client.post("/api/pull", json={"name": model, "stream": False})
            r.raise_for_status()
            return r.json()

    async def delete_model(self, model: str) -> bool:
        async with self._client() as client:
            r = await client.delete("/api/delete", json={"name": model})
            return r.status_code == 200

    async def show_model(self, model: str) -> dict[str, Any]:
        async with self._client() as client:
            r = await client.post("/api/show", json={"name": model})
            r.raise_for_status()
            return r.json()

    # ── chat completion ──────────────────────────────────────────

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options

        async with self._client() as client:
            r = await client.post("/api/chat", json=payload)
            r.raise_for_status()
            return r.json()

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[bytes]:
        """Streaming chat completion -- yields raw JSON lines."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options

        async with self._client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        yield (line + "\n").encode()

    # ── simple generate ──────────────────────────────────────────

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        async with self._client() as client:
            r = await client.post("/api/generate", json=payload)
            r.raise_for_status()
            return r.json()

    # ── bulk model setup ─────────────────────────────────────────

    async def ensure_models(self, models: Optional[list[str]] = None) -> dict[str, str]:
        """Pull every model in *models* that isn't already present.

        Returns a dict mapping model name -> status ('ready' | 'pulled' | 'error').
        """
        models = models or OLLAMA_MODELS
        existing = {m["name"].split(":")[0] for m in await self.list_models()}

        results: dict[str, str] = {}
        for model in models:
            base = model.split(":")[0]
            if base in existing:
                results[model] = "ready"
                continue
            try:
                await self.pull_model(model)
                results[model] = "pulled"
            except Exception as exc:
                logger.warning("Failed to pull %s: %s", model, exc)
                results[model] = f"error: {exc}"
        return results


# Module-level singleton
_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
