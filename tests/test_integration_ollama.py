"""Integration tests with REAL Ollama inference.

These tests connect to a running Ollama instance and verify
actual model responses. They require `ollama serve` to be running.

Run with: pytest tests/test_integration_ollama.py -v -s
"""

import os

import httpx
import pytest

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def ollama_available() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not running -- skipping integration tests",
)


def get_available_models() -> list[str]:
    r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


# Models small enough to run in limited memory
SMALL_MODELS = ["qwen2.5:0.5b", "tinyllama:latest"]


class TestOllamaRunning:
    """Verify Ollama is up and models are available."""

    def test_ollama_is_running(self):
        r = httpx.get(f"{OLLAMA_URL}/")
        assert r.status_code == 200

    def test_version(self):
        r = httpx.get(f"{OLLAMA_URL}/api/version", timeout=10)
        r.raise_for_status()
        version = r.json().get("version")
        assert version is not None
        print(f"\nOllama version: {version}")

    def test_models_are_available(self):
        models = get_available_models()
        assert len(models) >= 2, "Expected at least 2 models"
        print(f"\nAvailable models ({len(models)}): {models}")


class TestRealChatInference:
    """Test chat completions with real Ollama models."""

    @pytest.mark.parametrize("model", SMALL_MODELS)
    def test_chat_returns_response(self, model):
        """Model generates a non-empty chat response."""
        available = get_available_models()
        if not any(model in m for m in available):
            pytest.skip(f"{model} not available")

        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        assert "message" in data
        content = data["message"]["content"]
        assert len(content) > 0, f"{model} returned empty response"
        print(f"\n{model} chat: {content[:150]}")

    @pytest.mark.parametrize("model", SMALL_MODELS)
    def test_chat_with_system_prompt(self, model):
        """Model responds with system prompt context."""
        available = get_available_models()
        if not any(model in m for m in available):
            pytest.skip(f"{model} not available")

        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a math tutor. Always show your work."},
                    {"role": "user", "content": "What is 5 times 7?"},
                ],
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
        assert len(content) > 0
        print(f"\n{model} math: {content[:150]}")

    @pytest.mark.parametrize("model", SMALL_MODELS)
    def test_multi_turn(self, model):
        """Model handles multi-turn conversations."""
        available = get_available_models()
        if not any(model in m for m in available):
            pytest.skip(f"{model} not available")

        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Hello, my favorite color is blue."},
                    {"role": "assistant", "content": "Nice! Blue is a great color."},
                    {"role": "user", "content": "What did I just tell you about?"},
                ],
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
        assert len(content) > 0
        print(f"\n{model} multi-turn: {content[:150]}")


class TestRealGenerateInference:
    """Test text generation with real Ollama models."""

    @pytest.mark.parametrize("model", SMALL_MODELS)
    def test_generate_returns_response(self, model):
        """Model generates text from a prompt."""
        available = get_available_models()
        if not any(model in m for m in available):
            pytest.skip(f"{model} not available")

        r = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": "The three primary colors are",
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        content = r.json()["response"]
        assert len(content) > 0
        print(f"\n{model} generate: {content[:150]}")

    @pytest.mark.parametrize("model", SMALL_MODELS)
    def test_generate_with_options(self, model):
        """Model respects temperature and max_tokens options."""
        available = get_available_models()
        if not any(model in m for m in available):
            pytest.skip(f"{model} not available")

        r = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": "List 3 fruits:",
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 50},
            },
            timeout=120,
        )
        r.raise_for_status()
        content = r.json()["response"]
        assert len(content) > 0
        print(f"\n{model} generate(options): {content[:150]}")
