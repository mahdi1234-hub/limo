"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ollama import OllamaClient


@pytest.fixture
def client():
    """Synchronous test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_ollama():
    """Patch the module-level Ollama client with a mock."""
    mock = AsyncMock(spec=OllamaClient)
    mock.base_url = "http://mock-ollama:11434"
    with patch("app.routers.health.get_ollama_client", return_value=mock), \
         patch("app.routers.chat.get_ollama_client", return_value=mock), \
         patch("app.routers.models.get_ollama_client", return_value=mock):
        yield mock
