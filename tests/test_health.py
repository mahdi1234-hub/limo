"""Tests for health endpoints."""


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "limo"
    assert "docs" in data
    assert data["status"] == "running"


def test_health_ollama_connected(client, mock_ollama):
    mock_ollama.ping.return_value = True
    mock_ollama.list_models.return_value = [
        {"name": "llama3.2", "size": 4_000_000_000},
        {"name": "mistral", "size": 4_500_000_000},
    ]
    mock_ollama.version.return_value = "0.3.0"

    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["ollama_connected"] is True
    assert "llama3.2" in data["available_models"]
    assert data["version"] == "0.3.0"


def test_health_ollama_disconnected_shows_fallback(client, mock_ollama):
    mock_ollama.ping.return_value = False

    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "fallback" in data["status"]
    assert data["ollama_connected"] is False
    # Fallback mode still shows available models
    assert len(data["available_models"]) > 0
