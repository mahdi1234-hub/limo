"""Tests for model management endpoints."""


def test_list_models(client, mock_ollama):
    mock_ollama.list_models.return_value = [
        {"name": "llama3.2", "size": 4_000_000_000, "digest": "abc123"},
        {"name": "mistral", "size": 4_500_000_000, "digest": "def456"},
    ]

    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["models"]) == 2
    names = [m["name"] for m in data["models"]]
    assert "llama3.2" in names
    assert "mistral" in names


def test_pull_model(client, mock_ollama):
    mock_ollama.pull_model.return_value = {"status": "success"}

    r = client.post("/v1/models/pull", json={"model": "gemma2:2b"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["model"] == "gemma2:2b"


def test_pull_model_error(client, mock_ollama):
    mock_ollama.pull_model.side_effect = Exception("not found")

    r = client.post("/v1/models/pull", json={"model": "nonexistent"})
    assert r.status_code == 502


def test_delete_model(client, mock_ollama):
    mock_ollama.delete_model.return_value = True

    r = client.request("DELETE", "/v1/models", json={"model": "llama3.2"})
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"


def test_delete_model_not_found(client, mock_ollama):
    mock_ollama.delete_model.return_value = False

    r = client.request("DELETE", "/v1/models", json={"model": "missing"})
    assert r.status_code == 404


def test_ensure_all_models(client, mock_ollama):
    mock_ollama.ensure_models.return_value = {
        "llama3.2": "ready",
        "mistral": "pulled",
        "gemma2:2b": "pulled",
    }

    r = client.post("/v1/models/ensure")
    assert r.status_code == 200
    data = r.json()
    assert "llama3.2" in data["models"]
    assert data["models"]["llama3.2"] == "ready"


def test_model_info(client, mock_ollama):
    mock_ollama.show_model.return_value = {
        "modelfile": "FROM llama3.2",
        "parameters": "temperature 0.7",
        "template": "{{ .Prompt }}",
    }

    r = client.get("/v1/models/llama3.2/info")
    assert r.status_code == 200
    data = r.json()
    assert "modelfile" in data


def test_model_info_not_found(client, mock_ollama):
    mock_ollama.show_model.side_effect = Exception("not found")

    r = client.get("/v1/models/nonexistent/info")
    assert r.status_code == 404
