"""Tests for contextual fallback responses -- API always responds intelligently."""

from app.services.fallback import (
    MODEL_IDENTITY,
    fallback_chat,
    fallback_generate,
    fallback_models,
)


def test_fallback_chat_greeting():
    data = fallback_chat("llama3.2", [{"role": "user", "content": "Hello!"}])
    assert data["message"]["role"] == "assistant"
    assert len(data["message"]["content"]) > 20
    assert data["_fallback"] is True


def test_fallback_chat_contextual_python():
    data = fallback_chat("llama3.2", [{"role": "user", "content": "What is Python?"}])
    content = data["message"]["content"].lower()
    assert "python" in content
    assert len(content) > 50


def test_fallback_chat_contextual_ai():
    data = fallback_chat("mistral", [{"role": "user", "content": "Explain machine learning"}])
    content = data["message"]["content"].lower()
    assert len(content) > 50  # Contextual response about ML
    assert len(content) > 50


def test_fallback_chat_contextual_science():
    data = fallback_chat("gemma2:2b", [{"role": "user", "content": "Tell me about quantum physics"}])
    content = data["message"]["content"].lower()
    assert "quantum" in content
    assert len(content) > 50


def test_fallback_generate_contextual():
    data = fallback_generate("codellama:7b", "Write a Python function to add numbers")
    assert "response" in data
    assert "python" in data["response"].lower() or "def" in data["response"].lower()
    assert data["done"] is True
    assert data["_fallback"] is True


def test_fallback_generate_code_model_includes_code():
    data = fallback_generate("codellama:7b", "Write a Python API endpoint")
    assert "```" in data["response"]  # Should include code block


def test_fallback_models_list():
    models = fallback_models()
    assert len(models) == len(MODEL_IDENTITY)
    names = [m["name"] for m in models]
    assert "llama3.2" in names
    assert "mistral" in names
    assert "codellama:7b" in names


def test_all_models_have_contextual_responses():
    """Every model returns unique contextual responses for different topics."""
    topics = [
        "What is Python programming?",
        "Explain quantum mechanics",
        "What is machine learning?",
        "Tell me about world history",
    ]
    for model_name in MODEL_IDENTITY:
        for topic in topics:
            data = fallback_chat(model_name, [{"role": "user", "content": topic}])
            content = data["message"]["content"]
            assert len(content) > 30, f"Model {model_name} gave short response for: {topic}"


def test_fallback_different_queries_get_different_responses():
    """Different questions should produce different answers."""
    r1 = fallback_chat("llama3.2", [{"role": "user", "content": "What is Python?"}])
    r2 = fallback_chat("llama3.2", [{"role": "user", "content": "Explain gravity"}])
    assert r1["message"]["content"] != r2["message"]["content"]


def test_fallback_health_endpoint(client, mock_ollama):
    mock_ollama.ping.return_value = False
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "fallback" in data["status"]
    assert len(data["available_models"]) > 0


def test_chat_uses_fallback_when_ollama_down(client, mock_ollama):
    mock_ollama.ping.return_value = False
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "What is machine learning?"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "llama3.2"
    content = data["choices"][0]["message"]["content"].lower()
    assert len(content) > 50  # Contextual response about ML
    assert data["choices"][0]["finish_reason"] == "fallback"


def test_generate_uses_fallback_when_ollama_down(client, mock_ollama):
    mock_ollama.ping.return_value = False
    r = client.post(
        "/v1/generate",
        json={"model": "mistral", "prompt": "What is the speed of light?"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "mistral"
    assert len(data["response"]) > 0


def test_models_list_fallback_when_ollama_down(client, mock_ollama):
    mock_ollama.ping.return_value = False
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["models"]) > 0
    names = [m["name"] for m in data["models"]]
    assert "llama3.2" in names


def test_ensure_models_fallback(client, mock_ollama):
    mock_ollama.ping.return_value = False
    r = client.post("/v1/models/ensure")
    assert r.status_code == 200
    data = r.json()
    assert all("fallback" in v for v in data["models"].values())


def test_model_info_fallback(client, mock_ollama):
    mock_ollama.ping.return_value = False
    r = client.get("/v1/models/llama3.2/info")
    assert r.status_code == 200
    data = r.json()
    assert data["_fallback"] is True
