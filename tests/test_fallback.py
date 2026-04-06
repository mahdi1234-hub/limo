"""Tests for fallback mode -- API always responds even without Ollama."""

from app.services.fallback import (
    MODEL_RESPONSES,
    fallback_chat,
    fallback_generate,
    fallback_models,
)


def test_fallback_chat_greeting():
    data = fallback_chat("llama3.2", [{"role": "user", "content": "Hello!"}])
    assert data["message"]["role"] == "assistant"
    assert "Llama 3.2" in data["message"]["content"]
    assert data["_fallback"] is True


def test_fallback_chat_default():
    data = fallback_chat("mistral", [{"role": "user", "content": "Tell me about AI"}])
    assert data["message"]["role"] == "assistant"
    assert len(data["message"]["content"]) > 10
    assert data["_fallback"] is True


def test_fallback_generate():
    data = fallback_generate("codellama:7b", "Write a function")
    assert "response" in data
    assert len(data["response"]) > 10
    assert data["done"] is True
    assert data["_fallback"] is True


def test_fallback_models_list():
    models = fallback_models()
    assert len(models) == len(MODEL_RESPONSES)
    names = [m["name"] for m in models]
    assert "llama3.2" in names
    assert "mistral" in names
    assert "codellama:7b" in names


def test_all_models_have_fallback_responses():
    """Every model in MODEL_RESPONSES returns a non-empty response."""
    for model_name in MODEL_RESPONSES:
        chat_data = fallback_chat(model_name, [{"role": "user", "content": "test query"}])
        assert len(chat_data["message"]["content"]) > 0, f"Empty chat response for {model_name}"

        gen_data = fallback_generate(model_name, "test prompt")
        assert len(gen_data["response"]) > 0, f"Empty generate response for {model_name}"


def test_fallback_chat_all_models_greeting():
    """Every model returns a greeting response."""
    for model_name in MODEL_RESPONSES:
        data = fallback_chat(model_name, [{"role": "user", "content": "Hello"}])
        content = data["message"]["content"]
        assert len(content) > 10, f"Model {model_name} greeting too short: {content}"


def test_fallback_health_endpoint(client, mock_ollama):
    """Health endpoint shows fallback models when Ollama is down."""
    mock_ollama.ping.return_value = False

    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "fallback" in data["status"]
    assert len(data["available_models"]) > 0


def test_chat_uses_fallback_when_ollama_down(client, mock_ollama):
    """Chat endpoint returns fallback response when Ollama is unavailable."""
    mock_ollama.ping.return_value = False

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello!"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "llama3.2"
    assert len(data["choices"]) == 1
    assert len(data["choices"][0]["message"]["content"]) > 0
    assert data["choices"][0]["finish_reason"] == "fallback"


def test_generate_uses_fallback_when_ollama_down(client, mock_ollama):
    """Generate endpoint returns fallback response when Ollama is unavailable."""
    mock_ollama.ping.return_value = False

    r = client.post(
        "/v1/generate",
        json={"model": "mistral", "prompt": "What is AI?"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "mistral"
    assert len(data["response"]) > 0


def test_models_list_fallback_when_ollama_down(client, mock_ollama):
    """Models endpoint returns fallback list when Ollama is unavailable."""
    mock_ollama.ping.return_value = False

    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["models"]) > 0
    names = [m["name"] for m in data["models"]]
    assert "llama3.2" in names


def test_ensure_models_fallback(client, mock_ollama):
    """Ensure endpoint reports all models ready in fallback mode."""
    mock_ollama.ping.return_value = False

    r = client.post("/v1/models/ensure")
    assert r.status_code == 200
    data = r.json()
    assert all("fallback" in v for v in data["models"].values())


def test_model_info_fallback(client, mock_ollama):
    """Model info endpoint returns fallback info when Ollama is down."""
    mock_ollama.ping.return_value = False

    r = client.get("/v1/models/llama3.2/info")
    assert r.status_code == 200
    data = r.json()
    assert data["_fallback"] is True
    assert "llama3.2" in data["modelfile"]
