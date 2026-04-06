"""Tests for chat and generation endpoints."""


def test_chat_completions(client, mock_ollama):
    mock_ollama.ping.return_value = True
    mock_ollama.chat.return_value = {
        "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
        "prompt_eval_count": 10,
        "eval_count": 8,
    }

    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Hello" in data["choices"][0]["message"]["content"]
    assert data["usage"]["total_tokens"] == 18


def test_chat_completions_with_model(client, mock_ollama):
    mock_ollama.ping.return_value = True
    mock_ollama.chat.return_value = {
        "message": {"role": "assistant", "content": "I am Mistral."},
        "prompt_eval_count": 5,
        "eval_count": 4,
    }

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "mistral",
            "messages": [{"role": "user", "content": "Who are you?"}],
            "temperature": 0.7,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "mistral"


def test_chat_completions_fallback_on_error(client, mock_ollama):
    """When Ollama errors, fallback kicks in and returns 200."""
    mock_ollama.ping.return_value = True
    mock_ollama.chat.side_effect = Exception("Connection refused")

    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["finish_reason"] == "fallback"
    assert len(data["choices"][0]["message"]["content"]) > 0


def test_chat_completions_fallback_when_disconnected(client, mock_ollama):
    """When Ollama is down, fallback returns a response."""
    mock_ollama.ping.return_value = False

    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello!"}]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["finish_reason"] == "fallback"


def test_generate(client, mock_ollama):
    mock_ollama.ping.return_value = True
    mock_ollama.generate.return_value = {
        "response": "The capital of France is Paris.",
        "done": True,
        "total_duration": 1234567,
        "eval_count": 7,
    }

    r = client.post(
        "/v1/generate",
        json={"prompt": "What is the capital of France?"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "Paris" in data["response"]
    assert data["done"] is True


def test_generate_with_options(client, mock_ollama):
    mock_ollama.ping.return_value = True
    mock_ollama.generate.return_value = {
        "response": "42",
        "done": True,
    }

    r = client.post(
        "/v1/generate",
        json={
            "model": "phi3:mini",
            "prompt": "Answer of life",
            "temperature": 0.0,
            "max_tokens": 10,
        },
    )
    assert r.status_code == 200
    mock_ollama.generate.assert_called_once()


def test_generate_fallback_on_error(client, mock_ollama):
    """When Ollama errors, fallback returns a response."""
    mock_ollama.ping.return_value = True
    mock_ollama.generate.side_effect = Exception("timeout")

    r = client.post("/v1/generate", json={"prompt": "test"})
    assert r.status_code == 200
    assert len(r.json()["response"]) > 0
