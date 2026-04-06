"""Fallback response generator when Ollama is not available.

Provides meaningful responses for all configured models so the API
always works -- useful for demos and when Ollama is starting up.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

# Pre-built response templates per model family
MODEL_RESPONSES: dict[str, dict[str, str]] = {
    "llama3.2": {
        "greeting": "Hello! I'm Llama 3.2 by Meta. I'm a large language model trained to be helpful, harmless, and honest. How can I assist you today?",
        "default": "I'm Llama 3.2, a language model by Meta AI. Based on your question, I can provide the following insight: Language models like me process natural language to understand context and generate relevant responses. I can help with writing, analysis, coding, math, and general knowledge questions.",
    },
    "llama3.2:1b": {
        "greeting": "Hi there! I'm Llama 3.2 1B, a compact version of Meta's language model. What can I help you with?",
        "default": "As Llama 3.2 1B, I'm a smaller but efficient language model. I can assist with text generation, simple Q&A, and basic reasoning tasks. The answer to your question involves understanding the context and applying relevant knowledge.",
    },
    "mistral": {
        "greeting": "Bonjour! I'm Mistral 7B, an open-weight language model by Mistral AI. I'm here to help!",
        "default": "As Mistral 7B, I process your input using transformer architecture with grouped-query attention and sliding window attention. For your query, the key points are: effective communication requires clarity, context matters, and I can help you explore topics in depth.",
    },
    "gemma2:2b": {
        "greeting": "Hello! I'm Gemma 2 2B by Google DeepMind. I'm designed to be helpful and safe. What would you like to know?",
        "default": "I'm Gemma 2 2B from Google. Based on your question, I can share that modern AI systems use attention mechanisms to weigh the importance of different parts of input. I'm built with responsible AI principles and trained on diverse, high-quality data.",
    },
    "qwen2.5:0.5b": {
        "greeting": "Hello! I'm Qwen 2.5 0.5B by Alibaba Cloud. Despite my compact size, I can help with many tasks!",
        "default": "As Qwen 2.5, I'm developed by Alibaba Cloud's research team. I can assist with text understanding, generation, and basic reasoning. For your query: knowledge is built through systematic learning and applying concepts across different domains.",
    },
    "phi3:mini": {
        "greeting": "Hi! I'm Phi-3 Mini by Microsoft Research. I'm a small but capable language model. How can I help?",
        "default": "I'm Phi-3 Mini from Microsoft. Despite being compact, I was trained on high-quality textbook-level data. For your question: the answer involves understanding patterns in data and applying logical reasoning to derive conclusions.",
    },
    "tinyllama": {
        "greeting": "Hey! I'm TinyLlama 1.1B. I'm small but I try my best to help!",
        "default": "As TinyLlama, I'm a compact 1.1B parameter model trained on a large corpus. While I'm smaller than other models, I can still provide useful responses. The key insight for your question is that efficient models can still deliver valuable results.",
    },
    "codellama:7b": {
        "greeting": "Hello! I'm Code Llama 7B by Meta, specialized in code generation and understanding. What would you like to code?",
        "default": "As Code Llama 7B, I specialize in programming tasks. Here's a Python example:\n\n```python\ndef solve(input_data):\n    \"\"\"Process the input and return a result.\"\"\"\n    result = process(input_data)\n    return result\n```\n\nI can help with code generation, debugging, explanation, and refactoring across many languages.",
    },
    "deepseek-coder:1.3b": {
        "greeting": "Hi! I'm DeepSeek Coder 1.3B, focused on code intelligence. Let's write some code!",
        "default": "As DeepSeek Coder, I'm trained specifically for coding tasks. Here's an example:\n\n```python\ndef add(a: int, b: int) -> int:\n    return a + b\n\ndef main():\n    result = add(2, 3)\n    print(f'Result: {result}')\n```\n\nI can help with code completion, bug fixing, and code explanation.",
    },
    "nomic-embed-text": {
        "greeting": "Hello! I'm Nomic Embed Text, designed for creating text embeddings rather than generating text.",
        "default": "As Nomic Embed Text, I'm an embedding model that converts text into vector representations. These embeddings capture semantic meaning and can be used for similarity search, clustering, and retrieval-augmented generation (RAG) systems.",
    },
}


def _pick_response(model: str, prompt: str) -> str:
    """Select a response based on model and prompt content."""
    base = model.split(":")[0] if ":" in model else model
    # Find best matching model template
    templates = MODEL_RESPONSES.get(model) or MODEL_RESPONSES.get(base, MODEL_RESPONSES["llama3.2"])

    lower = prompt.lower().strip()
    if any(g in lower for g in ("hello", "hi", "hey", "greet", "who are you")):
        return templates["greeting"]
    return templates["default"]


def fallback_chat(
    model: str,
    messages: list[dict[str, str]],
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Generate a fallback chat response."""
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    content = _pick_response(model, last_user_msg)
    prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
    completion_tokens = len(content.split())

    return {
        "message": {"role": "assistant", "content": content},
        "model": model,
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
        "total_duration": int(time.time_ns() % 1_000_000_000),
        "_fallback": True,
    }


def fallback_generate(
    model: str,
    prompt: str,
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Generate a fallback text generation response."""
    content = _pick_response(model, prompt)
    return {
        "response": content,
        "model": model,
        "done": True,
        "total_duration": int(time.time_ns() % 1_000_000_000),
        "eval_count": len(content.split()),
        "_fallback": True,
    }


def fallback_models() -> list[dict[str, Any]]:
    """Return a list of all configured models as if they were available."""
    models = []
    for name in MODEL_RESPONSES:
        digest = hashlib.md5(name.encode()).hexdigest()
        models.append({
            "name": name,
            "size": 4_000_000_000,
            "digest": digest,
            "modified_at": "2025-01-01T00:00:00Z",
            "details": {"family": name.split(":")[0], "parameter_size": "varies"},
        })
    return models
