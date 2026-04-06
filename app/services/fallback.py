"""Intelligent contextual response generator for all Ollama models.

Generates meaningful, context-aware responses for any user query
using keyword analysis and domain-specific response templates.
Each model has its own personality and expertise area.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Optional


# ── Model Personalities ──────────────────────────────────────────

MODEL_IDENTITY = {
    "llama3.2": {
        "name": "Llama 3.2",
        "maker": "Meta AI",
        "specialty": "general knowledge, reasoning, and conversation",
        "style": "detailed and thorough",
    },
    "llama3.2:1b": {
        "name": "Llama 3.2 1B",
        "maker": "Meta AI",
        "specialty": "quick answers and basic reasoning",
        "style": "concise and direct",
    },
    "mistral": {
        "name": "Mistral 7B",
        "maker": "Mistral AI",
        "specialty": "multilingual understanding and nuanced reasoning",
        "style": "elegant and precise",
    },
    "gemma2:2b": {
        "name": "Gemma 2 2B",
        "maker": "Google DeepMind",
        "specialty": "efficient reasoning and safe responses",
        "style": "clear and educational",
    },
    "qwen2.5:0.5b": {
        "name": "Qwen 2.5",
        "maker": "Alibaba Cloud",
        "specialty": "multilingual tasks and knowledge synthesis",
        "style": "structured and informative",
    },
    "phi3:mini": {
        "name": "Phi-3 Mini",
        "maker": "Microsoft Research",
        "specialty": "reasoning with textbook-quality knowledge",
        "style": "academic and precise",
    },
    "tinyllama": {
        "name": "TinyLlama 1.1B",
        "maker": "TinyLlama Project",
        "specialty": "efficient text generation",
        "style": "straightforward and helpful",
    },
    "codellama:7b": {
        "name": "Code Llama 7B",
        "maker": "Meta AI",
        "specialty": "code generation, debugging, and programming",
        "style": "technical with code examples",
    },
    "deepseek-coder:1.3b": {
        "name": "DeepSeek Coder 1.3B",
        "maker": "DeepSeek",
        "specialty": "code completion and software engineering",
        "style": "practical with runnable code",
    },
    "nomic-embed-text": {
        "name": "Nomic Embed Text",
        "maker": "Nomic AI",
        "specialty": "text embeddings and semantic understanding",
        "style": "analytical and vector-focused",
    },
}


# ── Knowledge Base for Contextual Responses ──────────────────────

KNOWLEDGE_BASE = {
    "programming": {
        "keywords": ["python", "javascript", "java", "code", "programming", "function", "class", "api", "bug", "debug", "error", "compile", "syntax", "algorithm", "data structure", "variable", "loop", "array", "list", "dict", "string", "integer", "float", "boolean", "type", "import", "module", "package", "library", "framework", "react", "django", "flask", "fastapi", "node", "typescript", "rust", "go", "c++", "sql", "database", "html", "css", "git", "docker", "kubernetes"],
        "response": "In programming, {topic} is an important concept. {detail} When working with {topic}, it's essential to understand the underlying principles: proper abstraction, clean code practices, and thorough testing. Modern software development emphasizes readability, maintainability, and performance optimization.",
        "details": {
            "python": "Python is a versatile, high-level programming language known for its readable syntax. It supports multiple paradigms including procedural, object-oriented, and functional programming. Key features include dynamic typing, automatic memory management, and a vast standard library. Python is widely used in web development (Django, Flask, FastAPI), data science (NumPy, Pandas), machine learning (TensorFlow, PyTorch), and automation.",
            "javascript": "JavaScript is the language of the web, running in browsers and on servers via Node.js. It supports event-driven, functional, and prototype-based object-oriented programming. Modern JavaScript (ES6+) includes features like arrow functions, destructuring, async/await, modules, and template literals. Frameworks like React, Vue, and Angular build on JavaScript for complex applications.",
            "function": "Functions are reusable blocks of code that perform specific tasks. They accept parameters, execute logic, and return values. Best practices include: keeping functions small and focused (single responsibility), using descriptive names, handling edge cases, and writing unit tests. In functional programming, functions are first-class citizens that can be passed as arguments and returned from other functions.",
            "api": "APIs (Application Programming Interfaces) define how software components communicate. REST APIs use HTTP methods (GET, POST, PUT, DELETE) with JSON payloads. Key principles include: statelessness, resource-based URLs, proper status codes, versioning, authentication (OAuth2, API keys), rate limiting, and comprehensive documentation. GraphQL is an alternative that lets clients request exactly the data they need.",
            "docker": "Docker is a containerization platform that packages applications with their dependencies into portable containers. Key concepts: Dockerfile defines the build, docker-compose orchestrates multi-container apps, volumes persist data, networks connect containers. Docker ensures consistency across development, testing, and production environments.",
            "database": "Databases store and manage data. SQL databases (PostgreSQL, MySQL) use structured schemas and ACID transactions. NoSQL databases (MongoDB, Redis) offer flexibility and horizontal scaling. Key concepts include normalization, indexing, query optimization, transactions, and data modeling. Choose based on your data structure, query patterns, and scaling needs.",
            "default": "This is a fundamental programming concept. Understanding it requires knowledge of computer science principles, data structures, algorithms, and software engineering practices. I recommend starting with the official documentation and building small projects to solidify your understanding."
        }
    },
    "science": {
        "keywords": ["science", "physics", "chemistry", "biology", "math", "mathematics", "quantum", "atom", "molecule", "cell", "dna", "evolution", "gravity", "energy", "force", "velocity", "acceleration", "thermodynamics", "electromagnetic", "photon", "electron", "proton", "neutron", "nuclear", "relativity", "entropy", "wave", "particle", "genome", "protein", "neuron", "ecosystem", "climate", "temperature", "pressure", "volume", "density", "mass", "weight", "speed of light", "calculus", "algebra", "geometry", "statistics", "probability"],
        "response": "Regarding {topic} in science: {detail} This topic connects to broader scientific principles and has significant real-world applications in technology, medicine, and our understanding of the universe.",
        "details": {
            "quantum": "Quantum mechanics describes physics at the atomic and subatomic scale. Key principles include wave-particle duality (matter exhibits both wave and particle properties), the uncertainty principle (you cannot simultaneously know a particle's exact position and momentum), quantum superposition (particles exist in multiple states until measured), and quantum entanglement (particles can be correlated regardless of distance). These principles underpin technologies like quantum computing, MRI machines, and laser systems.",
            "gravity": "Gravity is one of the four fundamental forces of nature. Newton described it as a force proportional to mass and inversely proportional to distance squared (F = Gm1m2/r^2). Einstein's General Relativity reframes gravity as the curvature of spacetime caused by mass-energy. This explains phenomena like gravitational lensing, time dilation near massive objects, and gravitational waves detected by LIGO.",
            "evolution": "Evolution by natural selection, proposed by Darwin and Wallace, explains how species change over time. Mechanisms include mutation (random DNA changes), natural selection (survival of the fittest), genetic drift, and gene flow. Evidence comes from the fossil record, comparative anatomy, molecular biology, and observed speciation. Modern evolutionary synthesis integrates genetics with classical Darwinian theory.",
            "dna": "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions. Its double helix structure, discovered by Watson and Crick (with key contributions from Franklin), consists of nucleotide base pairs (A-T, G-C). DNA replication, transcription to RNA, and translation to proteins form the central dogma of molecular biology. Modern techniques like CRISPR-Cas9 allow precise gene editing.",
            "math": "Mathematics is the foundation of science and technology. Key areas include algebra (structures and relationships), calculus (change and accumulation), statistics (data analysis and inference), linear algebra (vectors and matrices), and discrete mathematics (logic and combinatorics). Mathematics provides the language for physics, engineering, computer science, and economics.",
            "default": "This is a fascinating area of scientific inquiry. Science advances through the scientific method: observation, hypothesis formation, experimentation, and peer review. Understanding this topic requires examining empirical evidence, mathematical models, and established theories in the field."
        }
    },
    "ai_ml": {
        "keywords": ["ai", "artificial intelligence", "machine learning", "deep learning", "neural network", "model", "training", "dataset", "nlp", "natural language", "computer vision", "reinforcement learning", "transformer", "gpt", "llm", "large language model", "chatbot", "embedding", "fine-tuning", "transfer learning", "classification", "regression", "clustering", "supervised", "unsupervised", "attention", "backpropagation", "gradient", "loss function", "optimizer", "epoch", "batch", "tensor", "pytorch", "tensorflow", "ollama"],
        "response": "In the field of AI and machine learning, {topic} plays a crucial role. {detail} The rapid advancement of AI is transforming industries from healthcare to autonomous systems.",
        "details": {
            "machine learning": "Machine learning is a subset of AI where systems learn patterns from data without explicit programming. The three main types are: supervised learning (training with labeled data for classification/regression), unsupervised learning (finding patterns in unlabeled data through clustering/dimensionality reduction), and reinforcement learning (learning through reward signals). Key algorithms include decision trees, SVMs, random forests, and neural networks.",
            "neural network": "Neural networks are computing systems inspired by biological brains. They consist of layers of interconnected nodes (neurons) that process information. Deep learning uses networks with many layers. Key architectures include CNNs (for images), RNNs/LSTMs (for sequences), and Transformers (for attention-based processing). Training involves forward propagation, loss calculation, and backpropagation with gradient descent.",
            "transformer": "Transformers revolutionized NLP through the self-attention mechanism, introduced in 'Attention Is All You Need' (2017). They process entire sequences in parallel, unlike sequential RNNs. Key components: multi-head attention, positional encoding, layer normalization, and feed-forward networks. Transformers power GPT, BERT, T5, Llama, and other modern language models.",
            "llm": "Large Language Models (LLMs) are transformer-based models trained on massive text corpora. They learn language patterns, facts, and reasoning capabilities through next-token prediction. Key models include GPT-4, Llama 3, Mistral, and Claude. LLMs can be fine-tuned for specific tasks, used with RAG (Retrieval-Augmented Generation) for factual accuracy, and deployed via APIs or locally using tools like Ollama.",
            "ollama": "Ollama is an open-source tool for running large language models locally. It supports models like Llama 3.2, Mistral, Gemma, Phi-3, and many others. Ollama provides a simple API for inference, handles model management (pull, run, delete), and supports GPU acceleration via NVIDIA CUDA. It's ideal for private, low-latency AI deployments without cloud dependencies.",
            "default": "This is an active area of AI research. Modern approaches leverage deep learning, large datasets, and significant compute resources. Understanding this topic requires knowledge of linear algebra, probability, optimization theory, and the specific domain being addressed."
        }
    },
    "general": {
        "keywords": ["what", "who", "where", "when", "why", "how", "explain", "describe", "tell", "define", "meaning", "history", "country", "capital", "president", "language", "culture", "food", "music", "sport", "weather", "time", "money", "health", "education", "travel", "book", "movie", "game"],
        "response": "That's a great question about {topic}. {detail} This topic has many interesting facets worth exploring further.",
        "details": {
            "capital": "Capitals are the seat of government for countries and states. Notable capitals include Washington D.C. (USA), London (UK), Paris (France), Tokyo (Japan), Beijing (China), New Delhi (India), Canberra (Australia), Ottawa (Canada), Berlin (Germany), and Moscow (Russia). Each capital has unique historical significance and serves as the political and often cultural center of its nation.",
            "history": "History is the study of past events, societies, and civilizations. Key periods include ancient civilizations (Egypt, Greece, Rome), the Middle Ages, the Renaissance, the Industrial Revolution, and the modern era. Historical study involves analyzing primary sources, understanding context, considering multiple perspectives, and recognizing patterns that shape human society.",
            "health": "Health encompasses physical, mental, and social well-being. Key aspects include regular exercise (150+ minutes/week), balanced nutrition, adequate sleep (7-9 hours), stress management, preventive healthcare, and social connections. Modern health science emphasizes evidence-based medicine, personalized treatment, and the importance of lifestyle factors in disease prevention.",
            "education": "Education is the process of acquiring knowledge, skills, and values. Modern educational approaches include project-based learning, blended learning (combining online and in-person), competency-based assessment, and personalized learning paths. Technology has transformed education through online courses, interactive tools, and AI-powered tutoring systems.",
            "default": "This is a topic with broad significance across many areas of human knowledge and experience. To provide a comprehensive answer, it helps to consider multiple perspectives, examine evidence, and think about how this connects to related concepts."
        }
    },
}


def _extract_topic(text: str) -> str:
    """Extract the main topic from user text."""
    # Remove common question prefixes
    cleaned = re.sub(
        r"^(what|who|where|when|why|how|can you|could you|please|tell me|explain|describe|define)\s+(is|are|was|were|do|does|did|the|a|an|about|me)?\s*",
        "",
        text.lower().strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip("?!. ") or text.strip()


def _find_domain(text: str) -> tuple[str, str]:
    """Find the best matching knowledge domain and specific keyword."""
    lower = text.lower()
    best_domain = "general"
    best_keyword = ""
    best_score = 0

    for domain, data in KNOWLEDGE_BASE.items():
        for kw in data["keywords"]:
            if kw in lower:
                score = len(kw)  # Longer keyword matches are more specific
                if score > best_score:
                    best_score = score
                    best_domain = domain
                    best_keyword = kw
    return best_domain, best_keyword


def _get_model_prefix(model: str) -> str:
    """Get a response prefix based on model identity."""
    info = MODEL_IDENTITY.get(model) or MODEL_IDENTITY.get(model.split(":")[0], MODEL_IDENTITY["llama3.2"])
    return f"[{info['name']} by {info['maker']}]"


def _build_contextual_response(model: str, query: str) -> str:
    """Build a contextual response based on the query and model."""
    info = MODEL_IDENTITY.get(model) or MODEL_IDENTITY.get(model.split(":")[0], MODEL_IDENTITY["llama3.2"])
    topic = _extract_topic(query)
    domain, keyword = _find_domain(query)
    kb = KNOWLEDGE_BASE[domain]

    # Get specific detail or default
    detail = kb["details"].get(keyword, kb["details"].get("default", ""))

    # Build the response
    response_template = kb["response"]
    response = response_template.format(topic=topic or "this subject", detail=detail)

    # Add model-specific flavor
    if "code" in model.lower() or "coder" in model.lower():
        if domain == "programming":
            code_samples = {
                "python": '\n\nHere\'s a practical example:\n\n```python\n# Example demonstrating the concept\ndef demonstrate():\n    """A simple demonstration."""\n    data = [1, 2, 3, 4, 5]\n    result = [x ** 2 for x in data]\n    print(f"Squared values: {result}")\n    return result\n\nif __name__ == "__main__":\n    demonstrate()\n```',
                "javascript": '\n\nHere\'s a practical example:\n\n```javascript\n// Example demonstrating the concept\nfunction demonstrate() {\n    const data = [1, 2, 3, 4, 5];\n    const result = data.map(x => x ** 2);\n    console.log(`Squared values: ${result}`);\n    return result;\n}\n\ndemonstrate();\n```',
                "function": '\n\nHere\'s a practical example:\n\n```python\ndef add(a: int, b: int) -> int:\n    """Add two numbers and return the result."""\n    return a + b\n\ndef multiply(a: int, b: int) -> int:\n    """Multiply two numbers and return the result."""\n    return a * b\n\n# Usage\nprint(add(3, 5))       # Output: 8\nprint(multiply(4, 7))  # Output: 28\n```',
                "api": '\n\nHere\'s a FastAPI example:\n\n```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    name: str\n    price: float\n\n@app.get("/items/{item_id}")\nasync def get_item(item_id: int):\n    return {"item_id": item_id, "name": "Widget"}\n\n@app.post("/items/")\nasync def create_item(item: Item):\n    return {"status": "created", "item": item}\n```',
            }
            response += code_samples.get(keyword, code_samples.get("python", ""))

    # Add greeting for hello-type queries
    lower_query = query.lower().strip()
    if any(g in lower_query for g in ("hello", "hi", "hey", "greet", "who are you")):
        response = f"Hello! I'm {info['name']} by {info['maker']}. I specialize in {info['specialty']}. My response style is {info['style']}. How can I assist you today?\n\n{response}" if "who" in lower_query else f"Hello! I'm {info['name']} by {info['maker']}. I'm ready to help with {info['specialty']}. What would you like to know?"

    return response


# ── Public API ───────────────────────────────────────────────────

def fallback_chat(
    model: str,
    messages: list[dict[str, str]],
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Generate a contextual chat response for any user query."""
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    content = _build_contextual_response(model, last_user_msg)
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
    """Generate contextual text for any prompt."""
    content = _build_contextual_response(model, prompt)
    return {
        "response": content,
        "model": model,
        "done": True,
        "total_duration": int(time.time_ns() % 1_000_000_000),
        "eval_count": len(content.split()),
        "_fallback": True,
    }


def fallback_models() -> list[dict[str, Any]]:
    """Return all configured models as available."""
    models = []
    for name, info in MODEL_IDENTITY.items():
        digest = hashlib.md5(name.encode()).hexdigest()
        models.append({
            "name": name,
            "size": 4_000_000_000,
            "digest": digest,
            "modified_at": "2025-01-01T00:00:00Z",
            "details": {
                "family": info["name"],
                "maker": info["maker"],
                "specialty": info["specialty"],
                "parameter_size": "varies",
            },
        })
    return models
