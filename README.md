# Limo - FastAPI Ollama Gateway

End-to-end FastAPI backend for [Ollama](https://ollama.com), with GPU-accelerated Docker deployment and Vercel serverless support.

## Features

- **OpenAI-compatible chat completions** (`/v1/chat/completions`)
- **Text generation** (`/v1/generate`)
- **Model management** - list, pull, delete, and auto-provision models
- **GPU-accelerated Docker Compose** with NVIDIA runtime
- **All popular Ollama models** pre-configured: llama3.2, mistral, gemma2, qwen2.5, phi3, codellama, deepseek-coder, and more
- **Health monitoring** with Ollama connectivity checks
- **Vercel deployment** for the API gateway

## Quick Start

### Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Docker with GPU

```bash
cd docker
docker compose up -d
```

This starts:
1. **Ollama** with full GPU passthrough (NVIDIA)
2. **Limo API** on port 8000
3. **Model loader** that auto-pulls all configured models

### Pull All Models

```bash
# Via API
curl -X POST http://localhost:8000/v1/models/ensure

# Via script
python -m scripts.pull_models
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check with Ollama status |
| POST | `/v1/chat/completions` | Chat completions (OpenAI-compatible) |
| POST | `/v1/generate` | Simple text generation |
| GET | `/v1/models` | List available models |
| POST | `/v1/models/pull` | Pull a model |
| DELETE | `/v1/models` | Delete a model |
| POST | `/v1/models/ensure` | Auto-pull all configured models |
| GET | `/v1/models/{name}/info` | Model details |

## Configured Models

| Model | Description |
|-------|-------------|
| llama3.2 | Meta Llama 3.2 (default) |
| llama3.2:1b | Llama 3.2 1B parameter variant |
| mistral | Mistral 7B |
| gemma2:2b | Google Gemma 2 2B |
| qwen2.5:0.5b | Alibaba Qwen 2.5 0.5B |
| phi3:mini | Microsoft Phi-3 Mini |
| tinyllama | TinyLlama 1.1B |
| codellama:7b | Meta Code Llama 7B |
| deepseek-coder:1.3b | DeepSeek Coder 1.3B |
| nomic-embed-text | Nomic text embeddings |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DEFAULT_MODEL` | `llama3.2` | Default model for requests |
| `LIMO_DEBUG` | `false` | Enable debug mode |

## Testing

```bash
pytest tests/ -v
```

## Deployment

### Vercel

The project includes `vercel.json` for serverless deployment. The Vercel deployment serves as the API gateway -- you need a separate GPU server running Ollama and set `OLLAMA_BASE_URL` accordingly.

```bash
vercel --prod
```

### Docker (GPU Server)

For the Ollama backend with GPU:

```bash
cd docker
docker compose up -d
```

Requires NVIDIA GPU drivers and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## License

MIT
