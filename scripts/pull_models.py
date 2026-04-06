"""Standalone script to pull all configured Ollama models.

Used by the model-loader Docker service and can also be run manually:
    python -m scripts.pull_models
"""

import asyncio
import sys

from app.services.ollama import OLLAMA_MODELS, OllamaClient


async def main() -> None:
    client = OllamaClient()

    print("Waiting for Ollama to be ready...")
    for _ in range(30):
        if await client.ping():
            break
        await asyncio.sleep(2)
    else:
        print("ERROR: Ollama did not become ready in time.")
        sys.exit(1)

    version = await client.version()
    print(f"Ollama version: {version}")

    print(f"Ensuring {len(OLLAMA_MODELS)} models are available...")
    results = await client.ensure_models(OLLAMA_MODELS)

    for model, status in results.items():
        print(f"  {model}: {status}")

    errors = [m for m, s in results.items() if s.startswith("error")]
    if errors:
        print(f"\nWARNING: {len(errors)} model(s) failed to pull.")
    else:
        print("\nAll models ready.")


if __name__ == "__main__":
    asyncio.run(main())
