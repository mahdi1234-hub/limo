"""Configuration for the Limo FastAPI application."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "Limo - Ollama Gateway"
    app_version: str = "0.1.0"
    debug: bool = False

    # Ollama connection
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Default model to use when none specified
    default_model: str = os.getenv("DEFAULT_MODEL", "llama3.2")

    # Request timeouts (seconds)
    request_timeout: int = 120
    stream_timeout: int = 300

    # CORS
    cors_origins: list[str] = ["*"]

    class Config:
        env_prefix = "LIMO_"


@lru_cache
def get_settings() -> Settings:
    return Settings()
