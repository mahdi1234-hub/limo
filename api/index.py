"""Vercel serverless entry point for the Limo FastAPI app."""

from app.main import app

# Vercel expects a module-level `app` variable or a handler.
# With the Python runtime, exporting the FastAPI app directly works.
handler = app
