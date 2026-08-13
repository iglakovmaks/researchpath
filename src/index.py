"""Vercel entrypoint for the ResearchPath FastAPI application."""

from researchpath.api import app

__all__ = ["app"]
