"""Vercel entrypoint for the ResearchPath FastAPI application."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from researchpath.api import app

__all__ = ["app"]
