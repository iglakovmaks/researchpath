from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from researchpath.models import Paper, ReadingPathStep, SearchResult
from researchpath.service import ResearchPathService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "demo.db"
EMBEDDING_INDEX_PATH = PROJECT_ROOT / "data" / "demo_embeddings.json"
WEB_PATH = PROJECT_ROOT / "web" / "index.html"


def create_app(
    data_path: str | Path = DEFAULT_DATA_PATH,
    read_only: bool | None = None,
) -> FastAPI:
    """Create the ResearchPath HTTP API."""

    if read_only is None:
        read_only = os.getenv("VERCEL") == "1"
    service = ResearchPathService.from_path(
        data_path,
        embedding_model=os.getenv("RESEARCHPATH_EMBEDDING_MODEL"),
        read_only=read_only,
    )
    app = FastAPI(
        title="ResearchPath",
        description="An explainable navigator for computer science literature.",
        version="0.5.1",
    )
    app.state.service = service

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_PATH)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/stats")
    def stats() -> dict[str, int | str]:
        return service.stats()

    @app.get("/api/search", response_model=list[SearchResult])
    def search(
        q: str = Query(min_length=2, description="Topic or research question."),
        limit: int = Query(default=10, ge=1, le=50),
        mode: Literal["hybrid", "bm25", "vector"] | None = Query(
            default=None,
            description="Retrieval mode; defaults to hybrid for JSON and bm25 for SQLite.",
        ),
    ) -> list[SearchResult]:
        try:
            return service.search(q, limit, mode=mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/reading-path", response_model=list[ReadingPathStep])
    def reading_path(
        q: str = Query(min_length=2, description="Topic or research question."),
        limit: int = Query(default=6, ge=2, le=12),
    ) -> list[ReadingPathStep]:
        return service.reading_path(q, limit)

    @app.get("/api/papers/{paper_id}", response_model=Paper)
    def paper(paper_id: str) -> Paper:
        found = service.get_paper(paper_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return found

    @app.get("/api/papers", response_model=list[Paper])
    def papers(limit: int = Query(default=100, ge=1, le=1000)) -> list[Paper]:
        return service.list_papers(limit)

    @app.get("/api/embedding-index", include_in_schema=False)
    def embedding_index() -> FileResponse:
        return FileResponse(EMBEDDING_INDEX_PATH, media_type="application/json")

    return app


app = create_app()
