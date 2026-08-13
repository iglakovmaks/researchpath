from __future__ import annotations

from pathlib import Path

from researchpath.data import load_papers, load_papers_from_path
from researchpath.graph import CitationGraph
from researchpath.models import Paper, ReadingPathStep, SearchResult
from researchpath.path import build_reading_path
from researchpath.search import HybridSearchEngine


class ResearchPathService:
    """Application service that keeps search and graph retrieval consistent."""

    def __init__(
        self,
        papers: list[Paper],
        embedding_model: str | None = None,
        storage_backend: str = "memory",
    ):
        self.papers = papers
        self.search_engine = HybridSearchEngine(papers, embedding_model=embedding_model)
        self.graph = CitationGraph(papers)
        self.storage_backend = storage_backend

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        embedding_model: str | None = None,
    ) -> ResearchPathService:
        return cls(load_papers(path), embedding_model=embedding_model, storage_backend="json")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        embedding_model: str | None = None,
    ) -> ResearchPathService:
        """Create a service from either a JSON or SQLite corpus."""

        papers, storage_backend = load_papers_from_path(path)
        return cls(papers, embedding_model=embedding_model, storage_backend=storage_backend)

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        return self.search_engine.search(query=query, limit=limit, mode=mode)

    def reading_path(
        self,
        query: str,
        limit: int = 6,
        mode: str = "hybrid",
    ) -> list[ReadingPathStep]:
        return build_reading_path(
            self.search(query, limit=max(limit * 4, 10), mode=mode), self.graph, limit
        )

    def get_paper(self, paper_id: str) -> Paper | None:
        return self.search_engine.get_paper(paper_id)

    def stats(self) -> dict[str, int | str]:
        return {
            **self.graph.stats(),
            "papers": len(self.papers),
            "retrieval_backend": self.search_engine.vector_backend,
            "storage_backend": self.storage_backend,
        }
