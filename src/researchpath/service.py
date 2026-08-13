from __future__ import annotations

from pathlib import Path

from researchpath.data import load_papers
from researchpath.graph import CitationGraph
from researchpath.models import Paper, ReadingPathStep, SearchResult
from researchpath.path import build_reading_path
from researchpath.search import HybridSearchEngine


class ResearchPathService:
    """Application service that keeps search and graph retrieval consistent."""

    def __init__(self, papers: list[Paper]):
        self.papers = papers
        self.search_engine = HybridSearchEngine(papers)
        self.graph = CitationGraph(papers)

    @classmethod
    def from_json(cls, path: str | Path) -> ResearchPathService:
        return cls(load_papers(path))

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        return self.search_engine.search(query=query, limit=limit)

    def reading_path(self, query: str, limit: int = 6) -> list[ReadingPathStep]:
        return build_reading_path(self.search(query, limit=max(limit * 4, 10)), self.graph, limit)

    def get_paper(self, paper_id: str) -> Paper | None:
        return self.search_engine.get_paper(paper_id)

    def stats(self) -> dict[str, int]:
        return {**self.graph.stats(), "papers": len(self.papers)}
