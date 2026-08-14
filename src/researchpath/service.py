from __future__ import annotations

from pathlib import Path

from researchpath.corpus import SQLiteCorpusStore
from researchpath.data import load_papers
from researchpath.graph import CitationGraph, SQLiteCitationGraph
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
        corpus_store: SQLiteCorpusStore | None = None,
    ):
        self.papers = papers
        self.corpus_store = corpus_store
        self.search_engine = HybridSearchEngine(
            papers,
            embedding_model=embedding_model,
            corpus_store=corpus_store,
        )
        self.graph = SQLiteCitationGraph(corpus_store) if corpus_store else CitationGraph(papers)
        self.storage_backend = storage_backend
        self.default_mode = "bm25" if corpus_store else "hybrid"

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
        read_only: bool = False,
    ) -> ResearchPathService:
        """Create a service from either a JSON or SQLite corpus."""

        corpus_path = Path(path)
        if corpus_path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            return cls(
                [],
                embedding_model=embedding_model,
                storage_backend="sqlite",
                corpus_store=SQLiteCorpusStore(corpus_path, read_only=read_only),
            )
        return cls(
            load_papers(corpus_path), embedding_model=embedding_model, storage_backend="json"
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str | None = None,
    ) -> list[SearchResult]:
        return self.search_engine.search(
            query=query,
            limit=limit,
            mode=mode or self.default_mode,
        )

    def reading_path(
        self,
        query: str,
        limit: int = 6,
        mode: str | None = None,
    ) -> list[ReadingPathStep]:
        return build_reading_path(
            self.search(query, limit=max(limit * 4, 10), mode=mode), self.graph, limit
        )

    def get_paper(self, paper_id: str) -> Paper | None:
        return self.search_engine.get_paper(paper_id)

    def list_papers(self, limit: int = 100) -> list[Paper]:
        """Return a bounded metadata page for API clients."""

        if self.corpus_store:
            return self.corpus_store.all_papers(limit=max(1, limit))
        return self.papers[: max(1, limit)]

    def stats(self) -> dict[str, int | str]:
        return {
            **self.graph.stats(),
            "papers": self.corpus_store.count() if self.corpus_store else len(self.papers),
            "retrieval_backend": (
                self.search_engine.bm25_backend
                if self.corpus_store
                else self.search_engine.vector_backend
            ),
            "storage_backend": self.storage_backend,
        }
