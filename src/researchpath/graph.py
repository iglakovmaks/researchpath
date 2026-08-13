from __future__ import annotations

from collections import defaultdict

from researchpath.corpus import SQLiteCorpusStore
from researchpath.models import Paper


class CitationGraph:
    """Directed citation graph over the currently indexed paper collection."""

    def __init__(self, papers: list[Paper]):
        self.papers = {paper.id: paper for paper in papers}
        self.outgoing: dict[str, set[str]] = defaultdict(set)
        self.incoming: dict[str, set[str]] = defaultdict(set)
        for paper in papers:
            for referenced_id in paper.referenced_works:
                if referenced_id not in self.papers:
                    continue
                self.outgoing[paper.id].add(referenced_id)
                self.incoming[referenced_id].add(paper.id)

    def is_connected(self, first_id: str, second_id: str) -> bool:
        """Return whether two papers have a direct citation relationship."""

        return second_id in self.outgoing.get(first_id, set()) or first_id in self.outgoing.get(
            second_id, set()
        )

    def neighbors(self, paper_id: str) -> set[str]:
        """Return papers directly citing or cited by a paper."""

        return self.outgoing.get(paper_id, set()) | self.incoming.get(paper_id, set())

    def stats(self) -> dict[str, int]:
        """Return compact graph statistics for the API and CLI."""

        edge_count = sum(len(edges) for edges in self.outgoing.values())
        return {
            "nodes": len(self.papers),
            "edges": edge_count,
            "connected_nodes": len(
                {node for node in self.outgoing if self.outgoing[node]}
                | {node for node in self.incoming if self.incoming[node]}
            ),
        }


class SQLiteCitationGraph:
    """Citation graph facade backed by normalized SQLite edges."""

    def __init__(self, store: SQLiteCorpusStore):
        self.store = store

    def is_connected(self, first_id: str, second_id: str) -> bool:
        return self.store.citation_is_connected(first_id, second_id)

    def neighbors(self, paper_id: str) -> set[str]:
        return self.store.citation_neighbors(paper_id)

    def stats(self) -> dict[str, int]:
        return self.store.citation_stats()
