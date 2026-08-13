"""Durable SQLite storage for normalized ResearchPath papers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from researchpath.models import Paper


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    publication_year INTEGER NOT NULL,
    authors_json TEXT NOT NULL,
    topics_json TEXT NOT NULL,
    cited_by_count INTEGER NOT NULL DEFAULT 0,
    referenced_works_json TEXT NOT NULL,
    doi TEXT,
    open_access_url TEXT,
    source TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_publication_year
    ON papers(publication_year);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    id UNINDEXED,
    searchable_text
);
"""


def _dump_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _paper_to_row(paper: Paper) -> tuple[object, ...]:
    return (
        paper.id,
        paper.title,
        paper.abstract,
        paper.publication_year,
        _dump_list(paper.authors),
        _dump_list(paper.topics),
        paper.cited_by_count,
        _dump_list(paper.referenced_works),
        paper.doi,
        paper.open_access_url,
        paper.source,
    )


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        id=row["id"],
        title=row["title"],
        abstract=row["abstract"],
        publication_year=row["publication_year"],
        authors=json.loads(row["authors_json"]),
        topics=json.loads(row["topics_json"]),
        cited_by_count=row["cited_by_count"],
        referenced_works=json.loads(row["referenced_works_json"]),
        doi=row["doi"],
        open_access_url=row["open_access_url"],
        source=row["source"],
    )


class SQLiteCorpusStore:
    """Store normalized papers and a lightweight SQLite FTS5 index."""

    backend_name = "sqlite"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def upsert(self, paper: Paper) -> None:
        """Insert or replace one paper and its FTS document."""

        self.connection.execute(
            """
            INSERT INTO papers (
                id, title, abstract, publication_year, authors_json, topics_json,
                cited_by_count, referenced_works_json, doi, open_access_url, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                abstract = excluded.abstract,
                publication_year = excluded.publication_year,
                authors_json = excluded.authors_json,
                topics_json = excluded.topics_json,
                cited_by_count = excluded.cited_by_count,
                referenced_works_json = excluded.referenced_works_json,
                doi = excluded.doi,
                open_access_url = excluded.open_access_url,
                source = excluded.source
            """,
            _paper_to_row(paper),
        )
        self.connection.execute("DELETE FROM papers_fts WHERE id = ?", (paper.id,))
        self.connection.execute(
            "INSERT INTO papers_fts (id, searchable_text) VALUES (?, ?)",
            (paper.id, paper.searchable_text),
        )

    def upsert_many(self, papers: Iterable[Paper]) -> int:
        """Insert or replace papers in one transaction and return the count."""

        count = 0
        with self.connection:
            for paper in papers:
                self.upsert(paper)
                count += 1
        return count

    def all_papers(self, limit: int | None = None) -> list[Paper]:
        """Load papers ordered deterministically for reproducible indexes."""

        query = "SELECT * FROM papers ORDER BY publication_year, id"
        params: tuple[object, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (max(0, limit),)
        rows = self.connection.execute(query, params).fetchall()
        return [_row_to_paper(row) for row in rows]

    def get(self, paper_id: str) -> Paper | None:
        """Return one paper by normalized identifier."""

        row = self.connection.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return _row_to_paper(row) if row else None

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Run a simple FTS5 lookup against the durable corpus."""

        terms = [term for term in query.split() if term.isalnum()]
        if not terms:
            return []
        match_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        rows = self.connection.execute(
            """
            SELECT papers.*
            FROM papers_fts
            JOIN papers ON papers.id = papers_fts.id
            WHERE papers_fts MATCH ?
            ORDER BY bm25(papers_fts)
            LIMIT ?
            """,
            (match_query, max(1, limit)),
        ).fetchall()
        return [_row_to_paper(row) for row in rows]

    def stats(self) -> dict[str, int | str]:
        """Return storage statistics for diagnostics and the API."""

        count = self.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        return {"storage_backend": self.backend_name, "stored_papers": count}

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SQLiteCorpusStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def migrate_json_to_sqlite(json_path: str | Path, database_path: str | Path) -> int:
    """Migrate a normalized JSON corpus into SQLite."""

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON list in {json_path}")
    validator = getattr(Paper, "model_validate", Paper.parse_obj)
    papers = [validator(item) for item in payload]
    with SQLiteCorpusStore(database_path) as store:
        return store.upsert_many(papers)
