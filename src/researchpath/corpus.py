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

CREATE TABLE IF NOT EXISTS citations (
    citing_id TEXT NOT NULL,
    cited_id TEXT NOT NULL,
    PRIMARY KEY (citing_id, cited_id)
);
CREATE INDEX IF NOT EXISTS idx_citations_cited_id ON citations(cited_id);

CREATE TABLE IF NOT EXISTS corpus_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

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

    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI executes synchronous handlers in a worker thread. The store
        # is read-mostly after startup, so allow that shared connection to be
        # used by the request workers.
        if read_only:
            database_uri = f"file:{self.path.resolve()}?mode=ro"
            self.connection = sqlite3.connect(
                database_uri,
                uri=True,
                check_same_thread=False,
            )
        else:
            self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        if not read_only:
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.executescript(SCHEMA)
            self._ensure_citations_ready()
            self.connection.commit()

    def _ensure_citations_ready(self) -> None:
        row = self.connection.execute(
            "SELECT value FROM corpus_meta WHERE key = 'citations_ready'"
        ).fetchone()
        if row and row[0] == "1":
            return
        self._rebuild_citations()

    def _rebuild_citations(self) -> None:
        self.connection.execute("DELETE FROM citations")
        rows = self.connection.execute("SELECT id, referenced_works_json FROM papers").fetchall()
        self.connection.executemany(
            "INSERT OR IGNORE INTO citations (citing_id, cited_id) VALUES (?, ?)",
            [
                (row["id"], cited_id)
                for row in rows
                for cited_id in json.loads(row["referenced_works_json"])
            ],
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO corpus_meta (key, value) VALUES ('citations_ready', '1')"
        )

    def upsert(self, paper: Paper) -> None:
        """Insert or replace one paper and its FTS document."""

        if self.read_only:
            raise RuntimeError("Cannot upsert into a read-only SQLite corpus")

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
        self.connection.execute("DELETE FROM citations WHERE citing_id = ?", (paper.id,))
        self.connection.executemany(
            "INSERT OR IGNORE INTO citations (citing_id, cited_id) VALUES (?, ?)",
            [(paper.id, cited_id) for cited_id in paper.referenced_works],
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO corpus_meta (key, value) VALUES ('citations_ready', '1')"
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

        return [paper for paper, _ in self.search_scored(query, limit)]

    def search_scored(self, query: str, limit: int = 10) -> list[tuple[Paper, float]]:
        """Return top papers with positive FTS5 BM25 relevance scores."""

        terms = [term for term in query.split() if term.isalnum()]
        if not terms:
            return []
        match_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        rows = self.connection.execute(
            """
            SELECT papers.*, -bm25(papers_fts) AS relevance
            FROM papers_fts
            JOIN papers ON papers.id = papers_fts.id
            WHERE papers_fts MATCH ?
            ORDER BY bm25(papers_fts)
            LIMIT ?
            """,
            (match_query, max(1, limit)),
        ).fetchall()
        return [(_row_to_paper(row), float(row["relevance"])) for row in rows]

    def citation_is_connected(self, first_id: str, second_id: str) -> bool:
        """Check a direct citation edge without loading the corpus."""

        row = self.connection.execute(
            """
            SELECT 1 FROM citations
            WHERE (citing_id = ? AND cited_id = ?)
               OR (citing_id = ? AND cited_id = ?)
            LIMIT 1
            """,
            (first_id, second_id, second_id, first_id),
        ).fetchone()
        return row is not None

    def citation_neighbors(self, paper_id: str) -> set[str]:
        """Return direct citation neighbors for one paper."""

        rows = self.connection.execute(
            """
            SELECT cited_id AS neighbor FROM citations WHERE citing_id = ?
            UNION
            SELECT citing_id AS neighbor FROM citations WHERE cited_id = ?
            """,
            (paper_id, paper_id),
        ).fetchall()
        return {row["neighbor"] for row in rows}

    def citation_stats(self) -> dict[str, int]:
        """Return graph statistics using SQL joins over the durable corpus."""

        edge_count = self.connection.execute(
            """
            SELECT COUNT(*) FROM citations AS c
            JOIN papers AS citing ON citing.id = c.citing_id
            JOIN papers AS cited ON cited.id = c.cited_id
            """
        ).fetchone()[0]
        connected_rows = self.connection.execute(
            """
            SELECT citing_id AS node FROM citations JOIN papers ON papers.id = citing_id
            UNION
            SELECT cited_id AS node FROM citations JOIN papers ON papers.id = cited_id
            """
        ).fetchall()
        return {"nodes": self.count(), "edges": edge_count, "connected_nodes": len(connected_rows)}

    def count(self) -> int:
        """Return the number of stored papers without loading rows."""

        return self.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def stats(self) -> dict[str, int | str]:
        """Return storage statistics for diagnostics and the API."""

        return {"storage_backend": self.backend_name, "stored_papers": self.count()}

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
