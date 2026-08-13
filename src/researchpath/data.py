from __future__ import annotations

import json
from pathlib import Path

from researchpath.corpus import SQLiteCorpusStore
from researchpath.models import Paper


def load_papers(path: str | Path) -> list[Paper]:
    """Load a JSON list of normalized papers."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON list in {path}")
    validator = getattr(Paper, "model_validate", Paper.parse_obj)
    return [validator(item) for item in payload]


def save_papers(papers: list[Paper], path: str | Path) -> None:
    """Save normalized papers as readable JSON."""

    Path(path).write_text(
        json.dumps(
            [
                paper.model_dump(mode="json") if hasattr(paper, "model_dump") else paper.dict()
                for paper in papers
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_papers_from_path(path: str | Path) -> tuple[list[Paper], str]:
    """Load a JSON or SQLite corpus and report its storage backend."""

    corpus_path = Path(path)
    if corpus_path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with SQLiteCorpusStore(corpus_path) as store:
            return store.all_papers(), store.backend_name
    return load_papers(corpus_path), "json"
