from pathlib import Path

from researchpath.corpus import SQLiteCorpusStore, migrate_json_to_sqlite
from researchpath.service import ResearchPathService


DATA_PATH = Path(__file__).parents[1] / "data" / "demo_papers.json"


def test_json_migration_preserves_papers_and_builds_fts(tmp_path: Path) -> None:
    database_path = tmp_path / "researchpath.db"

    assert migrate_json_to_sqlite(DATA_PATH, database_path) == 16

    with SQLiteCorpusStore(database_path) as store:
        assert store.stats() == {"storage_backend": "sqlite", "stored_papers": 16}
        assert store.get("demo-rag").title.startswith("Retrieval-Augmented")
        assert {paper.id for paper in store.search("distributed systems", limit=3)}


def test_sqlite_upsert_replaces_existing_document(tmp_path: Path) -> None:
    database_path = tmp_path / "researchpath.db"
    migrate_json_to_sqlite(DATA_PATH, database_path)

    with SQLiteCorpusStore(database_path) as store:
        papers = store.all_papers()
        paper = (
            papers[0].model_copy(update={"title": "Updated title"})
            if hasattr(papers[0], "model_copy")
            else papers[0].copy(update={"title": "Updated title"})
        )
        store.upsert(paper)
        assert store.stats()["stored_papers"] == 16
        assert store.get(paper.id).title == "Updated title"


def test_sqlite_service_searches_without_materializing_the_corpus(tmp_path: Path) -> None:
    database_path = tmp_path / "researchpath.db"
    migrate_json_to_sqlite(DATA_PATH, database_path)

    service = ResearchPathService.from_path(database_path)

    assert service.papers == []
    assert service.stats()["retrieval_backend"] == "sqlite-fts5"
    assert service.stats()["papers"] == 16
    results = service.search("distributed systems", limit=3)
    assert len(results) == 3
    assert all("SQLite FTS5 BM25" in result.reasons[0] for result in results)
    assert service.reading_path("distributed systems", limit=3)


def test_sqlite_service_rejects_non_scalable_modes(tmp_path: Path) -> None:
    database_path = tmp_path / "researchpath.db"
    migrate_json_to_sqlite(DATA_PATH, database_path)
    service = ResearchPathService.from_path(database_path)

    try:
        service.search("distributed systems", mode="vector")
    except ValueError as exc:
        assert "scalable bm25 mode" in str(exc)
    else:
        raise AssertionError("SQLite service should reject vector mode")


def test_read_only_sqlite_store_rejects_writes(tmp_path: Path) -> None:
    database_path = tmp_path / "researchpath.db"
    migrate_json_to_sqlite(DATA_PATH, database_path)

    with SQLiteCorpusStore(database_path, read_only=True) as store:
        paper = store.get("demo-rag")
        assert paper is not None
        try:
            store.upsert(paper)
        except RuntimeError as exc:
            assert "read-only" in str(exc)
        else:
            raise AssertionError("Read-only SQLite store should reject writes")
