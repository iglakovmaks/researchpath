from pathlib import Path

from researchpath.corpus import SQLiteCorpusStore, migrate_json_to_sqlite


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
