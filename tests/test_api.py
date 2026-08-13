import asyncio
from pathlib import Path

import httpx

from researchpath.api import create_app
from researchpath.corpus import migrate_json_to_sqlite


DATA_PATH = Path(__file__).parents[1] / "data" / "demo_papers.json"


def request(app, method: str, path: str) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path)

    return asyncio.run(send())


def test_api_exposes_search_path_and_health() -> None:
    app = create_app(DATA_PATH)

    assert request(app, "GET", "/health").json() == {"status": "ok"}
    stats = request(app, "GET", "/api/stats").json()
    assert stats["papers"] == 16
    assert stats["retrieval_backend"] == "tf-idf"
    assert stats["storage_backend"] == "json"
    assert request(app, "GET", "/api/search?q=machine%20learning").status_code == 200
    assert request(app, "GET", "/api/search?q=machine%20learning&mode=bm25").status_code == 200
    assert len(request(app, "GET", "/api/papers").json()) == 16
    embedding_index = request(app, "GET", "/api/embedding-index")
    assert embedding_index.status_code == 200
    assert embedding_index.json()["dimensions"] == 384
    path_response = request(app, "GET", "/api/reading-path?q=machine%20learning&limit=4")
    assert path_response.status_code == 200
    assert len(path_response.json()) == 4


def test_api_returns_404_for_unknown_paper() -> None:
    app = create_app(DATA_PATH)

    assert request(app, "GET", "/api/papers/not-a-paper").status_code == 404


def test_api_loads_a_sqlite_corpus(tmp_path: Path) -> None:
    database_path = tmp_path / "demo.db"
    migrate_json_to_sqlite(DATA_PATH, database_path)
    app = create_app(database_path)

    stats = request(app, "GET", "/api/stats").json()
    assert stats["storage_backend"] == "sqlite"
    assert stats["papers"] == 16
