import asyncio
from pathlib import Path

import httpx

from researchpath.api import create_app


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
    assert request(app, "GET", "/api/stats").json()["papers"] == 16
    assert request(app, "GET", "/api/search?q=machine%20learning").status_code == 200
    path_response = request(app, "GET", "/api/reading-path?q=machine%20learning&limit=4")
    assert path_response.status_code == 200
    assert len(path_response.json()) == 4


def test_api_returns_404_for_unknown_paper() -> None:
    app = create_app(DATA_PATH)

    assert request(app, "GET", "/api/papers/not-a-paper").status_code == 404
