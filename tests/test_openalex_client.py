import httpx

from researchpath.openalex import OpenAlexClient


def test_openalex_cursor_pagination_stops_at_max_results() -> None:
    requests: list[httpx.QueryParams] = []
    payloads = iter(
        [
            {
                "meta": {"next_cursor": "next-page"},
                "results": [
                    {"id": "https://openalex.org/W1", "title": "First", "publication_year": 2024},
                    {"id": "https://openalex.org/W2", "title": "Second", "publication_year": 2024},
                ],
            },
            {
                "meta": {"next_cursor": None},
                "results": [
                    {"id": "https://openalex.org/W3", "title": "Third", "publication_year": 2024},
                ],
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.params)
        return httpx.Response(200, json=next(payloads), request=request)

    client = OpenAlexClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
        mailto="researchpath@example.org",
    )
    papers = client.search_all("retrieval", max_results=3, per_page=2)

    assert [paper.id for paper in papers] == ["W1", "W2", "W3"]
    assert requests[0]["cursor"] == "*"
    assert requests[1]["cursor"] == "next-page"
    assert requests[0]["mailto"] == "researchpath@example.org"
