from __future__ import annotations

import os
from typing import Any

import httpx

from researchpath.data import save_papers
from researchpath.models import Paper

OPENALEX_API_URL = "https://api.openalex.org"


def _short_id(value: str | None) -> str:
    """Convert an OpenAlex URL into a stable short identifier."""

    if not value:
        return ""
    return value.rstrip("/").rsplit("/", maxsplit=1)[-1]


def _reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    """Reconstruct OpenAlex's inverted-index abstract representation."""

    if not index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        words.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(words))


def normalize_openalex_work(work: dict[str, Any]) -> Paper:
    """Convert an OpenAlex work response into ResearchPath's data model."""

    authors = [
        authorship.get("author", {}).get("display_name", "")
        for authorship in work.get("authorships", [])
    ]
    authors = [author for author in authors if author]

    topics = [
        topic.get("display_name", "") for topic in work.get("topics", []) if isinstance(topic, dict)
    ]
    topics = [topic for topic in topics if topic]

    location = work.get("best_oa_location") or work.get("primary_location") or {}
    open_access_url = location.get("pdf_url") or location.get("landing_page_url")

    return Paper(
        id=_short_id(work.get("id")),
        title=work.get("title") or "Untitled work",
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
        publication_year=work.get("publication_year") or 0,
        authors=authors,
        topics=topics,
        cited_by_count=work.get("cited_by_count") or 0,
        referenced_works=[_short_id(item) for item in work.get("referenced_works", [])],
        doi=work.get("doi"),
        open_access_url=open_access_url,
        source="openalex",
    )


class OpenAlexClient:
    """Small client for importing papers from the public OpenAlex API."""

    def __init__(
        self,
        base_url: str = OPENALEX_API_URL,
        api_key: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY")
        self.timeout = timeout

    def search(self, query: str, per_page: int = 25) -> list[Paper]:
        """Search OpenAlex and return normalized works."""

        params: dict[str, str | int] = {
            "search": query,
            "per_page": min(max(per_page, 1), 100),
        }
        if self.api_key:
            params["api_key"] = self.api_key

        headers = {"User-Agent": "ResearchPath/0.1 (literature-navigation-demo)"}
        response = httpx.get(
            f"{self.base_url}/works",
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [normalize_openalex_work(work) for work in payload.get("results", [])]

    def search_to_file(
        self,
        query: str,
        output_path: str,
        per_page: int = 25,
    ) -> list[Paper]:
        """Search OpenAlex and persist the normalized result set."""

        papers = self.search(query=query, per_page=per_page)
        save_papers(papers, output_path)
        return papers
