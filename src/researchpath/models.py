from __future__ import annotations

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A normalized scholarly work used by the retrieval and graph layers."""

    class Config:
        extra = "ignore"

    id: str
    title: str
    abstract: str = ""
    publication_year: int = Field(ge=0, le=3000)
    authors: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    cited_by_count: int = Field(default=0, ge=0)
    referenced_works: list[str] = Field(default_factory=list)
    doi: str | None = None
    open_access_url: str | None = None
    source: str = "demo"

    @property
    def searchable_text(self) -> str:
        """Return the fields that should participate in retrieval."""

        return " ".join(
            [
                self.title,
                self.abstract,
                " ".join(self.authors),
                " ".join(self.topics),
            ]
        )


class SearchResult(BaseModel):
    """A paper plus transparent scores and explanations."""

    paper: Paper
    bm25_score: float
    vector_score: float
    final_score: float
    matched_terms: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ReadingPathStep(BaseModel):
    """One recommended step in a topic learning path."""

    position: int
    role: str
    paper: Paper
    score: float
    reason: str
    prerequisites: list[str] = Field(default_factory=list)
