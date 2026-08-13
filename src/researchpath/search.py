from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from researchpath.corpus import SQLiteCorpusStore
from researchpath.embeddings import SentenceTransformerIndex
from researchpath.models import Paper, SearchResult

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize English literature metadata for the lexical index."""

    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """A compact, dependency-free BM25 implementation for the demo corpus."""

    def __init__(self, papers: list[Paper], k1: float = 1.5, b: float = 0.75):
        self.papers = papers
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(paper.searchable_text) for paper in papers]
        self.document_lengths = [len(document) for document in self.documents]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )
        self.document_frequency: Counter[str] = Counter()
        for document in self.documents:
            self.document_frequency.update(set(document))

    def scores(self, query: str) -> np.ndarray:
        """Score every document against a query using BM25."""

        query_terms = tokenize(query)
        scores = np.zeros(len(self.papers), dtype=float)
        if not query_terms or not self.papers:
            return scores

        document_count = len(self.papers)
        for index, document in enumerate(self.documents):
            frequencies = Counter(document)
            length_factor = (
                self.document_lengths[index] / self.average_document_length
                if self.average_document_length
                else 1.0
            )
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_factor)
                scores[index] += idf * numerator / denominator
        return scores


class VectorIndex:
    """A transparent TF-IDF vector baseline for semantic retrieval experiments."""

    def __init__(self, papers: list[Paper]):
        self.papers = papers
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        texts = [paper.searchable_text for paper in papers]
        self.document_matrix = self.vectorizer.fit_transform(texts) if texts else np.empty((0, 0))

    def scores(self, query: str) -> np.ndarray:
        """Return cosine similarity between the query and every document."""

        if not self.papers:
            return np.zeros(0, dtype=float)
        query_vector = self.vectorizer.transform([query])
        return cosine_similarity(query_vector, self.document_matrix).ravel()


class SQLiteBM25Index:
    """BM25 retrieval delegated to SQLite FTS5's indexed corpus."""

    backend_name = "sqlite-fts5"

    def __init__(self, store: SQLiteCorpusStore):
        self.store = store

    def search(self, query: str, limit: int) -> list[tuple[Paper, float]]:
        return self.store.search_scored(query, limit)


def _normalize(scores: np.ndarray) -> np.ndarray:
    maximum = float(scores.max()) if scores.size else 0.0
    return scores / maximum if maximum > 0 else np.zeros_like(scores)


class HybridSearchEngine:
    """Combine lexical and vector retrieval with inspectable explanations."""

    def __init__(
        self,
        papers: list[Paper],
        bm25_weight: float = 0.6,
        embedding_model: str | None = None,
        corpus_store: SQLiteCorpusStore | None = None,
    ):
        self.papers = papers
        self.bm25_weight = bm25_weight
        self.corpus_store = corpus_store
        self.bm25 = SQLiteBM25Index(corpus_store) if corpus_store else BM25Index(papers)
        self.vector = None
        if papers:
            self.vector = (
                SentenceTransformerIndex(papers, embedding_model)
                if embedding_model
                else VectorIndex(papers)
            )
        self.vector_backend = getattr(self.vector, "backend_name", "tf-idf")
        self.bm25_backend = getattr(self.bm25, "backend_name", "python-bm25")
        self._by_id = {paper.id: paper for paper in papers}

    def get_paper(self, paper_id: str) -> Paper | None:
        if self.corpus_store:
            return self.corpus_store.get(paper_id)
        return self._by_id.get(paper_id)

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        """Search the corpus using BM25, vector, or hybrid ranking."""

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        if mode not in {"hybrid", "bm25", "vector"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        if self.corpus_store:
            if mode != "bm25":
                raise ValueError(
                    "SQLite-backed corpora currently support scalable bm25 mode; "
                    "load a JSON corpus to use vector or hybrid mode."
                )
            return self._search_sqlite(query, limit)

        bm25_scores = self.bm25.scores(query)
        if self.vector is None:
            return []
        vector_scores = self.vector.scores(query)
        if mode == "bm25":
            final_scores = _normalize(bm25_scores)
        elif mode == "vector":
            final_scores = _normalize(vector_scores)
        else:
            final_scores = self.bm25_weight * _normalize(bm25_scores) + (
                1 - self.bm25_weight
            ) * _normalize(vector_scores)

        ranked_indices = np.argsort(-final_scores)
        results: list[SearchResult] = []
        for index in ranked_indices:
            if final_scores[index] <= 0:
                continue
            paper = self.papers[int(index)]
            title_terms = set(tokenize(paper.title))
            topic_terms = set(tokenize(" ".join(paper.topics)))
            matched_terms = sorted(query_tokens & set(tokenize(paper.searchable_text)))
            reasons = [
                {
                    "bm25": "Ranked by BM25 lexical relevance.",
                    "vector": f"Ranked by {self.vector_backend} similarity.",
                    "hybrid": "Hybrid score combines BM25 lexical relevance and "
                    f"{self.vector_backend} similarity.",
                }[mode]
            ]
            title_matches = sorted(query_tokens & title_terms)
            topic_matches = sorted(query_tokens & topic_terms)
            if title_matches:
                reasons.append(f"Matches title terms: {', '.join(title_matches)}.")
            elif topic_matches:
                reasons.append(f"Matches topic metadata: {', '.join(topic_matches)}.")
            else:
                reasons.append("Matches the abstract or author metadata.")
            results.append(
                SearchResult(
                    paper=paper,
                    bm25_score=round(float(bm25_scores[index]), 6),
                    vector_score=round(float(vector_scores[index]), 6),
                    final_score=round(float(final_scores[index]), 6),
                    matched_terms=matched_terms,
                    reasons=reasons,
                )
            )
            if len(results) >= max(1, limit):
                break
        return results

    def _search_sqlite(self, query: str, limit: int) -> list[SearchResult]:
        """Build results from only the top rows returned by SQLite FTS5."""

        scored_papers = self.bm25.search(query, max(1, limit))
        if not scored_papers:
            return []
        raw_scores = np.asarray([score for _, score in scored_papers], dtype=float)
        final_scores = _normalize(raw_scores)
        query_tokens = set(tokenize(query))
        results: list[SearchResult] = []
        for index, (paper, bm25_score) in enumerate(scored_papers):
            title_terms = set(tokenize(paper.title))
            topic_terms = set(tokenize(" ".join(paper.topics)))
            matched_terms = sorted(query_tokens & set(tokenize(paper.searchable_text)))
            reasons = ["Ranked by SQLite FTS5 BM25 without materializing the full corpus."]
            title_matches = sorted(query_tokens & title_terms)
            topic_matches = sorted(query_tokens & topic_terms)
            if title_matches:
                reasons.append(f"Matches title terms: {', '.join(title_matches)}.")
            elif topic_matches:
                reasons.append(f"Matches topic metadata: {', '.join(topic_matches)}.")
            else:
                reasons.append("Matches the abstract or author metadata.")
            results.append(
                SearchResult(
                    paper=paper,
                    bm25_score=round(float(bm25_score), 6),
                    vector_score=0.0,
                    final_score=round(float(final_scores[index]), 6),
                    matched_terms=matched_terms,
                    reasons=reasons,
                )
            )
        return results
