from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def _normalize(scores: np.ndarray) -> np.ndarray:
    maximum = float(scores.max()) if scores.size else 0.0
    return scores / maximum if maximum > 0 else np.zeros_like(scores)


class HybridSearchEngine:
    """Combine lexical BM25 and TF-IDF vector retrieval with explanations."""

    def __init__(self, papers: list[Paper], bm25_weight: float = 0.6):
        self.papers = papers
        self.bm25_weight = bm25_weight
        self.bm25 = BM25Index(papers)
        self.vector = VectorIndex(papers)
        self._by_id = {paper.id: paper for paper in papers}

    def get_paper(self, paper_id: str) -> Paper | None:
        return self._by_id.get(paper_id)

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search the corpus and return transparent ranked results."""

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        bm25_scores = self.bm25.scores(query)
        vector_scores = self.vector.scores(query)
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
            reasons = ["Hybrid score combines BM25 lexical relevance and TF-IDF vector similarity."]
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
