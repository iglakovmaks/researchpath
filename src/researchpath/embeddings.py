from __future__ import annotations

from typing import Any

import numpy as np

from researchpath.models import Paper


class SentenceTransformerIndex:
    """Dense semantic index backed by Sentence Transformers."""

    backend_name = "sentence-transformers"

    def __init__(self, papers: list[Paper], model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install the embedding extra with "
                "`pip install -e '.[embeddings]'` to use dense retrieval."
            ) from error

        self.papers = papers
        self.model_name = model_name
        self.model: Any = SentenceTransformer(model_name)
        self.document_embeddings = self._encode_documents(
            [paper.searchable_text for paper in papers]
        )

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        if hasattr(self.model, "encode_document"):
            embeddings = self.model.encode_document(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def _encode_query(self, query: str) -> np.ndarray:
        if hasattr(self.model, "encode_query"):
            embedding = self.model.encode_query(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        return np.asarray(embedding, dtype=np.float32)

    def scores(self, query: str) -> np.ndarray:
        """Return cosine similarity between a query and every paper."""

        if not self.papers:
            return np.zeros(0, dtype=np.float32)
        query_embedding = self._encode_query(query)
        return self.document_embeddings @ query_embedding
