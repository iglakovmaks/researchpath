"""Run a reproducible dense-retrieval benchmark on a BEIR dataset.

Example:
    python scripts/benchmark_beir.py --dataset scifact --max-queries 50

The default full run evaluates the selected BEIR test split. Use
--max-corpus/--max-queries only for smoke tests; truncating the corpus or
queries is not a leaderboard-comparable evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


class SentenceTransformerRetriever:
    """Adapter implementing the retrieval interface expected by BEIR."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        if hasattr(self.model, "encode_document"):
            embeddings = self.model.encode_document(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def _encode_queries(self, texts: list[str]) -> np.ndarray:
        if hasattr(self.model, "encode_query"):
            embeddings = self.model.encode_query(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        score_function: str | None = "cos_sim",
        **_: Any,
    ) -> dict[str, dict[str, float]]:
        document_ids = list(corpus)
        documents = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
            for doc_id in document_ids
        ]
        query_ids = list(queries)
        query_texts = [queries[query_id] for query_id in query_ids]
        document_embeddings = self._encode_documents(documents)
        query_embeddings = self._encode_queries(query_texts)
        similarities = query_embeddings @ document_embeddings.T

        results: dict[str, dict[str, float]] = {}
        for row, query_id in enumerate(query_ids):
            ranked = np.argsort(-similarities[row])[:top_k]
            results[query_id] = {
                document_ids[index]: float(similarities[row, index]) for index in ranked
            }
        return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--data-dir", default="data/beir")
    parser.add_argument("--output", default="benchmarks/results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-corpus", type=int)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError as error:
        print(
            "Install the benchmark extra with `pip install -e '.[benchmark]'`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from error

    dataset_dir = Path(args.data_dir) / args.dataset
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_dir.is_dir():
        url = (
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
        )
        util.download_and_unzip(url, str(dataset_dir.parent))

    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_dir)).load(split="test")
    if args.max_corpus:
        corpus = dict(list(corpus.items())[: args.max_corpus])
    if args.max_queries:
        queries = dict(list(queries.items())[: args.max_queries])
        qrels = {
            query_id: relevance for query_id, relevance in qrels.items() if query_id in queries
        }

    retriever = SentenceTransformerRetriever(args.model, batch_size=args.batch_size)
    evaluator = EvaluateRetrieval(retriever, score_function="cos_sim", k_values=args.k)
    results = evaluator.retrieve(corpus, queries)
    ndcg, map_scores, recall, precision = evaluator.evaluate(qrels, results, args.k)

    payload = {
        "dataset": args.dataset,
        "split": "test",
        "model": args.model,
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "truncated": bool(args.max_corpus or args.max_queries),
        "metrics": {
            "ndcg": ndcg,
            "map": map_scores,
            "recall": recall,
            "precision": precision,
        },
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}-{args.model.rsplit('/', maxsplit=1)[-1]}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved benchmark result to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
