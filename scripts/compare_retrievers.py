"""Compare ResearchPath retrieval backends on an official BEIR split.

Example:
    python scripts/compare_retrievers.py --dataset scifact

The comparison keeps corpus, queries, qrels, cutoffs, and evaluator fixed
across BM25, TF-IDF, dense embeddings, and a BM25+dense hybrid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from researchpath.models import Paper
from researchpath.search import BM25Index, VectorIndex, _normalize


BACKENDS = ("bm25", "tfidf", "dense", "hybrid")


def corpus_to_papers(corpus: dict[str, dict[str, str]]) -> list[Paper]:
    """Adapt a BEIR corpus to ResearchPath's normalized paper schema."""

    return [
        Paper(
            id=doc_id,
            title=document.get("title", ""),
            abstract=document.get("text", ""),
            publication_year=0,
            source="beir",
        )
        for doc_id, document in corpus.items()
    ]


class ResearchPathRetriever:
    """BEIR adapter for one transparent ResearchPath retrieval backend."""

    def __init__(
        self,
        backend: str,
        papers: list[Paper],
        model_name: str,
        batch_size: int,
        dense_index: Any | None = None,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend: {backend}")
        self.backend = backend
        self.papers = papers
        self.bm25 = BM25Index(papers) if backend in {"bm25", "hybrid"} else None
        self.vector = VectorIndex(papers) if backend == "tfidf" else None
        if backend in {"dense", "hybrid"}:
            if dense_index is None:
                from researchpath.embeddings import SentenceTransformerIndex

                dense_index = SentenceTransformerIndex(papers, model_name)
            self.dense = dense_index
        else:
            self.dense = None
        self.batch_size = batch_size

    def _scores(self, query: str) -> np.ndarray:
        if self.backend == "bm25":
            return self.bm25.scores(query)
        if self.backend == "tfidf":
            return self.vector.scores(query)
        dense_scores = self.dense.scores(query)
        if self.backend == "dense":
            return dense_scores
        return 0.6 * _normalize(self.bm25.scores(query)) + 0.4 * _normalize(dense_scores)

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        score_function: str | None = "cos_sim",
        **_: Any,
    ) -> dict[str, dict[str, float]]:
        """Return BEIR-compatible ranked results."""

        del corpus, score_function
        document_ids = [paper.id for paper in self.papers]
        results: dict[str, dict[str, float]] = {}
        for query_id, query in queries.items():
            scores = self._scores(query)
            ranked = np.argsort(-scores)[:top_k]
            results[query_id] = {document_ids[index]: float(scores[index]) for index in ranked}
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
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    return parser


def load_dataset(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
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
    return GenericDataLoader(data_folder=str(dataset_dir)).load(split="test")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus, queries, qrels = load_dataset(args)
    if args.max_corpus:
        corpus = dict(list(corpus.items())[: args.max_corpus])
    if args.max_queries:
        queries = dict(list(queries.items())[: args.max_queries])
        qrels = {
            query_id: relevance for query_id, relevance in qrels.items() if query_id in queries
        }

    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError as error:
        print(
            "Install the benchmark extra with `pip install -e '.[benchmark]'`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from error

    papers = corpus_to_papers(corpus)
    dense_index = None
    comparison: dict[str, dict[str, Any]] = {}
    for backend in args.backends:
        started = time.perf_counter()
        retriever = ResearchPathRetriever(
            backend,
            papers,
            args.model,
            args.batch_size,
            dense_index=dense_index,
        )
        if backend == "dense":
            dense_index = retriever.dense
        evaluator = EvaluateRetrieval(retriever, score_function="cos_sim", k_values=args.k)
        results = evaluator.retrieve(corpus, queries)
        ndcg, map_scores, recall, precision = evaluator.evaluate(qrels, results, args.k)
        comparison[backend] = {
            "retrieval_seconds": round(time.perf_counter() - started, 3),
            "metrics": {
                "ndcg": ndcg,
                "map": map_scores,
                "recall": recall,
                "precision": precision,
            },
        }

    payload = {
        "dataset": args.dataset,
        "split": "test",
        "model": args.model,
        "backends": args.backends,
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "truncated": bool(args.max_corpus or args.max_queries),
        "results": comparison,
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}-comparison.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved comparison result to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
