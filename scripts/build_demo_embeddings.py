"""Build the browser-side embedding index for the checked-in demo corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from researchpath.data import load_papers
from researchpath.embeddings import SentenceTransformerIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/demo_papers.json")
    parser.add_argument("--output", default="data/demo_embeddings.json")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers model used to encode the demo documents.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    papers = load_papers(args.data)
    index = SentenceTransformerIndex(papers, args.model)
    payload = {
        "model": "Xenova/all-MiniLM-L6-v2",
        "source_model": args.model,
        "dimensions": int(index.document_embeddings.shape[1]),
        "papers": [
            {"id": paper.id, "embedding": embedding.tolist()}
            for paper, embedding in zip(papers, index.document_embeddings, strict=True)
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Saved {len(papers)} embeddings to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
