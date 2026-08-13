from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from researchpath.api import DEFAULT_DATA_PATH, create_app
from researchpath.openalex import OpenAlexClient
from researchpath.service import ResearchPathService


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _dump_model(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchpath",
        description="Explore computer science literature through explainable search paths.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="ResearchPath 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in [
        ("search", "Search the local paper collection."),
        ("path", "Build a citation-aware reading path."),
    ]:
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("query", help="Topic or research question.")
        command_parser.add_argument("--limit", type=int, default=10 if command == "search" else 6)
        command_parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Fetch works from OpenAlex and save a normalized local dataset.",
    )
    ingest_parser.add_argument("query", help="OpenAlex search query.")
    ingest_parser.add_argument("--output", default="data/openalex.local.json")
    ingest_parser.add_argument("--per-page", type=int, default=25)

    serve_parser = subparsers.add_parser("serve", help="Start the local web application.")
    serve_parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "ingest":
        papers = OpenAlexClient().search_to_file(
            query=args.query,
            output_path=args.output,
            per_page=args.per_page,
        )
        print(f"Saved {len(papers)} papers to {args.output}")
        return 0

    if args.command == "serve":
        uvicorn.run(create_app(args.data), host=args.host, port=args.port)
        return 0

    service = ResearchPathService.from_json(Path(args.data))
    if args.command == "search":
        _print_json([_dump_model(result) for result in service.search(args.query, args.limit)])
    elif args.command == "path":
        _print_json([_dump_model(step) for step in service.reading_path(args.query, args.limit)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
