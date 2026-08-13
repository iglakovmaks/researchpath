from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from researchpath.api import DEFAULT_DATA_PATH, create_app
from researchpath.corpus import migrate_json_to_sqlite
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
        version="ResearchPath 0.4.0",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in [
        ("search", "Search the local paper collection."),
        ("path", "Build a citation-aware reading path."),
    ]:
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("query", help="Topic or research question.")
        command_parser.add_argument("--limit", type=int, default=10 if command == "search" else 6)
        command_parser.add_argument(
            "--data",
            default=str(DEFAULT_DATA_PATH),
            help="Path to a normalized JSON or SQLite corpus.",
        )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Fetch works from OpenAlex and save a normalized local dataset.",
    )
    ingest_parser.add_argument("query", help="OpenAlex search query.")
    ingest_parser.add_argument("--output", default="data/openalex.local.json")
    ingest_parser.add_argument("--database", help="SQLite database to upsert into.")
    ingest_parser.add_argument("--per-page", type=int, default=25)
    ingest_parser.add_argument("--max-results", type=int, default=None)
    ingest_parser.add_argument("--filter", dest="openalex_filter")
    ingest_parser.add_argument(
        "--sort",
        default=None,
        help="Optional OpenAlex sort field. Search relevance is the default.",
    )
    ingest_parser.add_argument("--mailto", help="Contact email for the OpenAlex polite pool.")

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Convert a normalized JSON corpus into SQLite.",
    )
    migrate_parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    migrate_parser.add_argument("--database", default="data/researchpath.db")

    serve_parser = subparsers.add_parser("serve", help="Start the local web application.")
    serve_parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "ingest":
        client = OpenAlexClient(mailto=args.mailto)
        if args.database:
            count = client.search_to_sqlite(
                query=args.query,
                database_path=args.database,
                per_page=args.per_page,
                max_results=args.max_results or args.per_page,
                filter_query=args.openalex_filter,
                sort=args.sort,
            )
            print(f"Upserted {count} papers into {args.database}")
        else:
            papers = client.search_to_file(
                query=args.query,
                output_path=args.output,
                per_page=args.per_page,
            )
            print(f"Saved {len(papers)} papers to {args.output}")
        return 0

    if args.command == "migrate":
        count = migrate_json_to_sqlite(args.data, args.database)
        print(f"Migrated {count} papers into {args.database}")
        return 0

    if args.command == "serve":
        uvicorn.run(create_app(args.data), host=args.host, port=args.port)
        return 0

    service = ResearchPathService.from_path(Path(args.data))
    if args.command == "search":
        _print_json([_dump_model(result) for result in service.search(args.query, args.limit)])
    elif args.command == "path":
        _print_json([_dump_model(step) for step in service.reading_path(args.query, args.limit)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
