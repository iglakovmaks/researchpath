# ResearchPath

ResearchPath is an explainable research navigator for computer science
literature. Given a topic or research question, it returns ranked papers and a
transparent reading path from foundational work to more recent research.

The project is designed as a learning and research tool rather than a
black-box chatbot. Every recommendation exposes its retrieval scores, matched
terms, topic metadata, and citation relationships.

> Current status: public SQLite FTS5 BM25 demo with selectable retrieval modes,
> browser embeddings, citation-aware path generation, CLI, HTTP API, and BEIR
> comparisons.

Live demo: <https://researchpath-two.vercel.app>

## Why ResearchPath?

Finding papers is easy. Understanding how a field developed is harder.
ResearchPath tries to answer:

> “What should I read first, what should I read next, and why?”

The project combines transparent baselines with an optional dense model so that
retrieval quality can be inspected locally and measured on a public benchmark.

## Features

- BM25 lexical retrieval implemented from scratch.
- TF-IDF vector retrieval as a lightweight local baseline.
- Optional dense semantic retrieval with Sentence Transformers.
- Browser-side dense retrieval with a cached MiniLM ONNX model.
- Hybrid ranking with separate, inspectable component scores.
- Citation graph construction from referenced works.
- Chronological reading-path generation with foundation, core, and frontier
  roles.
- OpenAlex ingestion command for importing real scholarly metadata.
- SQLite corpus storage with FTS5 and cursor-paginated OpenAlex ingestion.
- JSON API and a small browser interface.
- Unit and API tests.

## Demo

Start the local application:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
researchpath serve
~~~

Open <http://127.0.0.1:8000> in a browser.

The checked-in demo corpus is available as both `data/demo_papers.json` and a
derived read-only `data/demo.db`. The public demo runs the SQLite database, so
its server-side default is FTS5 BM25 rather than an in-memory index.

The corpus includes three connected topic areas:

- Distributed systems.
- Information retrieval.
- Machine learning and natural language processing.

To enable dense embeddings locally, install the optional backend and set a
model name:

~~~bash
pip install -e ".[embeddings]"
export RESEARCHPATH_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
researchpath serve
~~~

The first local run downloads the model from Hugging Face and caches it.
The public demo offers a “Dense embeddings (browser)” mode powered by
[Transformers.js](https://huggingface.co/docs/transformers.js/en/installation):
the query is encoded on the user's device and compared with the checked-in
MiniLM document index. The Python API keeps a lightweight TF-IDF baseline for
JSON corpora. SQLite corpora use FTS5 BM25 and hydrate only the top-k metadata
rows.

Run the reproducible BEIR evaluation:

~~~bash
pip install -e ".[benchmark]"
python scripts/benchmark_beir.py --dataset scifact
~~~

The checked-in full SciFact test-split result uses
`sentence-transformers/all-MiniLM-L6-v2` over 5,183 documents and 300 queries:
`NDCG@10 = 0.64508`, `MAP@10 = 0.59593`, and `Recall@10 = 0.78333`.
See [benchmarks/README.md](benchmarks/README.md) and the JSON result for the
exact protocol and all reported metrics.

## Command-line Usage

Search the local corpus:

~~~bash
researchpath search "information retrieval" --limit 5
~~~

Build a reading path:

~~~bash
researchpath path "distributed systems" --limit 6
~~~

Import works from OpenAlex:

~~~bash
researchpath ingest "retrieval augmented generation" \
  --output data/rag.local.json \
  --per-page 25
researchpath serve --data data/rag.local.json
~~~

Store a larger import in SQLite instead of a JSON snapshot. SQLite-backed
services use FTS5 BM25 by default and hydrate only the requested top-k papers:

~~~bash
researchpath ingest "retrieval augmented generation" \
  --database data/researchpath.db \
  --max-results 500 \
  --per-page 100 \
  --mailto you@example.org
researchpath search "dense retrieval" --data data/researchpath.db
researchpath serve --data data/researchpath.db
~~~

Convert the JSON source corpus to SQLite:

~~~bash
researchpath migrate \
  --data data/demo_papers.json \
  --database data/demo.db
~~~

OpenAlex cursor pagination is designed for bounded imports. For very large
exports, use the official OpenAlex snapshot rather than paging the entire
works endpoint; the API supports cursor pagination beyond the basic 10,000
result limit. See the [OpenAlex works API](https://developers.openalex.org/api-reference/works/list-works).

OpenAlex provides a public REST API over a connected graph of scholarly works,
authors, sources, institutions, and topics. ResearchPath normalizes its work
records into a small internal schema while preserving citation identifiers.
See the [OpenAlex API reference](https://developers.openalex.org/api-reference/introduction).

## API

The deployed demo exposes the API and interactive OpenAPI documentation. Its
default backend is the checked-in read-only SQLite corpus.

When the server is running:

- `GET /health` — service health check.
- `GET /api/stats` — corpus and graph statistics.
- `GET /api/search?q=information%20retrieval&limit=10` — ranked results. JSON
  defaults to hybrid retrieval; SQLite defaults to FTS5 BM25.
- `GET /api/search?q=information%20retrieval&mode=bm25&limit=10` — explicit BM25.
- `GET /api/reading-path?q=distributed%20systems&limit=6` — reading path.
- `GET /api/papers` — the normalized demo corpus.
- `GET /api/papers/{paper_id}` — one normalized paper.
- `/docs` — interactive OpenAPI documentation.

## Architecture

~~~text
OpenAlex / JSON corpus
          │
          ▼
   Normalized Paper model
          │
    ┌─────┴─────────────┐
    ▼                   ▼
 SQLite + FTS5      JSON + in-memory indexes
    │                   │
    ▼                   ▼
 SQLite BM25       BM25 + TF-IDF / embeddings
    │                   │
    └─────────┬─────────┘
              ▼
        Ranked results
          │
    ┌─────┴─────┐
    ▼           ▼
 Search API   Citation graph
                    │
                    ▼
             Reading-path heuristic
~~~

The path generator is intentionally deterministic. It starts with an early
high-relevance paper and then selects later candidates using three visible
signals:

1. Retrieval relevance.
2. A direct citation relationship with an already selected paper.
3. Chronological progress through the topic.

This keeps the baseline easy to understand while making dense retrieval
reproducible instead of treating it as an opaque feature.

## Testing

Run the test suite:

~~~bash
pytest
~~~

Run a syntax and style check if Ruff is installed:

~~~bash
ruff check .
~~~

## Roadmap

- [x] Build a local end-to-end MVP.
- [x] Add transparent BM25 and TF-IDF baselines.
- [x] Add citation-aware reading paths.
- [x] Add OpenAlex ingestion.
- [x] Add optional Sentence Transformer embeddings.
- [x] Add a BEIR benchmark harness.
- [x] Compare BM25, TF-IDF, dense, and hybrid retrieval on SciFact.
- [x] Add public browser-side dense search.
- [x] Store normalized corpora in SQLite with an FTS5 index.
- [x] Add cursor-paginated OpenAlex ingestion.
- [x] Run scalable SQLite BM25 without materializing the full corpus in memory.
- [ ] Tune dense embedding retrieval as a separate benchmarked backend.
- [ ] Add interactive benchmark and citation-graph visualizations.
- [ ] Add bilingual query support for English and Russian.

## Data and Limitations

The checked-in JSON dataset and derived SQLite database are small, manually
curated demo fixtures. They make the application reproducible without network
access; they are not a scientific benchmark.

The public web demo uses SQLite FTS5 BM25 on the server and browser-side MiniLM
for dense search over the checked-in curated corpus. SQLite imports use the
same scalable server path and do not build a full in-memory retrieval index.
Dense and hybrid modes remain available for JSON corpora; a future release can
add ANN/vector-database indexing for large-scale semantic retrieval.

## License

ResearchPath is released under the MIT License. The demo metadata is synthetic
and curated for testing. Imported OpenAlex data should be used according to
the source's terms and attribution requirements.
