# ResearchPath

ResearchPath is an explainable research navigator for computer science
literature. Given a topic or research question, it returns ranked papers and a
transparent reading path from foundational work to more recent research.

The project is designed as a learning and research tool rather than a
black-box chatbot. Every recommendation exposes its retrieval scores, matched
terms, topic metadata, and citation relationships.

> Current status: local MVP with a curated demo corpus, hybrid retrieval,
> citation-aware path generation, CLI, HTTP API, and a browser interface.

## Why ResearchPath?

Finding papers is easy. Understanding how a field developed is harder.
ResearchPath tries to answer:

> “What should I read first, what should I read next, and why?”

The first version focuses on transparent algorithms that can be inspected,
tested, and benchmarked. The next versions will add larger OpenAlex datasets,
dense embeddings, and a reproducible retrieval benchmark.

## Features

- BM25 lexical retrieval implemented from scratch.
- TF-IDF vector retrieval as a lightweight local baseline.
- Hybrid ranking with separate, inspectable component scores.
- Citation graph construction from referenced works.
- Chronological reading-path generation with foundation, core, and frontier
  roles.
- OpenAlex ingestion command for importing real scholarly metadata.
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

The demo corpus includes three connected topic areas:

- Distributed systems.
- Information retrieval.
- Machine learning and natural language processing.

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

OpenAlex provides a public REST API over a connected graph of scholarly works,
authors, sources, institutions, and topics. ResearchPath normalizes its work
records into a small internal schema while preserving citation identifiers.
See the [OpenAlex API reference](https://developers.openalex.org/api-reference/introduction).

## API

When the server is running:

- `GET /health` — service health check.
- `GET /api/stats` — corpus and graph statistics.
- `GET /api/search?q=information%20retrieval&limit=10` — ranked results.
- `GET /api/reading-path?q=distributed%20systems&limit=6` — reading path.
- `GET /api/papers/{paper_id}` — one normalized paper.
- `/docs` — interactive OpenAPI documentation.

## Architecture

~~~text
OpenAlex / JSON corpus
          │
          ▼
   Normalized Paper model
          │
    ┌─────┴─────┐
    ▼           ▼
 BM25 index   TF-IDF index
    └─────┬─────┘
          ▼
    Hybrid ranking
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

This makes the first version easy to understand and gives us a clear baseline
before experimenting with learned ranking models.

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
- [ ] Add dense embedding retrieval as a separate benchmarked backend.
- [ ] Store larger corpora in SQLite or DuckDB.
- [ ] Evaluate Recall@k, MRR, nDCG@10, latency, and memory usage.
- [ ] Compare lexical, vector, and hybrid retrieval on public IR datasets such
  as [BEIR](https://arxiv.org/abs/2104.08663).
- [ ] Add interactive citation-graph visualization.
- [ ] Add bilingual query support for English and Russian.

## Data and Limitations

The checked-in dataset is a small, manually curated demo fixture. It is meant
to make the application reproducible without network access; it is not a
scientific benchmark.

The current TF-IDF vector index is a transparent retrieval baseline, not a
modern dense semantic model. The project will only claim an improvement after
publishing a reproducible experiment with fixed data, queries, metrics, and
hardware details.

## License

ResearchPath is released under the MIT License. The demo metadata is synthetic
and curated for testing. Imported OpenAlex data should be used according to
the source's terms and attribution requirements.
