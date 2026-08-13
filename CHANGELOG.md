# Changelog

All notable changes to ResearchPath are documented here.

## [0.5.0] - 2026-08-13

### Added

- SQLite FTS5 BM25 retrieval that returns only the requested top-k metadata rows.
- SQL-backed citation graph queries and corpus statistics for SQLite services.
- Explicit API validation for retrieval modes unsupported by the scalable SQLite backend.

## [0.4.0] - 2026-08-13

### Added

- SQLite corpus store with JSON serialization and FTS5 lookup support.
- JSON-to-SQLite migration command and SQLite-aware service loading.
- Cursor-paginated OpenAlex imports with bounded result counts.
- OpenAlex `mailto` polite-pool support and transient request retries.
- Storage backend diagnostics in `/api/stats` and the browser interface.

## [0.3.0] - 2026-08-13

### Added

- Server-side BM25, vector, and hybrid retrieval modes through the API.
- Browser-side dense search using the MiniLM ONNX embedding index.
- Public `/api/papers` and `/api/embedding-index` demo data endpoints.
- Reproducible comparison of BM25, TF-IDF, dense, and hybrid retrieval on
  the full SciFact test split.
- Checked-in comparison metrics and retrieval timing results.

## [0.2.1] - 2026-08-13

### Fixed

- Added the `src` package path to the Vercel Python entrypoint so the public
  FastAPI Function imports correctly at runtime.
- Documented the live demo URL.

## [0.2.0] - 2026-08-13

### Added

- Optional Sentence Transformer dense retrieval with query/document encoding.
- BEIR benchmark harness using the official `EvaluateRetrieval` metrics.
- Full SciFact test-split result for `all-MiniLM-L6-v2`.
- Vercel deployment entrypoint and a public web demo configuration.
- Retrieval backend and corpus statistics in the web UI.

## [0.1.0] - 2026-08-13

### Added

- A reproducible 16-paper demo corpus covering distributed systems,
  information retrieval, and machine learning.
- From-scratch BM25 lexical retrieval.
- TF-IDF vector retrieval baseline.
- Explainable hybrid ranking with matched terms and score breakdowns.
- Citation graph construction and deterministic reading-path generation.
- OpenAlex ingestion through the public REST API.
- CLI commands for search, paths, ingestion, and serving the web app.
- FastAPI endpoints and a browser interface.
- Tests, CI, MIT license, contribution guide, and code of conduct.
