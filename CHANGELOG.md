# Changelog

All notable changes to ResearchPath are documented here.

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
