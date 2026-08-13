# BEIR Benchmarking

ResearchPath uses the official BEIR evaluation protocol for dense retrieval.
The benchmark adapter encodes corpus documents and queries with
`sentence-transformers/all-MiniLM-L6-v2`, ranks documents by cosine similarity,
and delegates metric computation to BEIR's `EvaluateRetrieval`.

Run a small smoke test:

~~~bash
pip install -e ".[benchmark]"
python scripts/benchmark_beir.py \
  --dataset scifact \
  --max-queries 20
~~~

Run the full SciFact test split:

~~~bash
python scripts/benchmark_beir.py --dataset scifact
~~~

The smoke-test flags intentionally make the run faster but produce results
that are not comparable with the official leaderboard. Full runs should be
reported with the dataset, split, model, corpus size, query count, metrics,
and hardware details.

Metrics:

- `NDCG@k`: ranking quality with graded relevance.
- `MAP@k`: mean average precision.
- `Recall@k`: relevant documents retrieved.
- `P@k`: precision among the top-k results.
