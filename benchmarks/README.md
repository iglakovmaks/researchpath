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

Compare all ResearchPath backends on the same split:

~~~bash
python scripts/compare_retrievers.py --dataset scifact
~~~

This evaluates BM25, TF-IDF, MiniLM dense retrieval, and a 60/40 BM25+dense
hybrid with identical queries, relevance judgments, cutoffs, and BEIR metrics.
The full comparison is checked in at
`benchmarks/results/scifact-comparison.json`.

Full SciFact comparison (`all-MiniLM-L6-v2`, 5,183 documents, 300 queries):

| Backend | NDCG@10 | MAP@10 | Recall@10 | Retrieval seconds |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.66439 | 0.62096 | 0.78161 | 14.268 |
| TF-IDF | 0.64176 | 0.59169 | 0.77900 | 2.886 |
| Dense MiniLM | 0.64508 | 0.59593 | 0.78333 | 34.661 |
| Hybrid (60/40) | **0.71394** | **0.67019** | **0.83322** | 17.183 |

The smoke-test flags intentionally make the run faster but produce results
that are not comparable with the official leaderboard. Full runs should be
reported with the dataset, split, model, corpus size, query count, metrics,
and hardware details.

Metrics:

- `NDCG@k`: ranking quality with graded relevance.
- `MAP@k`: mean average precision.
- `Recall@k`: relevant documents retrieved.
- `P@k`: precision among the top-k results.
