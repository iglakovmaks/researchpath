from pathlib import Path

from researchpath.data import load_papers
from researchpath.graph import CitationGraph
from researchpath.path import build_reading_path
from researchpath.search import HybridSearchEngine


DATA_PATH = Path(__file__).parents[1] / "data" / "demo_papers.json"


def test_hybrid_search_returns_relevant_information_retrieval_papers() -> None:
    papers = load_papers(DATA_PATH)
    results = HybridSearchEngine(papers).search("information retrieval", limit=5)

    assert results
    assert results[0].paper.id in {"demo-okapi", "demo-pagerank", "demo-learning-to-rank"}
    assert results[0].matched_terms
    assert results[0].reasons


def test_search_is_empty_for_unknown_query() -> None:
    papers = load_papers(DATA_PATH)

    assert HybridSearchEngine(papers).search("quantum bicycles", limit=5) == []


def test_search_supports_each_server_side_retrieval_mode() -> None:
    papers = load_papers(DATA_PATH)
    engine = HybridSearchEngine(papers)

    for mode in ("bm25", "vector", "hybrid"):
        results = engine.search("information retrieval", limit=3, mode=mode)
        assert results
        assert results[0].reasons


def test_reading_path_progresses_through_a_citation_graph() -> None:
    papers = load_papers(DATA_PATH)
    engine = HybridSearchEngine(papers)
    graph = CitationGraph(papers)
    results = engine.search("distributed systems", limit=12)

    path = build_reading_path(results, graph, limit=5)

    assert len(path) == 5
    assert path[0].role == "foundation"
    assert path[-1].role == "frontier"
    assert path[0].paper.publication_year <= path[-1].paper.publication_year
    assert graph.stats()["edges"] > 0
