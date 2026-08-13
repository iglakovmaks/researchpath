from __future__ import annotations

from researchpath.graph import CitationGraph
from researchpath.models import ReadingPathStep, SearchResult


def _connection_count(
    candidate: SearchResult,
    selected: list[SearchResult],
    graph: CitationGraph,
) -> int:
    return sum(graph.is_connected(candidate.paper.id, previous.paper.id) for previous in selected)


def build_reading_path(
    search_results: list[SearchResult],
    graph: CitationGraph,
    limit: int = 6,
) -> list[ReadingPathStep]:
    """Select a chronological, citation-aware path from search results.

    This is deliberately a transparent heuristic rather than a black-box
    recommendation model. Relevance, citation links, and chronology are
    visible and easy to benchmark or replace with a learned ranker later.
    """

    if not search_results:
        return []

    candidates = search_results[: max(limit * 4, limit)]
    maximum_relevance = max(result.final_score for result in candidates)
    relevant_candidates = [
        result for result in candidates if result.final_score >= maximum_relevance * 0.45
    ]
    first = min(
        relevant_candidates or candidates,
        key=lambda result: (
            result.paper.publication_year,
            -result.final_score,
        ),
    )
    selected = [first]

    while len(selected) < min(limit, len(candidates)):
        remaining = [candidate for candidate in candidates if candidate not in selected]
        previous_year = selected[-1].paper.publication_year

        def selection_score(candidate: SearchResult) -> float:
            relevance = candidate.final_score
            connection = min(
                _connection_count(candidate, selected, graph) / max(len(selected), 1),
                1.0,
            )
            chronological_progress = (
                1.0 if candidate.paper.publication_year >= previous_year else 0.0
            )
            return 0.60 * relevance + 0.25 * connection + 0.15 * chronological_progress

        selected.append(max(remaining, key=selection_score))

    steps: list[ReadingPathStep] = []
    for position, result in enumerate(selected, start=1):
        if position == 1:
            role = "foundation"
            reason = (
                "Earliest high-relevance result in the candidate set; use it to "
                "establish the vocabulary and problem framing."
            )
        elif position == len(selected):
            role = "frontier"
            reason = (
                "Most recent step selected by the relevance-and-graph heuristic; "
                "use it to see how the topic is being extended."
            )
        else:
            role = "core"
            linked_titles = [
                previous.paper.title
                for previous in selected[: position - 1]
                if graph.is_connected(result.paper.id, previous.paper.id)
            ]
            if linked_titles:
                reason = (
                    "Selected as a core step because it is relevant and connected "
                    f"to: {', '.join(linked_titles[:2])}."
                )
            else:
                reason = (
                    "Selected as a core step because it adds a relevant method or "
                    "perspective while moving forward chronologically."
                )

        prerequisites = [
            previous.paper.title
            for previous in selected[: position - 1]
            if graph.is_connected(result.paper.id, previous.paper.id)
        ][:2]
        steps.append(
            ReadingPathStep(
                position=position,
                role=role,
                paper=result.paper,
                score=result.final_score,
                reason=reason,
                prerequisites=prerequisites,
            )
        )
    return steps
