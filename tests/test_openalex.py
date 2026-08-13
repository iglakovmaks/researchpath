from researchpath.openalex import normalize_openalex_work


def test_openalex_abstract_and_nested_metadata_are_normalized() -> None:
    work = {
        "id": "https://openalex.org/W123",
        "title": "A Small Retrieval Study",
        "publication_year": 2025,
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
        "topics": [{"display_name": "Information Retrieval"}],
        "cited_by_count": 7,
        "referenced_works": ["https://openalex.org/W456"],
        "abstract_inverted_index": {"retrieval": [2], "A": [0], "small": [1], "study": [3]},
        "best_oa_location": {"pdf_url": "https://example.org/paper.pdf"},
    }

    paper = normalize_openalex_work(work)

    assert paper.id == "W123"
    assert paper.abstract == "A small retrieval study"
    assert paper.authors == ["Ada Lovelace"]
    assert paper.topics == ["Information Retrieval"]
    assert paper.referenced_works == ["W456"]
    assert paper.open_access_url.endswith(".pdf")
