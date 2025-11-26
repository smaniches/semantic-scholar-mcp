"""Tests for semantic_scholar_mcp server."""

import pytest
from semantic_scholar_mcp.server import (
    PaperSearchInput,
    PaperDetailsInput,
    AuthorSearchInput,
    ResponseFormat,
)


class TestInputModels:
    """Test Pydantic input models."""

    def test_paper_search_input_valid(self):
        params = PaperSearchInput(query="machine learning")
        assert params.query == "machine learning"
        assert params.limit == 10
        assert params.offset == 0
        assert params.response_format == ResponseFormat.MARKDOWN

    def test_paper_search_with_filters(self):
        params = PaperSearchInput(
            query="deep learning",
            year="2023-2024",
            fields_of_study=["Computer Science"],
            min_citation_count=100,
            limit=20
        )
        assert params.year == "2023-2024"
        assert params.fields_of_study == ["Computer Science"]
        assert params.min_citation_count == 100

    def test_paper_details_input(self):
        params = PaperDetailsInput(
            paper_id="DOI:10.1038/nature12373",
            include_citations=True,
            citations_limit=50
        )
        assert params.paper_id == "DOI:10.1038/nature12373"
        assert params.include_citations is True
        assert params.citations_limit == 50

    def test_author_search_input(self):
        params = AuthorSearchInput(query="Yoshua Bengio", limit=5)
        assert params.query == "Yoshua Bengio"
        assert params.limit == 5


class TestResponseFormat:
    """Test response format enum."""

    def test_markdown_format(self):
        assert ResponseFormat.MARKDOWN.value == "markdown"

    def test_json_format(self):
        assert ResponseFormat.JSON.value == "json"
