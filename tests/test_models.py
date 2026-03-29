"""
Pydantic model validation tests.

Covers boundary values, invalid inputs, and constraint enforcement
for all input models used by MCP tools.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from semantic_scholar_mcp.server import (
    AuthorDetailsInput,
    AuthorSearchInput,
    BulkPaperInput,
    PaperDetailsInput,
    PaperRecommendationsInput,
    PaperSearchInput,
    ResponseFormat,
)

# ===============================================================================
# PaperSearchInput
# ===============================================================================


class TestPaperSearchInput:
    """Validate PaperSearchInput constraints."""

    def test_minimal_valid(self):
        """Only query is required."""
        m = PaperSearchInput(query="test")
        assert m.query == "test"
        assert m.limit == 10
        assert m.offset == 0
        assert m.response_format == ResponseFormat.MARKDOWN

    def test_all_fields(self):
        """All optional fields should be accepted."""
        m = PaperSearchInput(
            query="deep learning",
            year="2020-2024",
            fields_of_study=["Computer Science"],
            publication_types=["JournalArticle"],
            open_access_only=True,
            min_citation_count=50,
            limit=50,
            offset=100,
            response_format=ResponseFormat.JSON,
            api_key="my-key",
        )
        assert m.open_access_only is True
        assert m.min_citation_count == 50

    def test_query_empty_rejected(self):
        """Empty query should be rejected (min_length=1)."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="")

    def test_query_whitespace_stripped_then_rejected(self):
        """Whitespace-only query should be rejected after stripping."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="   ")

    def test_query_max_length(self):
        """Query over 500 chars should be rejected."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="a" * 501)

    def test_query_exactly_500(self):
        """Query of exactly 500 chars should be accepted."""
        m = PaperSearchInput(query="a" * 500)
        assert len(m.query) == 500

    def test_limit_zero_rejected(self):
        """limit=0 should be rejected (ge=1)."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="test", limit=0)

    def test_limit_101_rejected(self):
        """limit=101 should be rejected (le=100)."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="test", limit=101)

    def test_limit_boundary_1(self):
        """limit=1 should be accepted."""
        m = PaperSearchInput(query="test", limit=1)
        assert m.limit == 1

    def test_limit_boundary_100(self):
        """limit=100 should be accepted."""
        m = PaperSearchInput(query="test", limit=100)
        assert m.limit == 100

    def test_offset_negative_rejected(self):
        """Negative offset should be rejected (ge=0)."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="test", offset=-1)

    def test_min_citation_count_negative_rejected(self):
        """Negative min_citation_count should be rejected (ge=0)."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="test", min_citation_count=-1)

    def test_extra_fields_rejected(self):
        """Extra fields should be rejected (extra='forbid')."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="test", unknown_field="value")

    def test_response_format_invalid(self):
        """Invalid response format should be rejected."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="test", response_format="xml")


# ===============================================================================
# PaperDetailsInput
# ===============================================================================


class TestPaperDetailsInput:
    """Validate PaperDetailsInput constraints."""

    def test_minimal_valid(self):
        """Only paper_id is required."""
        m = PaperDetailsInput(paper_id="a" * 40)
        assert m.include_citations is False
        assert m.include_references is False

    def test_all_fields(self):
        m = PaperDetailsInput(
            paper_id="DOI:10.1234/test",
            include_citations=True,
            include_references=True,
            citations_limit=50,
            references_limit=50,
            response_format=ResponseFormat.JSON,
            api_key="key",
        )
        assert m.include_citations is True
        assert m.citations_limit == 50

    def test_paper_id_empty_rejected(self):
        with pytest.raises(PydanticValidationError):
            PaperDetailsInput(paper_id="")

    def test_citations_limit_zero_rejected(self):
        with pytest.raises(PydanticValidationError):
            PaperDetailsInput(paper_id="a" * 40, citations_limit=0)

    def test_citations_limit_101_rejected(self):
        with pytest.raises(PydanticValidationError):
            PaperDetailsInput(paper_id="a" * 40, citations_limit=101)

    def test_references_limit_boundaries(self):
        m1 = PaperDetailsInput(paper_id="a" * 40, references_limit=1)
        assert m1.references_limit == 1
        m100 = PaperDetailsInput(paper_id="a" * 40, references_limit=100)
        assert m100.references_limit == 100

    def test_extra_fields_rejected(self):
        with pytest.raises(PydanticValidationError):
            PaperDetailsInput(paper_id="a" * 40, extra="nope")


# ===============================================================================
# AuthorSearchInput
# ===============================================================================


class TestAuthorSearchInput:
    """Validate AuthorSearchInput constraints."""

    def test_minimal_valid(self):
        m = AuthorSearchInput(query="Smith")
        assert m.limit == 10

    def test_query_empty_rejected(self):
        with pytest.raises(PydanticValidationError):
            AuthorSearchInput(query="")

    def test_query_max_length_201_rejected(self):
        with pytest.raises(PydanticValidationError):
            AuthorSearchInput(query="a" * 201)

    def test_query_exactly_200(self):
        m = AuthorSearchInput(query="a" * 200)
        assert len(m.query) == 200


# ===============================================================================
# AuthorDetailsInput
# ===============================================================================


class TestAuthorDetailsInput:
    """Validate AuthorDetailsInput constraints."""

    def test_minimal_valid(self):
        m = AuthorDetailsInput(author_id="123")
        assert m.include_papers is True
        assert m.papers_limit == 20

    def test_papers_limit_boundaries(self):
        m1 = AuthorDetailsInput(author_id="123", papers_limit=1)
        assert m1.papers_limit == 1
        m100 = AuthorDetailsInput(author_id="123", papers_limit=100)
        assert m100.papers_limit == 100

    def test_papers_limit_zero_rejected(self):
        with pytest.raises(PydanticValidationError):
            AuthorDetailsInput(author_id="123", papers_limit=0)


# ===============================================================================
# PaperRecommendationsInput
# ===============================================================================


class TestPaperRecommendationsInput:
    """Validate PaperRecommendationsInput constraints."""

    def test_minimal_valid(self):
        m = PaperRecommendationsInput(paper_id="a" * 40)
        assert m.limit == 10

    def test_limit_boundaries(self):
        m1 = PaperRecommendationsInput(paper_id="a" * 40, limit=1)
        assert m1.limit == 1
        m100 = PaperRecommendationsInput(paper_id="a" * 40, limit=100)
        assert m100.limit == 100


# ===============================================================================
# BulkPaperInput
# ===============================================================================


class TestBulkPaperInput:
    """Validate BulkPaperInput constraints."""

    def test_minimal_valid(self):
        m = BulkPaperInput(paper_ids=["a" * 40])
        assert m.response_format == ResponseFormat.JSON  # default is JSON for bulk

    def test_empty_list_rejected(self):
        with pytest.raises(PydanticValidationError):
            BulkPaperInput(paper_ids=[])

    def test_501_ids_rejected(self):
        """More than 500 IDs should be rejected (max_length=500)."""
        with pytest.raises(PydanticValidationError):
            BulkPaperInput(paper_ids=["a" * 40] * 501)

    def test_exactly_500_ids(self):
        m = BulkPaperInput(paper_ids=["a" * 40] * 500)
        assert len(m.paper_ids) == 500

    def test_single_id(self):
        m = BulkPaperInput(paper_ids=["DOI:10.1234/test"])
        assert len(m.paper_ids) == 1
