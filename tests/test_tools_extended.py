"""
Extended tool tests covering pagination, JSON responses, edge cases,
and additional formatting scenarios not covered in test_server.py.
"""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response
from mcp.server.fastmcp.exceptions import ToolError

from semantic_scholar_mcp.server import (
    RECOMMENDATIONS_BASE,
    SEMANTIC_SCHOLAR_API_BASE,
    AuthorDetailsInput,
    AuthorSearchInput,
    BulkPaperInput,
    PaperDetailsInput,
    PaperRecommendationsInput,
    PaperSearchInput,
    ResponseFormat,
    _format_author_markdown,
    _format_paper_markdown,
    get_author_details,
    get_bulk_papers,
    get_paper_details,
    get_recommendations,
    search_authors,
    search_papers,
    server_status,
)

# ===============================================================================
# PAGINATION EDGE CASES
# ===============================================================================


class TestSearchPagination:
    """Test pagination messaging in search results."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_pagination_hint_shown(self, reset_all):
        """When more results exist, pagination hint should appear."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 100,
                    "data": [
                        {
                            "paperId": str(i),
                            "title": f"Paper {i}",
                            "year": 2024,
                            "citationCount": 0,
                            "influentialCitationCount": 0,
                        }
                        for i in range(10)
                    ],
                },
            )
        )

        params = PaperSearchInput(query="test", limit=10, offset=0)
        result = await search_papers(params)

        assert "offset=10" in result
        assert "showing 1-10" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_pagination_hint_at_end(self, reset_all):
        """When all results shown, no pagination hint should appear."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 3,
                    "data": [
                        {
                            "paperId": str(i),
                            "title": f"Paper {i}",
                            "year": 2024,
                            "citationCount": 0,
                            "influentialCitationCount": 0,
                        }
                        for i in range(3)
                    ],
                },
            )
        )

        params = PaperSearchInput(query="test", limit=10, offset=0)
        result = await search_papers(params)

        assert "offset=" not in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_search_results(self, reset_all):
        """Empty results should still show header."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 0,
                    "data": [],
                },
            )
        )

        params = PaperSearchInput(query="nonexistent gibberish")
        result = await search_papers(params)

        assert "Search Results" in result
        assert "Found:** 0" in result


# ===============================================================================
# JSON RESPONSE FORMAT FOR ALL TOOLS
# ===============================================================================


class TestJsonResponses:
    """Test JSON output mode for tools that support it."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_details_json(self, reset_all):
        """get_paper_details JSON format should be valid parseable JSON."""
        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "paperId": paper_id,
                    "title": "Test",
                    "year": 2024,
                    "citationCount": 0,
                    "influentialCitationCount": 0,
                },
            )
        )

        params = PaperDetailsInput(paper_id=paper_id, response_format=ResponseFormat.JSON)
        result = await get_paper_details(params)
        parsed = json.loads(result)

        assert "paper" in parsed
        assert parsed["paper"]["title"] == "Test"

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_authors_json(self, reset_all):
        """search_authors JSON should include query and total."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 1,
                    "data": [{"authorId": "1", "name": "Test Author"}],
                },
            )
        )

        params = AuthorSearchInput(query="Test", response_format=ResponseFormat.JSON)
        result = await search_authors(params)
        parsed = json.loads(result)

        assert parsed["query"] == "Test"
        assert parsed["total"] == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_author_details_json(self, reset_all):
        """get_author_details JSON should include author and papers."""
        author_id = "123"
        base_url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}"
        papers_url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}/papers"

        respx.get(base_url).mock(
            return_value=Response(
                200,
                json={
                    "authorId": author_id,
                    "name": "Smith",
                },
            )
        )
        respx.get(papers_url).mock(
            return_value=Response(
                200,
                json={
                    "data": [{"title": "Paper 1"}],
                },
            )
        )

        params = AuthorDetailsInput(author_id=author_id, response_format=ResponseFormat.JSON)
        result = await get_author_details(params)
        parsed = json.loads(result)

        assert "author" in parsed
        assert "papers" in parsed

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_recommendations_json(self, reset_all):
        """get_recommendations JSON should include seed and recommendations."""
        paper_id = "a" * 40
        url = f"{RECOMMENDATIONS_BASE}/papers/forpaper/{paper_id}"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "recommendedPapers": [{"paperId": "b" * 40, "title": "Rec"}],
                },
            )
        )

        params = PaperRecommendationsInput(paper_id=paper_id, response_format=ResponseFormat.JSON)
        result = await get_recommendations(params)
        parsed = json.loads(result)

        assert parsed["seed"] == paper_id
        assert len(parsed["recommendations"]) == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_papers_markdown_format(self, reset_all):
        """get_bulk_papers should support markdown format."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"
        respx.post(url).mock(
            return_value=Response(
                200,
                json=[
                    {
                        "paperId": "a" * 40,
                        "title": "Paper 1",
                        "year": 2024,
                        "citationCount": 10,
                        "influentialCitationCount": 1,
                    },
                ],
            )
        )

        params = BulkPaperInput(paper_ids=["a" * 40], response_format=ResponseFormat.MARKDOWN)
        result = await get_bulk_papers(params)

        assert "Bulk Retrieval" in result
        assert "Paper 1" in result


# ===============================================================================
# AUTHOR FORMATTING EDGE CASES
# ===============================================================================


class TestAuthorFormattingEdgeCases:
    """Additional edge cases for _format_author_markdown."""

    def test_author_with_none_affiliations(self):
        """None affiliations should not show affiliations line."""
        author = {"name": "Test", "affiliations": None}
        result = _format_author_markdown(author)
        assert "Affiliations" not in result

    def test_author_many_affiliations_truncated(self):
        """More than 3 affiliations should be truncated."""
        author = {
            "name": "Test",
            "affiliations": ["MIT", "Stanford", "Harvard", "Oxford", "Cambridge"],
        }
        result = _format_author_markdown(author)
        assert "MIT" in result
        assert "Stanford" in result
        assert "Harvard" in result
        # Only first 3 shown
        assert "Oxford" not in result

    def test_author_no_homepage(self):
        """Author without homepage should not show homepage line."""
        author = {"name": "Test", "homepage": None}
        result = _format_author_markdown(author)
        assert "Homepage" not in result

    def test_author_no_url(self):
        """Author without url should not show profile link."""
        author = {"name": "Test"}
        result = _format_author_markdown(author)
        assert "Profile" not in result

    def test_author_none_h_index(self):
        """Author with None hIndex should still format."""
        author = {"name": "Test", "hIndex": None}
        result = _format_author_markdown(author)
        assert "h-index" in result
        assert "None" in result  # Shows None value


# ===============================================================================
# PAPER FORMATTING EDGE CASES
# ===============================================================================


class TestPaperFormattingEdgeCases:
    """Additional edge cases for _format_paper_markdown."""

    def test_paper_with_complete_data(self, sample_paper):
        """Complete paper should format all sections."""
        result = _format_paper_markdown(sample_paper)
        assert "Attention Is All You Need (2017)" in result
        assert "Author One" in result
        assert "NeurIPS" in result
        assert "50000" in result
        assert "[PDF]" in result
        assert "Computer Science" in result
        assert "Transformers are great." in result
        assert "DOI:" in result
        assert "ArXiv:" in result

    def test_paper_with_pubmed_id(self):
        """Paper with PubMed ID should show PMID."""
        paper = {
            "title": "Medical Paper",
            "year": 2023,
            "externalIds": {"PubMed": "12345678"},
        }
        result = _format_paper_markdown(paper)
        assert "PMID: 12345678" in result

    def test_paper_abstract_exactly_500(self):
        """Abstract of exactly 500 chars should not be truncated."""
        paper = {"title": "Test", "year": 2024, "abstract": "A" * 500}
        result = _format_paper_markdown(paper)
        assert "..." not in result
        assert "A" * 500 in result

    def test_paper_empty_external_ids(self):
        """Paper with empty externalIds dict should not show IDs."""
        paper = {"title": "Test", "year": 2024, "externalIds": {}}
        result = _format_paper_markdown(paper)
        assert "IDs:" not in result

    def test_paper_no_pdf_url_in_open_access(self):
        """openAccessPdf without url should not show PDF link."""
        paper = {"title": "Test", "year": 2024, "openAccessPdf": {"status": "green"}}
        result = _format_paper_markdown(paper)
        assert "[PDF]" not in result

    def test_paper_empty_fields_of_study(self):
        """Empty fieldsOfStudy list should not show fields."""
        paper = {"title": "Test", "year": 2024, "fieldsOfStudy": []}
        result = _format_paper_markdown(paper)
        assert "Fields" not in result

    def test_paper_tldr_no_text(self):
        """tldr dict without text key should not show TL;DR."""
        paper = {"title": "Test", "year": 2024, "tldr": {"model": "v1"}}
        result = _format_paper_markdown(paper)
        assert "TL;DR" not in result


# ===============================================================================
# TOOL ERROR HANDLING EDGE CASES
# ===============================================================================


class TestToolErrorEdgeCases:
    """Edge cases for tool error handling."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_unexpected_response_type(self, reset_all):
        """get_paper_details should raise ToolError on non-dict response."""
        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        respx.get(url).mock(return_value=Response(200, json=[]))

        params = PaperDetailsInput(paper_id=paper_id)
        with pytest.raises(ToolError, match="Unexpected response"):
            await get_paper_details(params)

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_author_unexpected_response_type(self, reset_all):
        """get_author_details should raise ToolError on non-dict response."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/123"
        respx.get(url).mock(return_value=Response(200, json=[]))

        params = AuthorDetailsInput(author_id="123", include_papers=False)
        with pytest.raises(ToolError, match="Unexpected response"):
            await get_author_details(params)

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_papers_api_error(self, reset_all):
        """search_papers should gracefully handle API errors."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(403))

        params = PaperSearchInput(query="test")
        with pytest.raises(ToolError):
            await search_papers(params)

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_authors_api_error(self, reset_all):
        """search_authors should gracefully handle API errors."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/search"
        respx.get(url).mock(return_value=Response(500))

        params = AuthorSearchInput(query="test")
        with pytest.raises(ToolError):
            await search_authors(params)

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_recommendations_api_error(self, reset_all):
        """get_recommendations should gracefully handle API errors."""
        paper_id = "a" * 40
        url = f"{RECOMMENDATIONS_BASE}/papers/forpaper/{paper_id}"
        respx.get(url).mock(return_value=Response(500))

        params = PaperRecommendationsInput(paper_id=paper_id)
        with pytest.raises(ToolError):
            await get_recommendations(params)

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_papers_api_error(self, reset_all):
        """get_bulk_papers should gracefully handle API errors."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"
        respx.post(url).mock(return_value=Response(500))

        params = BulkPaperInput(paper_ids=["a" * 40])
        with pytest.raises(ToolError):
            await get_bulk_papers(params)

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_status_generic_exception(self, reset_all):
        """server_status should handle non-SemanticScholarError exceptions."""

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(side_effect=RuntimeError("Connection reset"))

        result = await server_status()
        parsed = json.loads(result)
        assert parsed["api_reachable"] is False
        assert "Connection reset" in parsed["error"]


# ===============================================================================
# GET AUTHOR WITHOUT PAPERS
# ===============================================================================


class TestGetAuthorNoPapers:
    """Test get_author_details with include_papers=False."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_author_without_papers(self, reset_all):
        """Should not fetch papers when include_papers=False."""
        author_id = "456"
        base_url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}"

        route = respx.get(base_url).mock(
            return_value=Response(
                200,
                json={
                    "authorId": author_id,
                    "name": "No Papers Author",
                    "hIndex": 0,
                    "paperCount": 0,
                    "citationCount": 0,
                },
            )
        )

        params = AuthorDetailsInput(author_id=author_id, include_papers=False)
        result = await get_author_details(params)

        assert "No Papers Author" in result
        assert "Publications" not in result
        # Only author endpoint should be called, not papers
        assert route.call_count == 1
