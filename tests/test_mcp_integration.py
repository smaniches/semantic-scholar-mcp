"""
MCP protocol integration tests.

Tests the full MCP tool registration and invocation via FastMCP.call_tool(),
validating that tools are properly registered, discoverable, and return
correct MCP content blocks.
"""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import TextContent

from semantic_scholar_mcp.server import RECOMMENDATIONS_BASE, SEMANTIC_SCHOLAR_API_BASE, mcp


def _text(result: tuple) -> str:
    """Extract text from call_tool result tuple."""
    content_blocks = result[0]
    assert len(content_blocks) > 0
    block = content_blocks[0]
    assert isinstance(block, TextContent)
    return block.text


# ===============================================================================
# TOOL DISCOVERY
# ===============================================================================


class TestToolDiscovery:
    """Test that all tools are properly registered and discoverable."""

    def test_all_tools_registered(self):
        """All 7 tools should be registered on the FastMCP instance."""
        tool_names = set(mcp._tool_manager._tools.keys())

        expected = {
            "semantic_scholar_search_papers",
            "semantic_scholar_get_paper",
            "semantic_scholar_search_authors",
            "semantic_scholar_get_author",
            "semantic_scholar_recommendations",
            "semantic_scholar_bulk_papers",
            "semantic_scholar_status",
        }
        assert expected == tool_names

    def test_tool_count(self):
        """Should have exactly 7 tools."""
        assert len(mcp._tool_manager._tools) == 7

    def test_server_name(self):
        """Server should have the correct name."""
        assert mcp.name == "semantic_scholar_mcp"

    def test_server_has_instructions(self):
        """Server should have instructions set."""
        assert mcp.instructions is not None
        assert "Semantic Scholar" in mcp.instructions


# ===============================================================================
# TOOL INVOCATIONS VIA FastMCP.call_tool()
# ===============================================================================


class TestSearchPapersIntegration:
    """Test search_papers via MCP call_tool."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_returns_content_blocks(self, reset_all):
        """Tool call should return content blocks."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 1,
                    "data": [
                        {
                            "paperId": "abc123",
                            "title": "Attention Is All You Need",
                            "year": 2017,
                            "citationCount": 50000,
                            "influentialCitationCount": 5000,
                        }
                    ],
                },
            )
        )

        result = await mcp.call_tool(
            "semantic_scholar_search_papers",
            {"params": {"query": "attention transformers"}},
        )

        text = _text(result)
        assert "Attention Is All You Need" in text

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_with_all_filters(self, reset_all):
        """Tool call with all optional filters should work."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(200, json={"total": 0, "data": []}))

        result = await mcp.call_tool(
            "semantic_scholar_search_papers",
            {
                "params": {
                    "query": "deep learning",
                    "year": "2023-2024",
                    "fields_of_study": ["Computer Science"],
                    "open_access_only": True,
                    "limit": 5,
                    "response_format": "json",
                }
            },
        )

        text = _text(result)
        parsed = json.loads(text)
        assert parsed["total"] == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_api_error_raises_tool_error(self, reset_all):
        """API errors should raise ToolError."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(500))

        with pytest.raises(ToolError):
            await mcp.call_tool("semantic_scholar_search_papers", {"params": {"query": "test"}})


class TestGetPaperIntegration:
    """Test get_paper via MCP call_tool."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_by_doi(self, reset_all):
        """Should accept DOI format paper ID."""
        paper_id = "DOI:10.1234/test"
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "paperId": "abc",
                    "title": "Test DOI Paper",
                    "year": 2024,
                    "citationCount": 10,
                    "influentialCitationCount": 1,
                },
            )
        )

        result = await mcp.call_tool(
            "semantic_scholar_get_paper", {"params": {"paper_id": paper_id}}
        )

        text = _text(result)
        assert "Test DOI Paper" in text

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_invalid_id(self, reset_all):
        """Invalid paper ID should raise ToolError."""
        with pytest.raises(ToolError):
            await mcp.call_tool(
                "semantic_scholar_get_paper", {"params": {"paper_id": "not-a-valid-id"}}
            )


class TestGetRecommendationsIntegration:
    """Test recommendations via MCP call_tool."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_recommendations_success(self, reset_all):
        """Should return recommendations for a valid paper."""
        paper_id = "a" * 40
        url = f"{RECOMMENDATIONS_BASE}/papers/forpaper/{paper_id}"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "recommendedPapers": [
                        {
                            "paperId": "b" * 40,
                            "title": "Related Paper",
                            "year": 2024,
                            "citationCount": 20,
                            "influentialCitationCount": 3,
                        }
                    ]
                },
            )
        )

        result = await mcp.call_tool(
            "semantic_scholar_recommendations", {"params": {"paper_id": paper_id}}
        )

        text = _text(result)
        assert "Related Paper" in text


class TestBulkPapersIntegration:
    """Test bulk papers via MCP call_tool."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_success(self, reset_all):
        """Should retrieve multiple papers in one call."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"
        respx.post(url).mock(
            return_value=Response(
                200,
                json=[
                    {"paperId": "a" * 40, "title": "Paper 1", "citationCount": 10},
                    {"paperId": "b" * 40, "title": "Paper 2", "citationCount": 20},
                ],
            )
        )

        result = await mcp.call_tool(
            "semantic_scholar_bulk_papers",
            {"params": {"paper_ids": ["a" * 40, "b" * 40]}},
        )

        text = _text(result)
        parsed = json.loads(text)
        assert parsed["retrieved"] == 2


class TestServerStatusIntegration:
    """Test status tool via MCP call_tool."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_status_check(self, reset_all):
        """Status tool should return JSON with server info."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(200, json={"data": []}))

        result = await mcp.call_tool("semantic_scholar_status", {})

        text = _text(result)
        parsed = json.loads(text)
        assert parsed["server"] == "semantic-scholar-mcp"
        assert "version" in parsed
        assert parsed["api_reachable"] is True


class TestAuthorToolsIntegration:
    """Test author tools via MCP call_tool."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_authors(self, reset_all):
        """Should search and return author results."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 1,
                    "data": [
                        {
                            "authorId": "42",
                            "name": "Yoshua Bengio",
                            "hIndex": 150,
                            "paperCount": 800,
                            "citationCount": 500000,
                        }
                    ],
                },
            )
        )

        result = await mcp.call_tool(
            "semantic_scholar_search_authors", {"params": {"query": "Bengio"}}
        )

        text = _text(result)
        assert "Yoshua Bengio" in text

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_author_details(self, reset_all):
        """Should return author profile."""
        author_id = "42"
        base_url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}"
        papers_url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}/papers"

        respx.get(base_url).mock(
            return_value=Response(
                200,
                json={
                    "authorId": author_id,
                    "name": "Yoshua Bengio",
                    "hIndex": 150,
                    "paperCount": 800,
                    "citationCount": 500000,
                },
            )
        )
        respx.get(papers_url).mock(
            return_value=Response(
                200,
                json={"data": [{"title": "GAN Paper", "year": 2014, "citationCount": 30000}]},
            )
        )

        result = await mcp.call_tool(
            "semantic_scholar_get_author", {"params": {"author_id": author_id}}
        )

        text = _text(result)
        assert "Yoshua Bengio" in text
        assert "GAN Paper" in text
