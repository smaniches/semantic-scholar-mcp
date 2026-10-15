"""
Provider-contract fidelity tests (S2-01).

Each test goes through the real FastMCP registration (``list_tools`` /
``call_tool``) with respx-mocked upstream HTTP, and pins one documented
Semantic Scholar contract:

- ``/paper/search`` takes a plain-text query; only ``/paper/search/bulk``
  has special query syntax, so discovery text must not promise boolean
  operators or quoted phrases for ordinary search.
- ``/paper/search/match`` answers "no title match" with HTTP 404; the match
  tool reports that as an ordinary no-match, while other statuses (and 404s
  on ID-addressed endpoints) still fail.
- An influential-citation count is rendered only when the provider supplied
  one: absent/null is omitted, a genuine zero is kept.
- The bulk endpoint may ignore ``limit``; the whole provider page and its
  continuation token are returned unchanged.
- Query-string credentials can appear in the HTTP server's access log, so the
  transport documentation must not claim they are never logged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import respx
from httpx import Response
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import TextContent

from semantic_scholar_mcp import transport
from semantic_scholar_mcp.server import SEMANTIC_SCHOLAR_API_BASE, mcp

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCH_URL = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/match"
BULK_URL = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/bulk"
SEARCH_URL = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"


def _text(result: list) -> str:
    assert len(result) > 0
    block = result[0]
    assert isinstance(block, TextContent)
    return block.text


async def _tool(name: str):
    tools = {t.name: t for t in await mcp.list_tools()}
    return tools[name]


# ---------------------------------------------------------------------------
# 1. Ordinary search is plain text; special syntax belongs to bulk search
# ---------------------------------------------------------------------------


class TestSearchSyntaxContract:
    @pytest.mark.asyncio
    async def test_search_discovery_does_not_promise_boolean_syntax(self):
        tool = await _tool("semantic_scholar_search_papers")
        query_desc = tool.inputSchema["$defs"]["PaperSearchInput"]["properties"]["query"][
            "description"
        ]
        for text in (tool.description or "", query_desc):
            assert "AND, OR, NOT" not in text
            assert "quoted phrases" not in text
            assert "supports boolean" not in text.lower()
        assert "plain-text" in query_desc.lower()
        assert "semantic_scholar_bulk_search" in query_desc

    @pytest.mark.asyncio
    async def test_bulk_search_discovery_carries_syntax_guidance(self):
        tool = await _tool("semantic_scholar_bulk_search")
        query_desc = tool.inputSchema["$defs"]["BulkSearchInput"]["properties"]["query"][
            "description"
        ]
        assert "+" in query_desc and "|" in query_desc and '"' in query_desc

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_query_is_forwarded_verbatim(self, reset_all):
        route = respx.get(SEARCH_URL).mock(
            return_value=Response(200, json={"total": 0, "offset": 0, "data": []})
        )
        q = '"graph neural network" AND drug'
        await mcp.call_tool("semantic_scholar_search_papers", {"params": {"query": q}})
        assert route.calls.last.request.url.params["query"] == q

    def test_readme_search_row_does_not_promise_boolean_syntax(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "supports AND, OR, NOT operators" not in readme


# ---------------------------------------------------------------------------
# 2. Title match: provider 404 is a no-match, other failures stay failures
# ---------------------------------------------------------------------------


class TestMatchPaperNoMatch:
    @respx.mock
    @pytest.mark.asyncio
    async def test_provider_404_is_ordinary_no_match(self, reset_all):
        respx.get(MATCH_URL).mock(
            return_value=Response(404, json={"error": "Title match not found"})
        )
        result = await mcp.call_tool(
            "semantic_scholar_match_paper", {"params": {"query": "no such title zzqx"}}
        )
        text = _text(result)
        assert text == "No matching paper found."
        assert "Check ID format" not in text

    @respx.mock
    @pytest.mark.asyncio
    async def test_provider_404_json_format_is_no_match(self, reset_all):
        respx.get(MATCH_URL).mock(
            return_value=Response(404, json={"error": "Title match not found"})
        )
        result = await mcp.call_tool(
            "semantic_scholar_match_paper",
            {"params": {"query": "no such title zzqx", "response_format": "json"}},
        )
        assert _text(result) == "No matching paper found."

    @respx.mock
    @pytest.mark.asyncio
    async def test_match_200_still_returns_paper(self, reset_all):
        respx.get(MATCH_URL).mock(
            return_value=Response(
                200,
                json={"data": [{"paperId": "p1", "title": "Attention", "matchScore": 99.5}]},
            )
        )
        result = await mcp.call_tool(
            "semantic_scholar_match_paper", {"params": {"query": "Attention"}}
        )
        assert "**Match Score:** 99.5" in _text(result)

    @respx.mock
    @pytest.mark.asyncio
    async def test_match_500_still_fails(self, reset_all):
        respx.get(MATCH_URL).mock(return_value=Response(500))
        with pytest.raises(ToolError):
            await mcp.call_tool("semantic_scholar_match_paper", {"params": {"query": "x"}})

    @respx.mock
    @pytest.mark.asyncio
    async def test_404_elsewhere_still_fails(self, reset_all):
        respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/paper/DOI:10.0/none").mock(
            return_value=Response(404, json={"error": "Paper not found"})
        )
        with pytest.raises(ToolError, match="Not found"):
            await mcp.call_tool(
                "semantic_scholar_get_paper", {"params": {"paper_id": "DOI:10.0/none"}}
            )


# ---------------------------------------------------------------------------
# 3. Influential-citation rendering: absent/None omitted, zero preserved
# ---------------------------------------------------------------------------


def _bulk_page(papers: list[dict], token: str | None = None) -> dict:
    page: dict = {"total": 5000, "data": papers}
    if token is not None:
        page["token"] = token
    return page


class TestInfluentialCountFidelity:
    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_markdown_has_no_invented_influential_zero(self, reset_all):
        # Bulk search never requests influentialCitationCount, so the provider
        # does not supply it; the rendering must not invent one.
        papers = [{"paperId": "a", "title": "A", "year": 2020, "citationCount": 7}]
        respx.get(BULK_URL).mock(return_value=Response(200, json=_bulk_page(papers)))
        text = _text(
            await mcp.call_tool("semantic_scholar_bulk_search", {"params": {"query": "x"}})
        )
        assert "**Citations:** 7" in text
        assert "influential" not in text

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_does_not_request_influential_field(self, reset_all):
        route = respx.get(BULK_URL).mock(return_value=Response(200, json=_bulk_page([])))
        await mcp.call_tool("semantic_scholar_bulk_search", {"params": {"query": "x"}})
        fields = route.calls.last.request.url.params["fields"].split(",")
        assert "influentialCitationCount" not in fields

    @pytest.mark.parametrize(
        ("paper", "expected", "absent"),
        [
            ({"citationCount": 3}, "**Citations:** 3", "influential"),
            (
                {"citationCount": 3, "influentialCitationCount": None},
                "**Citations:** 3",
                "influential",
            ),
            (
                {"citationCount": 3, "influentialCitationCount": 0},
                "**Citations:** 3 (0 influential)",
                None,
            ),
            (
                {"citationCount": 3, "influentialCitationCount": 2},
                "**Citations:** 3 (2 influential)",
                None,
            ),
        ],
        ids=["missing", "none", "explicit-zero", "nonzero"],
    )
    def test_formatter_distinguishes_absent_from_zero(self, paper, expected, absent):
        from semantic_scholar_mcp.formatters import format_paper_markdown

        text = format_paper_markdown({"title": "T", **paper})
        assert expected in text
        if absent:
            assert absent not in text
        assert "None" not in text

    @respx.mock
    @pytest.mark.asyncio
    async def test_regular_search_keeps_supplied_counts(self, reset_all):
        respx.get(SEARCH_URL).mock(
            return_value=Response(
                200,
                json={
                    "total": 2,
                    "offset": 0,
                    "data": [
                        {
                            "paperId": "z",
                            "title": "Z",
                            "citationCount": 4,
                            "influentialCitationCount": 0,
                        },
                        {
                            "paperId": "n",
                            "title": "N",
                            "citationCount": 9,
                            "influentialCitationCount": 5,
                        },
                    ],
                },
            )
        )
        text = _text(
            await mcp.call_tool("semantic_scholar_search_papers", {"params": {"query": "x"}})
        )
        assert "**Citations:** 4 (0 influential)" in text
        assert "**Citations:** 9 (5 influential)" in text


# ---------------------------------------------------------------------------
# 4. Bulk limit: full provider page and continuation token preserved
# ---------------------------------------------------------------------------


class TestBulkPagination:
    @respx.mock
    @pytest.mark.asyncio
    async def test_page_larger_than_limit_is_returned_whole(self, reset_all):
        papers = [{"paperId": f"p{i}", "title": f"T{i}"} for i in range(1000)]
        route = respx.get(BULK_URL).mock(
            return_value=Response(200, json=_bulk_page(papers, token="NEXT"))
        )
        out = json.loads(
            _text(
                await mcp.call_tool(
                    "semantic_scholar_bulk_search",
                    {"params": {"query": "x", "limit": 5, "response_format": "json"}},
                )
            )
        )
        assert route.calls.last.request.url.params["limit"] == "5"
        assert out["papers"] == papers
        assert out["token"] == "NEXT"
        assert out["total"] == 5000

    @respx.mock
    @pytest.mark.asyncio
    async def test_token_round_trip_reaches_every_row(self, reset_all):
        page1 = [{"paperId": f"a{i}", "title": "A"} for i in range(3)]
        page2 = [{"paperId": f"b{i}", "title": "B"} for i in range(2)]

        def responder(request):
            if request.url.params.get("token") == "T1":
                return Response(200, json=_bulk_page(page2))
            return Response(200, json=_bulk_page(page1, token="T1"))

        respx.get(BULK_URL).mock(side_effect=responder)
        seen: list[str] = []
        token = None
        for _ in range(3):
            args: dict = {"query": "x", "limit": 1, "response_format": "json"}
            if token:
                args["token"] = token
            out = json.loads(
                _text(await mcp.call_tool("semantic_scholar_bulk_search", {"params": args}))
            )
            seen += [p["paperId"] for p in out["papers"]]
            token = out.get("token")
            if not token:
                break
        assert seen == [p["paperId"] for p in page1 + page2]

    @respx.mock
    @pytest.mark.asyncio
    async def test_markdown_reports_shown_count_and_token(self, reset_all):
        papers = [{"paperId": f"p{i}", "title": f"T{i}"} for i in range(7)]
        respx.get(BULK_URL).mock(return_value=Response(200, json=_bulk_page(papers, token="NX")))
        text = _text(
            await mcp.call_tool(
                "semantic_scholar_bulk_search", {"params": {"query": "x", "limit": 2}}
            )
        )
        assert "(showing 7)" in text
        assert "`NX`" in text

    @pytest.mark.asyncio
    async def test_limit_description_discloses_provider_behavior(self):
        tool = await _tool("semantic_scholar_bulk_search")
        desc = tool.inputSchema["$defs"]["BulkSearchInput"]["properties"]["limit"]["description"]
        assert "ignore" in desc.lower()
        assert "1000" in desc

    def test_readme_bulk_limit_row_discloses_provider_behavior(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("### 7. `semantic_scholar_bulk_search`", 1)[1].split("### 8.", 1)[0]
        limit_row = next(line for line in section.splitlines() if line.startswith("| `limit`"))
        assert "ignore" in limit_row.lower()


# ---------------------------------------------------------------------------
# 5. Logging language for query-string credentials
# ---------------------------------------------------------------------------


class TestCredentialLoggingLanguage:
    def test_transport_doc_qualifies_query_string_logging(self):
        doc = " ".join((transport.__doc__ or "").split())
        assert "it is never logged" not in doc
        assert "access log" in doc

    def test_security_md_qualifies_query_string_logging(self):
        sec = " ".join((REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8").split())
        assert "is never written to the server's own logs" not in sec
        assert "access log" in sec
