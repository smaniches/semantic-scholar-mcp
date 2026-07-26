"""Targeted tests closing the remaining line/branch-coverage gaps.

Each test here exercises a specific behavior that the broader suite left
uncovered and asserts a meaningful outcome — not a smoke test. The gaps
addressed (see ``pytest --cov --cov-branch --cov-report=term-missing``):

* ``server._get_accepted_params`` — missing-tool and unresolvable-hint arms.
* ``server._friendly_validation_message`` — the mixed extra/constraint arm
  whose extra field is *not* ``fields``.
* Per-tool optional-parameter, JSON-format, and empty-section markdown arms
  that the happy-path tests skip.
* Cache-hit arms (``get_author_details`` / ``export_citation``).
* ``server.main()`` — both the keyed and keyless startup paths.
* ``client._parse_retry_after`` — HTTP-date branches.
* ``validators.is_valid_paper_id`` — empty-input early return.
* ``logging_config.StructuredFormatter`` — exception-info branch.
"""

from __future__ import annotations

import json
import logging

import pytest
import respx
from httpx import Response
from mcp.server.fastmcp.exceptions import ToolError

from semantic_scholar_mcp import client as client_mod
from semantic_scholar_mcp.client import SEMANTIC_SCHOLAR_API_BASE, _parse_retry_after, get_headers
from semantic_scholar_mcp.logging_config import StructuredFormatter
from semantic_scholar_mcp.server import (
    AuthorBatchInput,
    AuthorDetailsInput,
    BulkPaperInput,
    BulkSearchInput,
    CitationExportInput,
    MultiRecommendInput,
    PaperAuthorsInput,
    PaperDetailsInput,
    PaperMatchInput,
    ResponseFormat,
    SnippetSearchInput,
    _friendly_validation_message,
    _get_accepted_params,
    bulk_search,
    export_citation,
    get_author_batch,
    get_author_details,
    get_bulk_papers,
    get_paper_authors,
    get_paper_details,
    match_paper,
    mcp,
    multi_recommend,
    snippet_search,
)
from semantic_scholar_mcp.validators import is_valid_paper_id

RECOMMENDATIONS_BASE = "https://api.semanticscholar.org/recommendations/v1"


# ===========================================================================
# server._get_accepted_params
# ===========================================================================


class TestGetAcceptedParams:
    """Cover the two non-happy-path arms of _get_accepted_params."""

    def test_missing_tool_returns_empty(self):
        """An unknown tool name yields an empty accepted-params list (line 126)."""
        assert _get_accepted_params("does_not_exist", mcp._tool_manager) == []

    def test_unresolvable_type_hints_returns_empty(self):
        """If get_type_hints() raises, fall back to [] (lines 129-130).

        A fake tool whose callable carries an annotation referencing an
        undefined forward-ref name makes get_type_hints() raise NameError.
        """

        def fn(params: "UndefinedForwardRef") -> None:  # noqa: F821, UP037
            ...

        class _FakeTool:
            pass

        tool = _FakeTool()
        tool.fn = fn

        class _FakeManager:
            def get_tool(self, name: str):
                return tool

        assert _get_accepted_params("whatever", _FakeManager()) == []

    def test_hint_without_model_fields_is_skipped(self):
        """A non-model annotation is skipped, not returned (branch 134->131).

        With only a plain ``str`` hint (no pydantic model), the loop exhausts
        and the function returns the empty default.
        """

        def fn(name: str) -> None: ...

        class _FakeTool:
            pass

        tool = _FakeTool()
        tool.fn = fn

        class _FakeManager:
            def get_tool(self, name: str):
                return tool

        assert _get_accepted_params("whatever", _FakeManager()) == []


# ===========================================================================
# server._friendly_validation_message
# ===========================================================================


class TestFriendlyValidationMixedNonFieldsExtra:
    """Cover the all_msgs arm where the extra field is not named 'fields'."""

    @pytest.mark.asyncio
    async def test_mixed_non_fields_extra_omits_fields_note(self, reset_all):
        """Extra non-'fields' param + a constraint error: combined message,
        no 'field selection' note (branch 177->182).
        """
        with pytest.raises(ToolError) as exc_info:
            await mcp.call_tool(
                "semantic_scholar_search_papers",
                {"params": {"query": "test", "bogus_param": 42, "limit": 0}},
            )
        msg = str(exc_info.value)
        assert "Invalid parameter" in msg
        assert "bogus_param" in msg
        # limit constraint message is folded in:
        assert "limit" in msg
        # The 'fields'-specific note must NOT appear for a non-'fields' extra.
        assert "field selection is managed internally" not in msg

    def test_direct_call_mixed_non_fields_extra(self, reset_all):
        """Direct unit call with a real pydantic error reaches the all_msgs arm."""
        from pydantic import ValidationError as PydanticValidationError

        from semantic_scholar_mcp.models import PaperSearchInput

        with pytest.raises(PydanticValidationError) as exc_info:
            PaperSearchInput(query="x", bogus=1, limit=0)

        msg = _friendly_validation_message(
            "semantic_scholar_search_papers", exc_info.value, mcp._tool_manager
        )
        assert "bogus" in msg
        assert "limit" in msg
        assert "field selection is managed internally" not in msg


# ===========================================================================
# get_paper_details — empty citingPaper / citedPaper markdown arms
# ===========================================================================


class TestPaperDetailsMarkdownEmptySubPapers:
    """Empty nested citingPaper/citedPaper objects are skipped in markdown."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_citing_and_cited_papers_skipped(self, reset_all):
        """Citation/reference entries with empty nested papers produce headers
        but no bullet lines (branches 344->342 and 353->351).
        """
        paper_id = "a" * 40
        base = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        respx.get(base).mock(
            return_value=Response(200, json={"paperId": paper_id, "title": "Seed", "year": 2020})
        )
        respx.get(f"{base}/citations").mock(
            return_value=Response(200, json={"data": [{"citingPaper": {}}]})
        )
        respx.get(f"{base}/references").mock(
            return_value=Response(200, json={"data": [{"citedPaper": {}}]})
        )

        params = PaperDetailsInput(
            paper_id=paper_id,
            include_citations=True,
            include_references=True,
            response_format=ResponseFormat.MARKDOWN,
        )
        result = await get_paper_details(params)

        # Section headers render (data is non-empty)...
        assert "Citing Papers (1 shown)" in result
        assert "References (1 shown)" in result
        # ...but the empty nested papers contribute no bullet lines.
        assert "- **" not in result


# ===========================================================================
# get_author_details — cache hit + error arms
# ===========================================================================


class TestAuthorDetailsCacheAndError:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_hit_skips_fetch(self, reset_all):
        """A cached author short-circuits the network fetch (branch 406->416)."""
        author_id = "777"
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}"
        route = respx.get(url).mock(
            return_value=Response(200, json={"authorId": author_id, "name": "Cached"})
        )

        params = AuthorDetailsInput(
            author_id=author_id, include_papers=False, response_format=ResponseFormat.JSON
        )
        # First call populates the cache.
        await get_author_details(params)
        assert route.call_count == 1

        # Second call must be served from cache: no new author request.
        result = await get_author_details(params)
        assert route.call_count == 1
        parsed = json.loads(result)
        assert parsed["author"]["name"] == "Cached"

    @respx.mock
    @pytest.mark.asyncio
    async def test_api_error_becomes_tool_error(self, reset_all):
        """An API failure is surfaced as ToolError (line 429)."""
        author_id = "888"
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}"
        respx.get(url).mock(return_value=Response(500))

        params = AuthorDetailsInput(author_id=author_id, include_papers=False)
        with pytest.raises(ToolError):
            await get_author_details(params)


# ===========================================================================
# get_bulk_papers — >10 invalid IDs, >20 not-found in markdown
# ===========================================================================


class TestBulkPapersTruncation:
    @pytest.mark.asyncio
    async def test_more_than_ten_invalid_ids_truncated(self, reset_all):
        """>10 invalid IDs: error message truncates and reports the overflow
        count (line 491).
        """
        bad_ids = [f"bad id {i}!" for i in range(12)]
        params = BulkPaperInput(paper_ids=bad_ids)
        with pytest.raises(ToolError) as exc_info:
            await get_bulk_papers(params)
        msg = str(exc_info.value)
        assert "+2 more" in msg

    @respx.mock
    @pytest.mark.asyncio
    async def test_more_than_twenty_not_found_markdown_truncated(self, reset_all):
        """>20 not-found IDs in markdown: list truncates with overflow note
        (line 533).
        """
        ids = [f"{i:040x}" for i in range(25)]
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"
        # All entries null => all not found.
        respx.post(url).mock(return_value=Response(200, json=[None] * 25))

        params = BulkPaperInput(paper_ids=ids, response_format=ResponseFormat.MARKDOWN)
        result = await get_bulk_papers(params)
        assert "Not found (25)" in result
        assert "+5 more" in result


# ===========================================================================
# bulk_search — all optional params + markdown no-token arm
# ===========================================================================


class TestBulkSearchOptionalParams:
    @respx.mock
    @pytest.mark.asyncio
    async def test_all_optional_filters_forwarded(self, reset_all):
        """token/year/fields_of_study/publication_types/min_citation_count are
        all forwarded to the API (lines 560, 562, 564, 566, 568).
        """
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/bulk"
        route = respx.get(url).mock(return_value=Response(200, json={"total": 0, "data": []}))

        params = BulkSearchInput(
            query="topology",
            token="cursor-xyz",
            year="2020-2024",
            fields_of_study=["Computer Science", "Mathematics"],
            publication_types=["JournalArticle"],
            min_citation_count=5,
            response_format=ResponseFormat.JSON,
        )
        await bulk_search(params)

        sent = route.calls[0].request.url.params
        assert sent.get("token") == "cursor-xyz"
        assert sent.get("year") == "2020-2024"
        assert sent.get("fieldsOfStudy") == "Computer Science,Mathematics"
        assert sent.get("publicationTypes") == "JournalArticle"
        assert sent.get("minCitationCount") == "5"

    @respx.mock
    @pytest.mark.asyncio
    async def test_markdown_without_token_omits_next_page(self, reset_all):
        """Last page (no token) markdown omits the next-page hint (branch
        593->595).
        """
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/bulk"
        respx.get(url).mock(
            return_value=Response(
                200, json={"total": 1, "data": [{"paperId": "a" * 40, "title": "T"}]}
            )
        )
        params = BulkSearchInput(query="x", response_format=ResponseFormat.MARKDOWN)
        result = await bulk_search(params)
        assert "Next page token" not in result


# ===========================================================================
# export_citation — cache hit + non-dict response arms
# ===========================================================================


class TestExportCitationCacheAndType:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_hit_skips_fetch(self, reset_all):
        """A cached paper (from a prior detail fetch) supplies citationStyles
        without a second network call (branch 612->620).
        """
        paper_id = "b" * 40
        detail_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        bibtex = "@article{x, title={Cached}}"
        route = respx.get(detail_url).mock(
            return_value=Response(
                200,
                json={
                    "paperId": paper_id,
                    "title": "Cached",
                    "citationStyles": {"bibtex": bibtex},
                },
            )
        )

        # Populate cache under key "paper:<id>" via get_paper_details.
        await get_paper_details(
            PaperDetailsInput(paper_id=paper_id, response_format=ResponseFormat.JSON)
        )
        assert route.call_count == 1

        # export_citation reads the same cache key; no extra fetch.
        result = await export_citation(CitationExportInput(paper_id=paper_id))
        assert route.call_count == 1
        assert result == bibtex

    @respx.mock
    @pytest.mark.asyncio
    async def test_non_dict_response_raises(self, reset_all):
        """A non-dict API response raises ToolError (line 621)."""
        paper_id = "c" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        respx.get(url).mock(return_value=Response(200, json=["unexpected", "list"]))

        with pytest.raises(ToolError, match="Unexpected response format"):
            await export_citation(CitationExportInput(paper_id=paper_id))


# ===========================================================================
# match_paper / get_paper_authors — JSON-format arms
# ===========================================================================


class TestMatchPaperJson:
    @respx.mock
    @pytest.mark.asyncio
    async def test_match_paper_json(self, reset_all):
        """match_paper JSON output includes matchScore and the paper (line 666)."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/match"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={"data": [{"paperId": "a" * 40, "title": "Match", "matchScore": 91.5}]},
            )
        )
        params = PaperMatchInput(query="Match", response_format=ResponseFormat.JSON)
        result = await match_paper(params)
        parsed = json.loads(result)
        assert parsed["matchScore"] == 91.5
        assert parsed["paper"]["title"] == "Match"


class TestPaperAuthorsJson:
    @respx.mock
    @pytest.mark.asyncio
    async def test_paper_authors_json(self, reset_all):
        """get_paper_authors JSON output includes paper_id and authors (line 698)."""
        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}/authors"
        respx.get(url).mock(
            return_value=Response(200, json={"data": [{"authorId": "1", "name": "Ada"}]})
        )
        params = PaperAuthorsInput(paper_id=paper_id, response_format=ResponseFormat.JSON)
        result = await get_paper_authors(params)
        parsed = json.loads(result)
        assert parsed["paper_id"] == paper_id
        assert parsed["authors"][0]["name"] == "Ada"


# ===========================================================================
# get_author_batch — error arm + markdown rendering
# ===========================================================================


class TestAuthorBatchErrorAndMarkdown:
    @respx.mock
    @pytest.mark.asyncio
    async def test_api_error_becomes_tool_error(self, reset_all):
        """An API failure during author batch becomes ToolError (lines 723-724)."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/batch"
        respx.post(url).mock(return_value=Response(500))
        with pytest.raises(ToolError):
            await get_author_batch(AuthorBatchInput(author_ids=["1", "2"]))

    @respx.mock
    @pytest.mark.asyncio
    async def test_markdown_with_not_found(self, reset_all):
        """Markdown output lists retrieved authors and a not-found block
        (lines 740-750).
        """
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/batch"
        respx.post(url).mock(
            return_value=Response(200, json=[{"authorId": "1", "name": "Grace Hopper"}, None])
        )
        params = AuthorBatchInput(
            author_ids=["1", "999999999"], response_format=ResponseFormat.MARKDOWN
        )
        result = await get_author_batch(params)
        assert "Batch Author Retrieval" in result
        assert "Requested:** 2 | **Retrieved:** 1" in result
        assert "Not found (1)" in result
        assert "999999999" in result
        assert "Grace Hopper" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_markdown_all_succeed_omits_not_found(self, reset_all):
        """When every author resolves, markdown lists them and omits the
        not-found block (branch 748->751).
        """
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/batch"
        respx.post(url).mock(
            return_value=Response(
                200,
                json=[
                    {"authorId": "1", "name": "Ada Lovelace"},
                    {"authorId": "2", "name": "Alan Turing"},
                ],
            )
        )
        params = AuthorBatchInput(author_ids=["1", "2"], response_format=ResponseFormat.MARKDOWN)
        result = await get_author_batch(params)
        assert "Requested:** 2 | **Retrieved:** 2" in result
        assert "Not found" not in result
        assert "Ada Lovelace" in result
        assert "Alan Turing" in result


# ===========================================================================
# multi_recommend — error arm + JSON-format arm
# ===========================================================================


class TestMultiRecommendErrorAndJson:
    @respx.mock
    @pytest.mark.asyncio
    async def test_api_error_becomes_tool_error(self, reset_all):
        """An API failure during multi-recommend becomes ToolError (lines 783-784)."""
        url = f"{RECOMMENDATIONS_BASE}/papers/"
        respx.post(url).mock(return_value=Response(500))
        params = MultiRecommendInput(positive_paper_ids=["a" * 40])
        with pytest.raises(ToolError):
            await multi_recommend(params)

    @respx.mock
    @pytest.mark.asyncio
    async def test_json_output(self, reset_all):
        """JSON output includes positive/negative seeds and recommendations
        (line 787).
        """
        url = f"{RECOMMENDATIONS_BASE}/papers/"
        respx.post(url).mock(
            return_value=Response(
                200, json={"recommendedPapers": [{"paperId": "b" * 40, "title": "Rec"}]}
            )
        )
        params = MultiRecommendInput(
            positive_paper_ids=["a" * 40],
            negative_paper_ids=["c" * 40],
            response_format=ResponseFormat.JSON,
        )
        result = await multi_recommend(params)
        parsed = json.loads(result)
        assert parsed["positive"] == ["a" * 40]
        assert parsed["negative"] == ["c" * 40]
        assert parsed["recommendations"][0]["title"] == "Rec"


# ===========================================================================
# snippet_search — all optional params, JSON arm, sectionless markdown arm
# ===========================================================================


class TestSnippetSearchOptionalParamsAndFormats:
    @respx.mock
    @pytest.mark.asyncio
    async def test_all_optional_filters_forwarded(self, reset_all):
        """paper_ids/year/fields_of_study/min_citation_count are forwarded
        (lines 825, 827, 829, 831).
        """
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/snippet/search"
        route = respx.get(url).mock(return_value=Response(200, json={"data": []}))

        params = SnippetSearchInput(
            query="attention",
            paper_ids=["a" * 40, "b" * 40],
            year="2017",
            fields_of_study=["Computer Science"],
            min_citation_count=10,
        )
        await snippet_search(params)

        sent = route.calls[0].request.url.params
        assert sent.get("paperIds") == f"{'a' * 40},{'b' * 40}"
        assert sent.get("year") == "2017"
        assert sent.get("fieldsOfStudy") == "Computer Science"
        assert sent.get("minCitationCount") == "10"

    @respx.mock
    @pytest.mark.asyncio
    async def test_json_output(self, reset_all):
        """JSON output includes query and results (line 842)."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/snippet/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={"data": [{"paper": {"title": "P"}, "snippet": {"text": "t"}}]},
            )
        )
        params = SnippetSearchInput(query="q", response_format=ResponseFormat.JSON)
        result = await snippet_search(params)
        parsed = json.loads(result)
        assert parsed["query"] == "q"
        assert len(parsed["results"]) == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_markdown_snippet_without_section(self, reset_all):
        """A snippet lacking a section renders text but no Section line
        (branch 852->854).
        """
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/snippet/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "data": [
                        {
                            "paper": {"title": "No Section Paper"},
                            "snippet": {"text": "Body text only.", "section": ""},
                        }
                    ]
                },
            )
        )
        params = SnippetSearchInput(query="q", response_format=ResponseFormat.MARKDOWN)
        result = await snippet_search(params)
        assert "No Section Paper" in result
        assert "Body text only." in result
        assert "**Section:**" not in result


# ===========================================================================
# server.main() — keyed and keyless startup paths
# ===========================================================================


class TestMainEntryPoint:
    def test_main_without_key_warns_and_runs(self, monkeypatch):
        """Keyless start logs the rate-limit warning then calls mcp.run()
        (lines 918-924).
        """
        import semantic_scholar_mcp.server as server_mod

        monkeypatch.setattr(server_mod, "SEMANTIC_SCHOLAR_API_KEY", "")
        warnings: list[str] = []
        monkeypatch.setattr(server_mod.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
        ran: list[bool] = []
        monkeypatch.setattr(server_mod.mcp, "run", lambda: ran.append(True))

        server_mod.main([])

        assert ran == [True]
        assert any("API_KEY not set" in w for w in warnings)

    def test_main_with_key_skips_warning(self, monkeypatch):
        """With a key set, main() runs without emitting the keyless warning
        (the False arm of the line-918 guard).
        """
        import semantic_scholar_mcp.server as server_mod

        monkeypatch.setattr(server_mod, "SEMANTIC_SCHOLAR_API_KEY", "present")
        warnings: list[str] = []
        monkeypatch.setattr(server_mod.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
        ran: list[bool] = []
        monkeypatch.setattr(server_mod.mcp, "run", lambda: ran.append(True))

        server_mod.main([])

        assert ran == [True]
        assert warnings == []


# ===========================================================================
# client._parse_retry_after — HTTP-date branches
# ===========================================================================


class TestParseRetryAfterDates:
    def test_future_http_date_returns_positive_delay(self):
        """A future RFC-1123 date yields a positive delay (covers the
        timezone-aware path through line 157).
        """
        # parsedate_to_datetime returns a tz-aware datetime for this format.
        delay = _parse_retry_after("Wed, 21 Oct 2099 07:28:00 GMT", default=1.0)
        assert delay > 0

    def test_naive_http_date_is_treated_as_utc(self):
        """A date with the RFC 5322 '-0000' (unknown) zone parses to a *naive*
        datetime, exercising the tzinfo-is-None branch (line 156).

        '-0000' specifically means "no timezone information"; email.utils
        returns a naive datetime for it on every supported Python (3.10-3.13),
        whereas '+0000'/'GMT' yield an aware UTC datetime. The branch under
        test only runs for the naive case, so we assert that precondition
        explicitly: if a future runtime changed it, this test would fail loudly
        rather than silently drop the line from coverage.
        """
        from email.utils import parsedate_to_datetime

        parsed = parsedate_to_datetime("Wed, 21 Oct 2099 07:28:00 -0000")
        assert parsed.tzinfo is None, "precondition: '-0000' must parse as naive"

        delay = _parse_retry_after("Wed, 21 Oct 2099 07:28:00 -0000", default=1.0)
        assert delay > 0

    def test_unparseable_date_returns_default(self):
        """A value that is neither a float nor a parseable date falls back to
        the default (covers the parse-failure path including line 151).
        """
        assert _parse_retry_after("not-a-date-at-all", default=2.5) == 2.5

    def test_past_date_clamps_to_zero(self):
        """A past date yields a non-negative (clamped) delay."""
        assert _parse_retry_after("Mon, 01 Jan 1990 00:00:00 GMT", default=1.0) == 0.0


# ===========================================================================
# client.get_headers — keyless arm (no x-api-key)
# ===========================================================================


class TestGetHeadersKeyless:
    def test_keyless_omits_api_key_header(self, monkeypatch):
        """With no key anywhere, no x-api-key header is added (branch 94->96)."""
        monkeypatch.setattr(client_mod, "SEMANTIC_SCHOLAR_API_KEY", "")
        headers = get_headers()
        assert "x-api-key" not in headers
        assert headers["Accept"] == "application/json"


# ===========================================================================
# validators.is_valid_paper_id — empty input
# ===========================================================================


class TestIsValidPaperIdEmpty:
    def test_empty_string_is_invalid(self):
        """Empty input returns False without touching the patterns (line 55)."""
        assert is_valid_paper_id("") is False

    def test_whitespace_only_is_invalid(self):
        """Whitespace-only input returns False (line 55)."""
        assert is_valid_paper_id("   ") is False


# ===========================================================================
# logging_config.StructuredFormatter — exception-info branch
# ===========================================================================


class TestStructuredFormatterException:
    def test_exc_info_is_serialized(self):
        """When a record carries exc_info, the formatted JSON includes an
        'exc' field with the traceback (line 25).
        """
        formatter = StructuredFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failure occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        parsed = json.loads(formatter.format(record))
        assert parsed["msg"] == "failure occurred"
        assert "exc" in parsed
        assert "ValueError: boom" in parsed["exc"]

    def test_without_exc_info_has_no_exc_field(self):
        """A record with no exception info omits the 'exc' field (False arm of
        the line-24 guard).
        """
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="plain message",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["msg"] == "plain message"
        assert "exc" not in parsed
