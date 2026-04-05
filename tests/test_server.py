"""
Tests for semantic-scholar-mcp server.

Coverage targets:
- _handle_error: All 7 status codes map to correct exception types
- _format_paper_markdown: Missing fields, empty authors, None values
- Retry paths: 429->retry->200, timeout->retry->200, 404->immediate raise
- Paper ID validation: Valid/invalid formats
"""

from __future__ import annotations

import httpx
import pytest
import respx
from httpx import Response
from mcp.server.fastmcp.exceptions import ToolError

from semantic_scholar_mcp.server import (
    SEMANTIC_SCHOLAR_API_BASE,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    SemanticScholarError,
    ServerError,
    ValidationError,
    __version__,
    _execute_request_with_retry,
    _format_paper_markdown,
    _get_client,
    _handle_error,
    _validate_paper_id,
)

# ===============================================================================
# VERSION TESTS
# ===============================================================================


class TestVersion:
    """Test version constant."""

    def test_version_exists(self):
        """__version__ should exist and be a string."""
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_version_format(self):
        """__version__ should follow semver format."""
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


# ===============================================================================
# ERROR HANDLING TESTS
# ===============================================================================


class TestHandleError:
    """Test _handle_error maps status codes to correct exception types."""

    def test_400_raises_validation_error(self):
        """400 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _handle_error(400)
        assert exc_info.value.status_code == 400
        assert "Bad request" in str(exc_info.value)

    def test_401_raises_authentication_error_no_key(self):
        """401 without api_key should raise AuthenticationError with env var hint."""
        with pytest.raises(AuthenticationError) as exc_info:
            _handle_error(401, api_key=None)
        assert exc_info.value.status_code == 401
        assert "SEMANTIC_SCHOLAR_API_KEY" in str(exc_info.value)

    def test_401_raises_authentication_error_with_key(self):
        """401 with api_key should raise AuthenticationError about provided key."""
        with pytest.raises(AuthenticationError) as exc_info:
            _handle_error(401, api_key="test-key")
        assert exc_info.value.status_code == 401
        assert "provided API key" in str(exc_info.value)

    def test_403_raises_authentication_error_no_key(self):
        """403 without api_key should raise AuthenticationError."""
        with pytest.raises(AuthenticationError) as exc_info:
            _handle_error(403, api_key=None)
        assert exc_info.value.status_code == 403
        assert "Forbidden" in str(exc_info.value)

    def test_403_raises_authentication_error_with_key(self):
        """403 with api_key should raise AuthenticationError about invalid key."""
        with pytest.raises(AuthenticationError) as exc_info:
            _handle_error(403, api_key="test-key")
        assert exc_info.value.status_code == 403
        assert "invalid or expired" in str(exc_info.value)

    def test_404_raises_not_found_error(self):
        """404 should raise NotFoundError."""
        with pytest.raises(NotFoundError) as exc_info:
            _handle_error(404)
        assert exc_info.value.status_code == 404
        assert "Not found" in str(exc_info.value)

    def test_429_raises_rate_limit_error(self):
        """429 should raise RateLimitError with retry_after."""
        with pytest.raises(RateLimitError) as exc_info:
            _handle_error(429, retry_after=5.0)
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 5.0
        assert "Rate limited" in str(exc_info.value)

    def test_500_raises_server_error(self):
        """500 should raise ServerError."""
        with pytest.raises(ServerError) as exc_info:
            _handle_error(500)
        assert exc_info.value.status_code == 500
        assert "Server error" in str(exc_info.value)

    def test_502_raises_server_error(self):
        """502 should raise ServerError."""
        with pytest.raises(ServerError) as exc_info:
            _handle_error(502)
        assert exc_info.value.status_code == 502

    def test_503_raises_server_error_with_unavailable_message(self):
        """503 should raise ServerError with 'unavailable' message."""
        with pytest.raises(ServerError) as exc_info:
            _handle_error(503)
        assert exc_info.value.status_code == 503
        assert "unavailable" in str(exc_info.value)

    def test_unknown_status_raises_semantic_scholar_error(self):
        """Unknown status codes should raise base SemanticScholarError."""
        with pytest.raises(SemanticScholarError) as exc_info:
            _handle_error(418)  # I'm a teapot
        assert exc_info.value.status_code == 418
        assert "418" in str(exc_info.value)


# ===============================================================================
# FORMAT PAPER MARKDOWN TESTS
# ===============================================================================


class TestFormatPaperMarkdown:
    """Test _format_paper_markdown handles edge cases."""

    def test_complete_paper(self):
        """Full paper with all fields should format correctly."""
        paper = {
            "title": "Attention Is All You Need",
            "year": 2017,
            "authors": [
                {"name": "Author One"},
                {"name": "Author Two"},
            ],
            "venue": "NeurIPS",
            "citationCount": 50000,
            "influentialCitationCount": 5000,
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            "fieldsOfStudy": ["Computer Science", "Machine Learning"],
            "tldr": {"text": "Transformers are great."},
            "abstract": "We propose a new architecture...",
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "1706.03762"},
            "paperId": "abc123",
            "url": "https://semanticscholar.org/paper/abc123",
        }
        result = _format_paper_markdown(paper)

        assert "### Attention Is All You Need (2017)" in result
        assert "Author One" in result
        assert "Author Two" in result
        assert "NeurIPS" in result
        assert "50000" in result
        assert "5000" in result
        assert "[PDF]" in result
        assert "Computer Science" in result
        assert "Transformers are great." in result
        assert "DOI: 10.1234/test" in result
        assert "ArXiv: 1706.03762" in result

    def test_missing_title(self):
        """Paper without title should show 'Unknown Title'."""
        paper = {"year": 2020}
        result = _format_paper_markdown(paper)
        assert "Unknown Title" in result

    def test_missing_year(self):
        """Paper without year should show 'N/A'."""
        paper = {"title": "Test Paper"}
        result = _format_paper_markdown(paper)
        assert "N/A" in result

    def test_empty_authors(self):
        """Paper with empty authors list should not show authors line."""
        paper = {"title": "Test", "year": 2020, "authors": []}
        result = _format_paper_markdown(paper)
        assert "Authors" not in result

    def test_none_authors(self):
        """Paper with None authors should not show authors line."""
        paper = {"title": "Test", "year": 2020, "authors": None}
        result = _format_paper_markdown(paper)
        assert "Authors" not in result

    def test_many_authors_truncated(self):
        """More than 5 authors should be truncated with '+N more'."""
        paper = {
            "title": "Test",
            "year": 2020,
            "authors": [{"name": f"Author {i}"} for i in range(10)],
        }
        result = _format_paper_markdown(paper)
        assert "+5 more" in result
        assert "Author 0" in result
        assert "Author 4" in result

    def test_author_missing_name(self):
        """Author without name should show '?'."""
        paper = {
            "title": "Test",
            "year": 2020,
            "authors": [{"id": "123"}],  # No name field
        }
        result = _format_paper_markdown(paper)
        assert "?" in result

    def test_none_venue(self):
        """Paper with None venue should not show venue line."""
        paper = {"title": "Test", "year": 2020, "venue": None}
        result = _format_paper_markdown(paper)
        assert "Venue" not in result

    def test_publication_venue_fallback(self):
        """Should use publicationVenue.name if venue is empty."""
        paper = {
            "title": "Test",
            "year": 2020,
            "venue": None,
            "publicationVenue": {"name": "ArXiv"},
        }
        result = _format_paper_markdown(paper)
        assert "ArXiv" in result

    def test_none_open_access_pdf(self):
        """Paper with None openAccessPdf should not show PDF link."""
        paper = {"title": "Test", "year": 2020, "openAccessPdf": None}
        result = _format_paper_markdown(paper)
        assert "Open Access" not in result

    def test_none_fields_of_study(self):
        """Paper with None fieldsOfStudy should not show fields line."""
        paper = {"title": "Test", "year": 2020, "fieldsOfStudy": None}
        result = _format_paper_markdown(paper)
        assert "Fields" not in result

    def test_none_tldr(self):
        """Paper with None tldr should not show TL;DR line."""
        paper = {"title": "Test", "year": 2020, "tldr": None}
        result = _format_paper_markdown(paper)
        assert "TL;DR" not in result

    def test_none_abstract(self):
        """Paper with None abstract should not show abstract."""
        paper = {"title": "Test", "year": 2020, "abstract": None}
        result = _format_paper_markdown(paper)
        assert "Abstract" not in result

    def test_long_abstract_truncated(self):
        """Abstract over 500 chars should be truncated."""
        paper = {
            "title": "Test",
            "year": 2020,
            "abstract": "A" * 600,
        }
        result = _format_paper_markdown(paper)
        assert "..." in result
        # Should have exactly 500 A's plus "..."
        assert "A" * 500 + "..." in result

    def test_none_external_ids(self):
        """Paper with None externalIds should not show IDs line."""
        paper = {"title": "Test", "year": 2020, "externalIds": None}
        result = _format_paper_markdown(paper)
        assert "IDs:" not in result

    def test_default_citation_counts(self):
        """Paper without citation counts should show 0."""
        paper = {"title": "Test", "year": 2020}
        result = _format_paper_markdown(paper)
        assert "0 (0 influential)" in result


# ===============================================================================
# PAPER ID VALIDATION TESTS
# ===============================================================================


class TestValidatePaperId:
    """Test _validate_paper_id regex patterns."""

    def test_valid_40_char_hex_lowercase(self):
        """40-char lowercase hex should be valid."""
        _validate_paper_id("a" * 40)  # Should not raise

    def test_valid_40_char_hex_uppercase(self):
        """40-char uppercase hex should be valid."""
        _validate_paper_id("A" * 40)  # Should not raise

    def test_valid_40_char_hex_mixed(self):
        """40-char mixed case hex should be valid."""
        _validate_paper_id("649def34f8be52c8b66281af98ae884c09aef38b")

    def test_valid_doi(self):
        """DOI:xxx format should be valid."""
        _validate_paper_id("DOI:10.1038/s41586-021-03819-2")

    def test_valid_doi_lowercase(self):
        """doi:xxx format should be valid (case insensitive)."""
        _validate_paper_id("doi:10.1234/test")

    def test_valid_arxiv(self):
        """ARXIV:xxx format should be valid."""
        _validate_paper_id("ARXIV:2106.15928")

    def test_valid_arxiv_with_version(self):
        """ARXIV:xxx with version should be valid."""
        _validate_paper_id("ARXIV:2106.15928v2")

    def test_valid_pmid(self):
        """PMID:xxx format should be valid."""
        _validate_paper_id("PMID:32908142")

    def test_valid_corpusid(self):
        """CorpusId:xxx format should be valid."""
        _validate_paper_id("CorpusId:215416146")

    def test_valid_url(self):
        """URL:xxx format should be valid."""
        _validate_paper_id("URL:https://arxiv.org/abs/2106.15928")

    def test_valid_acl(self):
        """ACL:xxx format should be valid."""
        _validate_paper_id("ACL:P19-1285")

    def test_invalid_empty_string(self):
        """Empty string should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_paper_id("")
        assert "empty" in str(exc_info.value)

    def test_invalid_whitespace_only(self):
        """Whitespace-only string should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_paper_id("   ")
        assert "empty" in str(exc_info.value)

    def test_invalid_short_hex(self):
        """39-char hex should be invalid."""
        with pytest.raises(ValidationError):
            _validate_paper_id("a" * 39)

    def test_invalid_long_hex(self):
        """41-char hex should be invalid."""
        with pytest.raises(ValidationError):
            _validate_paper_id("a" * 41)

    def test_invalid_non_hex_40_char(self):
        """40-char non-hex should be invalid."""
        with pytest.raises(ValidationError):
            _validate_paper_id("g" * 40)  # 'g' is not hex

    def test_invalid_random_string(self):
        """Random string should be invalid."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_paper_id("some-random-paper-id")
        assert "Invalid paper ID format" in str(exc_info.value)
        assert "Accepted formats" in str(exc_info.value)

    def test_invalid_doi_without_prefix(self):
        """DOI without prefix should be invalid."""
        with pytest.raises(ValidationError):
            _validate_paper_id("10.1038/s41586-021-03819-2")

    def test_invalid_arxiv_without_prefix(self):
        """ArXiv ID without prefix should be invalid."""
        with pytest.raises(ValidationError):
            _validate_paper_id("2106.15928")

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        _validate_paper_id("  DOI:10.1234/test  ")  # Should not raise


# ===============================================================================
# RETRY LOGIC TESTS
# ===============================================================================


class TestRetryLogic:
    """Test retry behavior for transient errors."""

    @pytest.fixture
    def reset_client(self):
        """Reset the global HTTP client before each test."""
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_then_success(self, reset_client):
        """429 should retry and eventually succeed."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        # First call returns 429, second returns 200
        route = respx.get(url).mock(
            side_effect=[
                Response(429, headers={"Retry-After": "0.1"}),
                Response(200, json={"data": [{"paperId": "123"}]}),
            ]
        )

        await _get_client()
        result = await _execute_request_with_retry("GET", url, {"query": "test"}, None, {}, None)

        assert result == {"data": [{"paperId": "123"}]}
        assert route.call_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_then_success(self, reset_client):
        """Timeout should retry and eventually succeed."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        # First call times out, second returns 200
        route = respx.get(url).mock(
            side_effect=[
                httpx.TimeoutException("Connection timed out"),
                Response(200, json={"data": []}),
            ]
        )

        await _get_client()
        result = await _execute_request_with_retry("GET", url, None, None, {}, None)

        assert result == {"data": []}
        assert route.call_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_404_no_retry(self, reset_client):
        """404 should raise immediately without retry."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/invalid-id"

        route = respx.get(url).mock(return_value=Response(404))

        await _get_client()
        with pytest.raises(NotFoundError):
            await _execute_request_with_retry("GET", url, None, None, {}, None)

        # Should only be called once - no retry for 404
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_503_retries_then_raises(self, reset_client):
        """503 should retry max times then raise."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        # All calls return 503
        route = respx.get(url).mock(return_value=Response(503))

        await _get_client()
        with pytest.raises(ServerError):
            await _execute_request_with_retry("GET", url, None, None, {}, None)

        # Should retry MAX_RETRIES times (3) + 1 initial = 4 calls
        assert route.call_count == 4

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_retries_then_raises(self, reset_client):
        """Timeout should retry max times then raise."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        route = respx.get(url).mock(side_effect=httpx.TimeoutException("Connection timed out"))

        await _get_client()
        with pytest.raises(SemanticScholarError) as exc_info:
            await _execute_request_with_retry("GET", url, None, None, {}, None)

        assert "timed out" in str(exc_info.value)
        assert route.call_count == 4  # MAX_RETRIES + 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_400_no_retry(self, reset_client):
        """400 should raise immediately without retry."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        route = respx.get(url).mock(return_value=Response(400))

        await _get_client()
        with pytest.raises(ValidationError):
            await _execute_request_with_retry("GET", url, None, None, {}, None)

        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_401_no_retry(self, reset_client):
        """401 should raise immediately without retry."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        route = respx.get(url).mock(return_value=Response(401))

        await _get_client()
        with pytest.raises(AuthenticationError):
            await _execute_request_with_retry("GET", url, None, None, {}, None)

        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_uses_retry_after_header(self, reset_client):
        """429 should use Retry-After header value."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        # First returns 429 with short retry, second succeeds
        route = respx.get(url).mock(
            side_effect=[
                Response(429, headers={"Retry-After": "0.01"}),
                Response(200, json={"total": 0, "data": []}),
            ]
        )

        await _get_client()
        result = await _execute_request_with_retry("GET", url, None, None, {}, None)

        assert result == {"total": 0, "data": []}
        assert route.call_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_post_request_retry(self, reset_client):
        """POST requests should also retry on 429."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"

        route = respx.post(url).mock(
            side_effect=[
                Response(429, headers={"Retry-After": "0.01"}),
                Response(200, json=[{"paperId": "123"}]),
            ]
        )

        await _get_client()
        result = await _execute_request_with_retry("POST", url, None, {"ids": ["123"]}, {}, None)

        assert result == [{"paperId": "123"}]
        assert route.call_count == 2


# ===============================================================================
# CLIENT LIFECYCLE TESTS
# ===============================================================================


class TestClientLifecycle:
    """Test HTTP client lifecycle management."""

    @pytest.fixture
    def reset_client(self):
        """Reset the global HTTP client before each test."""
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, reset_client):
        """_get_client should create client if none exists."""
        import semantic_scholar_mcp.server as server

        assert server._client is None

        client = await _get_client()

        assert client is not None
        assert not client.is_closed
        assert server._client is client

    @pytest.mark.asyncio
    async def test_get_client_returns_existing(self, reset_client):
        """_get_client should return existing client."""
        client1 = await _get_client()
        client2 = await _get_client()

        assert client1 is client2

    @pytest.mark.asyncio
    async def test_get_client_recreates_if_closed(self, reset_client):
        """_get_client should recreate client if closed."""
        client1 = await _get_client()
        await client1.aclose()

        client2 = await _get_client()

        assert client2 is not client1
        assert not client2.is_closed


# ===============================================================================
# TOOL FUNCTION TESTS
# ===============================================================================


class TestSearchPapersTool:
    """Test search_papers tool function."""

    @pytest.fixture
    def reset_client(self):
        """Reset the global HTTP client before each test."""
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_papers_success_markdown(self, reset_client):
        """search_papers should return markdown formatted results."""
        from semantic_scholar_mcp.server import PaperSearchInput, search_papers

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 1,
                    "data": [
                        {
                            "paperId": "123",
                            "title": "Test Paper",
                            "year": 2024,
                            "citationCount": 10,
                            "influentialCitationCount": 2,
                        }
                    ],
                },
            )
        )

        params = PaperSearchInput(query="test query")
        result = await search_papers(params)

        assert "Test Paper" in result
        assert "2024" in result
        assert "Search Results" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_papers_success_json(self, reset_client):
        """search_papers should return JSON when requested."""
        from semantic_scholar_mcp.server import PaperSearchInput, ResponseFormat, search_papers

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(
            return_value=Response(
                200, json={"total": 1, "data": [{"paperId": "123", "title": "Test"}]}
            )
        )

        params = PaperSearchInput(query="test", response_format=ResponseFormat.JSON)
        result = await search_papers(params)

        import json

        parsed = json.loads(result)
        assert parsed["query"] == "test"
        assert parsed["total"] == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_papers_with_filters(self, reset_client):
        """search_papers should apply all filters."""
        from semantic_scholar_mcp.server import PaperSearchInput, search_papers

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        route = respx.get(url).mock(return_value=Response(200, json={"total": 0, "data": []}))

        params = PaperSearchInput(
            query="machine learning",
            year="2023",
            fields_of_study=["Computer Science"],
            publication_types=["JournalArticle"],
            open_access_only=True,
            min_citation_count=100,
            limit=20,
            offset=10,
        )
        await search_papers(params)

        # Verify the filters were passed
        call = route.calls.last
        assert "year" in str(call.request.url)
        assert "fieldsOfStudy" in str(call.request.url)

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_papers_error_handling(self, reset_client):
        """search_papers should return error message on failure."""
        from semantic_scholar_mcp.server import PaperSearchInput, search_papers

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(500))

        params = PaperSearchInput(query="test")
        with pytest.raises(ToolError):
            await search_papers(params)


class TestGetPaperDetailsTool:
    """Test get_paper_details tool function."""

    @pytest.fixture
    def reset_client(self):
        """Reset the global HTTP client before each test."""
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_details_success(self, reset_client):
        """get_paper_details should return paper info."""
        from semantic_scholar_mcp.server import PaperDetailsInput, get_paper_details

        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "paperId": paper_id,
                    "title": "Test Paper",
                    "year": 2024,
                    "citationCount": 10,
                    "influentialCitationCount": 2,
                },
            )
        )

        params = PaperDetailsInput(paper_id=paper_id)
        result = await get_paper_details(params)

        assert "Test Paper" in result
        assert "Paper Details" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_details_with_citations(self, reset_client):
        """get_paper_details should include citations when requested."""
        from semantic_scholar_mcp.server import PaperDetailsInput, get_paper_details

        paper_id = "a" * 40
        base_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        cit_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}/citations"

        respx.get(base_url).mock(
            return_value=Response(
                200,
                json={
                    "paperId": paper_id,
                    "title": "Main Paper",
                    "year": 2024,
                    "citationCount": 100,
                    "influentialCitationCount": 10,
                },
            )
        )
        respx.get(cit_url).mock(
            return_value=Response(
                200,
                json={
                    "data": [
                        {"citingPaper": {"title": "Citing Paper", "year": 2024, "citationCount": 5}}
                    ]
                },
            )
        )

        params = PaperDetailsInput(paper_id=paper_id, include_citations=True)
        result = await get_paper_details(params)

        assert "Main Paper" in result
        assert "Citing Paper" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_details_with_references(self, reset_client):
        """get_paper_details should include references when requested."""
        from semantic_scholar_mcp.server import PaperDetailsInput, get_paper_details

        paper_id = "a" * 40
        base_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        ref_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}/references"

        respx.get(base_url).mock(
            return_value=Response(
                200,
                json={
                    "paperId": paper_id,
                    "title": "Main Paper",
                    "year": 2024,
                    "citationCount": 100,
                    "influentialCitationCount": 10,
                },
            )
        )
        respx.get(ref_url).mock(
            return_value=Response(
                200,
                json={
                    "data": [
                        {
                            "citedPaper": {
                                "title": "Referenced Paper",
                                "year": 2020,
                                "citationCount": 500,
                            }
                        }
                    ]
                },
            )
        )

        params = PaperDetailsInput(paper_id=paper_id, include_references=True)
        result = await get_paper_details(params)

        assert "Main Paper" in result
        assert "Referenced Paper" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_details_invalid_id(self, reset_client):
        """get_paper_details should validate paper ID."""
        from semantic_scholar_mcp.server import PaperDetailsInput, get_paper_details

        params = PaperDetailsInput(paper_id="invalid-id")
        with pytest.raises(ToolError):
            await get_paper_details(params)


class TestGetRecommendationsTool:
    """Test get_recommendations tool function."""

    @pytest.fixture
    def reset_client(self):
        """Reset the global HTTP client before each test."""
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_recommendations_success(self, reset_client):
        """get_recommendations should return related papers."""
        from semantic_scholar_mcp.server import (
            RECOMMENDATIONS_BASE,
            PaperRecommendationsInput,
            get_recommendations,
        )

        paper_id = "a" * 40
        url = f"{RECOMMENDATIONS_BASE}/papers/forpaper/{paper_id}"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "recommendedPapers": [
                        {
                            "paperId": "b" * 40,
                            "title": "Recommended Paper",
                            "year": 2024,
                            "citationCount": 50,
                            "influentialCitationCount": 5,
                        }
                    ]
                },
            )
        )

        params = PaperRecommendationsInput(paper_id=paper_id)
        result = await get_recommendations(params)

        assert "Recommended Paper" in result
        assert "Recommendations" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_recommendations_invalid_id(self, reset_client):
        """get_recommendations should validate paper ID."""
        from semantic_scholar_mcp.server import PaperRecommendationsInput, get_recommendations

        params = PaperRecommendationsInput(paper_id="invalid-id")
        with pytest.raises(ToolError):
            await get_recommendations(params)


class TestSearchAuthorsTool:
    """Test search_authors tool function."""

    @pytest.fixture
    def reset_client(self):
        """Reset the global HTTP client before each test."""
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_authors_success(self, reset_client):
        """search_authors should return author info."""
        from semantic_scholar_mcp.server import AuthorSearchInput, search_authors

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 1,
                    "data": [
                        {
                            "authorId": "123",
                            "name": "John Doe",
                            "hIndex": 50,
                            "paperCount": 100,
                            "citationCount": 5000,
                        }
                    ],
                },
            )
        )

        params = AuthorSearchInput(query="John Doe")
        result = await search_authors(params)

        assert "John Doe" in result
        assert "Author Search" in result


class TestGetAuthorDetailsTool:
    """Test get_author_details tool function."""

    @pytest.fixture
    def reset_client(self):
        """Reset the global HTTP client before each test."""
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_author_details_success(self, reset_client):
        """get_author_details should return author profile."""
        from semantic_scholar_mcp.server import AuthorDetailsInput, get_author_details

        author_id = "123"
        base_url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}"
        papers_url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}/papers"

        respx.get(base_url).mock(
            return_value=Response(
                200,
                json={
                    "authorId": author_id,
                    "name": "Jane Smith",
                    "hIndex": 45,
                    "paperCount": 80,
                    "citationCount": 4000,
                },
            )
        )
        respx.get(papers_url).mock(
            return_value=Response(
                200, json={"data": [{"title": "Author Paper", "year": 2024, "citationCount": 100}]}
            )
        )

        params = AuthorDetailsInput(author_id=author_id)
        result = await get_author_details(params)

        assert "Jane Smith" in result
        assert "Author Profile" in result


class TestBulkPapersTool:
    """Test get_bulk_papers tool function."""

    @pytest.fixture
    def reset_client(self):
        """Reset the global HTTP client before each test."""
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_papers_success(self, reset_client):
        """get_bulk_papers should return multiple papers."""
        from semantic_scholar_mcp.server import BulkPaperInput, get_bulk_papers

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

        params = BulkPaperInput(paper_ids=["a" * 40, "b" * 40])
        result = await get_bulk_papers(params)

        import json

        parsed = json.loads(result)
        assert parsed["requested"] == 2
        assert parsed["retrieved"] == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_papers_with_failures(self, reset_client):
        """get_bulk_papers should report papers not found."""
        from semantic_scholar_mcp.server import BulkPaperInput, get_bulk_papers

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"
        respx.post(url).mock(
            return_value=Response(
                200,
                json=[
                    {"paperId": "a" * 40, "title": "Paper 1", "citationCount": 10},
                    None,  # Paper not found
                ],
            )
        )

        params = BulkPaperInput(paper_ids=["a" * 40, "b" * 40])
        result = await get_bulk_papers(params)

        import json

        parsed = json.loads(result)
        assert parsed["requested"] == 2
        assert parsed["retrieved"] == 1
        assert "not_found" in parsed

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_papers_invalid_ids(self, reset_client):
        """get_bulk_papers should validate all paper IDs."""
        from semantic_scholar_mcp.server import BulkPaperInput, get_bulk_papers

        params = BulkPaperInput(paper_ids=["invalid-id-1", "invalid-id-2"])
        with pytest.raises(ToolError, match="Invalid paper ID"):
            await get_bulk_papers(params)


class TestServerStatusTool:
    """Test server_status tool function."""

    @pytest.fixture
    def reset_client(self):
        """Reset the global HTTP client before each test."""
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_status_api_reachable(self, reset_client):
        """server_status should report API as reachable."""
        from semantic_scholar_mcp.server import server_status

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(200, json={"data": []}))

        result = await server_status()

        import json

        parsed = json.loads(result)
        assert parsed["server"] == "semantic-scholar-mcp"
        assert parsed["version"] == __version__
        assert parsed["api_reachable"] is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_status_api_unreachable(self, reset_client):
        """server_status should report API as unreachable on error."""
        from semantic_scholar_mcp.server import server_status

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(500))

        result = await server_status()

        import json

        parsed = json.loads(result)
        assert parsed["api_reachable"] is False
        assert "error" in parsed


# ===============================================================================
# MAKE REQUEST TESTS
# ===============================================================================


class TestMakeRequest:
    """Test _make_request function."""

    @pytest.fixture
    def reset_client(self):
        """Reset the global HTTP client and rate limit state."""
        import semantic_scholar_mcp.server as server

        old_client = server._client
        old_time = server._last_request_time
        server._client = None
        server._last_request_time = 0.0
        yield
        server._client = old_client
        server._last_request_time = old_time

    @respx.mock
    @pytest.mark.asyncio
    async def test_make_request_get(self, reset_client):
        """_make_request should handle GET requests."""
        from semantic_scholar_mcp.server import _make_request

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(200, json={"data": []}))

        result = await _make_request("GET", "paper/search", params={"query": "test"})
        assert result == {"data": []}

    @respx.mock
    @pytest.mark.asyncio
    async def test_make_request_post(self, reset_client):
        """_make_request should handle POST requests."""
        from semantic_scholar_mcp.server import _make_request

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"
        respx.post(url).mock(return_value=Response(200, json=[{"paperId": "123"}]))

        result = await _make_request("POST", "paper/batch", json_body={"ids": ["123"]})
        assert result == [{"paperId": "123"}]

    @respx.mock
    @pytest.mark.asyncio
    async def test_make_request_custom_base_url(self, reset_client):
        """_make_request should use custom base URL when provided."""
        from semantic_scholar_mcp.server import RECOMMENDATIONS_BASE, _make_request

        url = f"{RECOMMENDATIONS_BASE}/papers/forpaper/123"
        respx.get(url).mock(return_value=Response(200, json={"recommendedPapers": []}))

        result = await _make_request("GET", "papers/forpaper/123", base_url=RECOMMENDATIONS_BASE)
        assert result == {"recommendedPapers": []}


# ===============================================================================
# HEADER TESTS
# ===============================================================================


class TestGetHeaders:
    """Test _get_headers function."""

    def test_headers_without_api_key(self):
        """_get_headers should return basic headers without API key."""
        from semantic_scholar_mcp.server import _get_headers

        headers = _get_headers(api_key=None)
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"
        # Authorization should only be present if env var is set
        # Don't check for Authorization absence as env might have it

    def test_headers_with_api_key(self):
        """_get_headers should include Bearer auth when API key provided."""
        from semantic_scholar_mcp.server import _get_headers

        headers = _get_headers(api_key="test-api-key")
        assert headers["Authorization"] == "Bearer test-api-key"


# ===============================================================================
# AUTHOR MARKDOWN TESTS
# ===============================================================================


class TestFormatAuthorMarkdown:
    """Test _format_author_markdown function."""

    def test_complete_author(self):
        """Full author info should format correctly."""
        from semantic_scholar_mcp.server import _format_author_markdown

        author = {
            "name": "John Researcher",
            "authorId": "12345",
            "affiliations": ["MIT", "Harvard"],
            "hIndex": 50,
            "paperCount": 100,
            "citationCount": 5000,
            "homepage": "https://johnresearcher.com",
            "url": "https://semanticscholar.org/author/12345",
        }
        result = _format_author_markdown(author)

        assert "John Researcher" in result
        assert "MIT" in result
        assert "h-index" in result
        assert "50" in result
        assert "Homepage" in result

    def test_minimal_author(self):
        """Author with minimal info should still format."""
        from semantic_scholar_mcp.server import _format_author_markdown

        author = {"name": "Anonymous"}
        result = _format_author_markdown(author)

        assert "Anonymous" in result

    def test_author_missing_name(self):
        """Author without name should show 'Unknown'."""
        from semantic_scholar_mcp.server import _format_author_markdown

        author = {"authorId": "123"}
        result = _format_author_markdown(author)

        assert "Unknown" in result

    def test_author_with_empty_affiliations(self):
        """Author with empty affiliations should not show affiliations line."""
        from semantic_scholar_mcp.server import _format_author_markdown

        author = {"name": "Test", "affiliations": []}
        result = _format_author_markdown(author)

        assert "Affiliations" not in result


# ===============================================================================
# ENTRY POINT TESTS
# ===============================================================================


# ===============================================================================
# COVERAGE GAP TESTS (v1.1.0)
# ===============================================================================


class TestPaperDetailFieldSelection:
    """Verify get_paper_details uses PAPER_DETAIL_FIELDS for main fetch."""

    @pytest.fixture
    def reset_client(self):
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_main_fetch_uses_detail_fields(self, reset_client):
        """Main paper fetch must use PAPER_DETAIL_FIELDS (includes abstract, publicationVenue)."""
        from semantic_scholar_mcp.server import (
            PAPER_DETAIL_FIELDS,
            PAPER_SEARCH_FIELDS,
            PaperDetailsInput,
            get_paper_details,
        )

        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"

        route = respx.get(url).mock(
            return_value=Response(200, json={"paperId": paper_id, "title": "Test"})
        )

        params = PaperDetailsInput(paper_id=paper_id)
        await get_paper_details(params)

        # Inspect the fields param sent in the request
        request = route.calls[0].request
        fields_sent = str(request.url.params.get("fields", ""))
        fields_list = fields_sent.split(",")

        # PAPER_DETAIL_FIELDS has fields that PAPER_SEARCH_FIELDS does not
        detail_only = set(PAPER_DETAIL_FIELDS) - set(PAPER_SEARCH_FIELDS)
        assert detail_only, "PAPER_DETAIL_FIELDS should have extra fields beyond PAPER_SEARCH_FIELDS"

        for field in detail_only:
            assert field in fields_list, (
                f"Main paper fetch missing '{field}' — "
                f"using PAPER_SEARCH_FIELDS instead of PAPER_DETAIL_FIELDS?"
            )


class TestBulkPapersMarkdownFailures:
    """Verify bulk papers markdown output reports unfound papers."""

    @pytest.fixture
    def reset_client(self):
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_markdown_includes_not_found(self, reset_client):
        """Markdown output must include 'Not found' section for null entries."""
        from semantic_scholar_mcp.server import BulkPaperInput, ResponseFormat, get_bulk_papers

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"
        respx.post(url).mock(
            return_value=Response(
                200,
                json=[
                    {"paperId": "a" * 40, "title": "Found Paper", "citationCount": 10},
                    None,
                ],
            )
        )

        params = BulkPaperInput(
            paper_ids=["a" * 40, "b" * 40],
            response_format=ResponseFormat.MARKDOWN,
        )
        result = await get_bulk_papers(params)

        assert "Not found" in result
        assert "b" * 40 in result
        assert "Requested:** 2" in result
        assert "Retrieved:** 1" in result


class TestRetryExhaustion429:
    """Verify 429 retry exhaustion raises RateLimitError."""

    @pytest.fixture
    def reset_client(self):
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_retries_then_raises_rate_limit_error(self, reset_client):
        """All retries return 429 — must raise RateLimitError after MAX_RETRIES."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        route = respx.get(url).mock(
            return_value=Response(429, headers={"Retry-After": "0.01"})
        )

        await _get_client()
        with pytest.raises(RateLimitError) as exc_info:
            await _execute_request_with_retry("GET", url, None, None, {}, None)

        assert exc_info.value.status_code == 429
        assert route.call_count == 4  # 1 initial + 3 retries


class TestBackoffTiming:
    """Verify exponential backoff timing on retries."""

    @pytest.fixture
    def reset_client(self):
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_503_backoff_is_exponential(self, reset_client):
        """503 retries should use exponential backoff: base*2^0, base*2^1, base*2^2."""
        from unittest.mock import AsyncMock, patch

        from semantic_scholar_mcp.server import RETRY_BACKOFF_BASE

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        route = respx.get(url).mock(return_value=Response(503))

        sleep_calls: list[float] = []
        original_sleep = __import__("asyncio").sleep

        async def capture_sleep(duration: float) -> None:
            sleep_calls.append(duration)
            # Don't actually wait — speed up the test
            return None

        await _get_client()
        with patch("semantic_scholar_mcp.server.asyncio.sleep", side_effect=capture_sleep):
            with pytest.raises(ServerError):
                await _execute_request_with_retry("GET", url, None, None, {}, None)

        assert route.call_count == 4  # 1 initial + 3 retries
        assert len(sleep_calls) == 3  # 3 sleeps between retries

        # Verify exponential progression: base*2^0 + jitter, base*2^1 + jitter, base*2^2 + jitter
        # Jitter is uniform(0, 0.5), so each wait is in [base*2^n, base*2^n + 0.5]
        for i, wait in enumerate(sleep_calls):
            expected_base = RETRY_BACKOFF_BASE * (2**i)
            assert expected_base <= wait <= expected_base + 0.5, (
                f"Retry {i}: wait={wait:.3f}, expected [{expected_base}, {expected_base + 0.5}]"
            )


class TestBulkSearchTool:
    """Test bulk_search tool function."""

    @pytest.fixture
    def reset_client(self):
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_search_success_json(self, reset_client):
        """bulk_search should return papers with total and token."""
        from semantic_scholar_mcp.server import BulkSearchInput, ResponseFormat, bulk_search

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/bulk"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 5000,
                    "token": "abc123next",
                    "data": [
                        {"paperId": "a" * 40, "title": "Paper 1", "citationCount": 100},
                    ],
                },
            )
        )

        params = BulkSearchInput(query="transformers", response_format=ResponseFormat.JSON)
        result = await bulk_search(params)

        import json

        parsed = json.loads(result)
        assert parsed["total"] == 5000
        assert parsed["token"] == "abc123next"
        assert len(parsed["papers"]) == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_search_with_sort(self, reset_client):
        """bulk_search should pass sort param to API."""
        from semantic_scholar_mcp.server import BulkSearchInput, ResponseFormat, bulk_search

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/bulk"
        route = respx.get(url).mock(
            return_value=Response(200, json={"total": 0, "data": []})
        )

        params = BulkSearchInput(
            query="test", sort="citationCount:desc", response_format=ResponseFormat.JSON
        )
        await bulk_search(params)

        request = route.calls[0].request
        assert "citationCount:desc" in str(request.url.params.get("sort", ""))

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_search_markdown_with_token(self, reset_client):
        """Markdown output should show continuation token."""
        from semantic_scholar_mcp.server import BulkSearchInput, bulk_search

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/bulk"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 100,
                    "token": "page2token",
                    "data": [{"paperId": "a" * 40, "title": "Test", "citationCount": 0}],
                },
            )
        )

        params = BulkSearchInput(query="test")
        result = await bulk_search(params)

        assert "Bulk Search" in result
        assert "page2token" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_search_no_token_last_page(self, reset_client):
        """JSON output should omit token when not present (last page)."""
        from semantic_scholar_mcp.server import BulkSearchInput, ResponseFormat, bulk_search

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/bulk"
        respx.get(url).mock(
            return_value=Response(200, json={"total": 1, "data": [{"paperId": "a" * 40}]})
        )

        params = BulkSearchInput(query="test", response_format=ResponseFormat.JSON)
        result = await bulk_search(params)

        import json

        parsed = json.loads(result)
        assert "token" not in parsed

    @respx.mock
    @pytest.mark.asyncio
    async def test_bulk_search_error_handling(self, reset_client):
        """bulk_search should convert API errors to ToolError."""
        from semantic_scholar_mcp.server import BulkSearchInput, bulk_search

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/bulk"
        respx.get(url).mock(return_value=Response(500))

        params = BulkSearchInput(query="test")
        with pytest.raises(ToolError):
            await bulk_search(params)


class TestExportCitationTool:
    """Test export_citation tool function."""

    @pytest.fixture
    def reset_client(self):
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_export_bibtex_success(self, reset_client):
        """export_citation should return raw BibTeX string."""
        from semantic_scholar_mcp.server import CitationExportInput, export_citation

        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        bibtex = "@article{vaswani2017attention,\n  title={Attention Is All You Need}\n}"

        respx.get(url).mock(
            return_value=Response(
                200, json={"title": "Attention Is All You Need", "citationStyles": {"bibtex": bibtex}}
            )
        )

        params = CitationExportInput(paper_id=paper_id)
        result = await export_citation(params)

        assert result == bibtex
        assert "@article" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_export_no_bibtex_available(self, reset_client):
        """export_citation should raise ToolError when no BibTeX exists."""
        from semantic_scholar_mcp.server import CitationExportInput, export_citation

        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        respx.get(url).mock(
            return_value=Response(200, json={"title": "No Cite Paper", "citationStyles": {}})
        )

        params = CitationExportInput(paper_id=paper_id)
        with pytest.raises(ToolError, match="No BibTeX"):
            await export_citation(params)

    @respx.mock
    @pytest.mark.asyncio
    async def test_export_unsupported_format(self, reset_client):
        """export_citation should reject unsupported formats."""
        from semantic_scholar_mcp.server import CitationExportInput, export_citation

        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        respx.get(url).mock(
            return_value=Response(200, json={"title": "Test", "citationStyles": {"bibtex": "@a{}"}})
        )

        params = CitationExportInput(paper_id=paper_id, format="apa")
        with pytest.raises(ToolError, match="Unsupported citation format"):
            await export_citation(params)

    def test_export_invalid_paper_id(self):
        """export_citation should reject invalid paper IDs."""
        from semantic_scholar_mcp.server import CitationExportInput

        with pytest.raises(Exception):
            CitationExportInput(paper_id="")

    @respx.mock
    @pytest.mark.asyncio
    async def test_export_paper_not_found(self, reset_client):
        """export_citation should raise ToolError for 404."""
        from semantic_scholar_mcp.server import CitationExportInput, export_citation

        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        respx.get(url).mock(return_value=Response(404))

        params = CitationExportInput(paper_id=paper_id)
        with pytest.raises(ToolError):
            await export_citation(params)


class TestParallelSubRequests:
    """Test that get_paper_details fires citations + references in parallel."""

    @pytest.fixture
    def reset_client(self):
        import semantic_scholar_mcp.server as server

        old_client = server._client
        server._client = None
        yield
        server._client = old_client

    @respx.mock
    @pytest.mark.asyncio
    async def test_both_citations_and_references(self, reset_client):
        """Requesting both citations and references should return both."""
        from semantic_scholar_mcp.server import PaperDetailsInput, get_paper_details

        paper_id = "a" * 40
        base_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        cit_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}/citations"
        ref_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}/references"

        respx.get(base_url).mock(
            return_value=Response(200, json={"paperId": paper_id, "title": "Main Paper"})
        )
        respx.get(cit_url).mock(
            return_value=Response(
                200, json={"data": [{"citingPaper": {"title": "Citer", "year": 2024}}]}
            )
        )
        respx.get(ref_url).mock(
            return_value=Response(
                200, json={"data": [{"citedPaper": {"title": "Reference", "year": 2020}}]}
            )
        )

        params = PaperDetailsInput(
            paper_id=paper_id, include_citations=True, include_references=True
        )
        result = await get_paper_details(params)

        assert "Citer" in result
        assert "Reference" in result


class TestEntryPoint:
    """Test module entry point."""

    def test_main_module_importable(self):
        """__main__.py should be importable."""
        import semantic_scholar_mcp.__main__  # noqa: F401

    def test_main_function_exists(self):
        """main() should be callable."""
        from semantic_scholar_mcp import main

        assert callable(main)

    def test_version_consistency(self):
        """__init__.py and server.py versions must match."""
        import semantic_scholar_mcp
        from semantic_scholar_mcp.server import __version__ as server_version

        assert semantic_scholar_mcp.__version__ == server_version
