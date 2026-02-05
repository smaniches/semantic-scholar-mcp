"""
Tests for semantic-scholar-mcp server.

Coverage targets:
- _handle_error: All 7 status codes map to correct exception types
- _format_paper_markdown: Missing fields, empty authors, None values
- Retry paths: 429->retry->200, timeout->retry->200, 404->immediate raise
- Paper ID validation: Valid/invalid formats
"""

from __future__ import annotations

import pytest
import httpx
import respx
from httpx import Response

from semantic_scholar_mcp.server import (
    __version__,
    _handle_error,
    _format_paper_markdown,
    _validate_paper_id,
    _execute_request_with_retry,
    _get_client,
    SemanticScholarError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ValidationError,
    ServerError,
    SEMANTIC_SCHOLAR_API_BASE,
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

        client = await _get_client()
        result = await _execute_request_with_retry(
            "GET", url, {"query": "test"}, None, {}, None
        )

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

        client = await _get_client()
        result = await _execute_request_with_retry(
            "GET", url, None, None, {}, None
        )

        assert result == {"data": []}
        assert route.call_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_404_no_retry(self, reset_client):
        """404 should raise immediately without retry."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/invalid-id"

        route = respx.get(url).mock(return_value=Response(404))

        client = await _get_client()
        with pytest.raises(NotFoundError):
            await _execute_request_with_retry(
                "GET", url, None, None, {}, None
            )

        # Should only be called once - no retry for 404
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_503_retries_then_raises(self, reset_client):
        """503 should retry max times then raise."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        # All calls return 503
        route = respx.get(url).mock(return_value=Response(503))

        client = await _get_client()
        with pytest.raises(ServerError):
            await _execute_request_with_retry(
                "GET", url, None, None, {}, None
            )

        # Should retry MAX_RETRIES times (3) + 1 initial = 4 calls
        assert route.call_count == 4

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_retries_then_raises(self, reset_client):
        """Timeout should retry max times then raise."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        route = respx.get(url).mock(
            side_effect=httpx.TimeoutException("Connection timed out")
        )

        client = await _get_client()
        with pytest.raises(SemanticScholarError) as exc_info:
            await _execute_request_with_retry(
                "GET", url, None, None, {}, None
            )

        assert "timed out" in str(exc_info.value)
        assert route.call_count == 4  # MAX_RETRIES + 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_400_no_retry(self, reset_client):
        """400 should raise immediately without retry."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        route = respx.get(url).mock(return_value=Response(400))

        client = await _get_client()
        with pytest.raises(ValidationError):
            await _execute_request_with_retry(
                "GET", url, None, None, {}, None
            )

        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_401_no_retry(self, reset_client):
        """401 should raise immediately without retry."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        route = respx.get(url).mock(return_value=Response(401))

        client = await _get_client()
        with pytest.raises(AuthenticationError):
            await _execute_request_with_retry(
                "GET", url, None, None, {}, None
            )

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

        client = await _get_client()
        result = await _execute_request_with_retry(
            "GET", url, None, None, {}, None
        )

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

        client = await _get_client()
        result = await _execute_request_with_retry(
            "POST", url, None, {"ids": ["123"]}, {}, None
        )

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
        from semantic_scholar_mcp.server import search_papers, PaperSearchInput

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(200, json={
            "total": 1,
            "data": [{
                "paperId": "123",
                "title": "Test Paper",
                "year": 2024,
                "citationCount": 10,
                "influentialCitationCount": 2,
            }]
        }))

        params = PaperSearchInput(query="test query")
        result = await search_papers(params)

        assert "Test Paper" in result
        assert "2024" in result
        assert "Search Results" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_papers_success_json(self, reset_client):
        """search_papers should return JSON when requested."""
        from semantic_scholar_mcp.server import search_papers, PaperSearchInput, ResponseFormat

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(200, json={
            "total": 1,
            "data": [{"paperId": "123", "title": "Test"}]
        }))

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
        from semantic_scholar_mcp.server import search_papers, PaperSearchInput

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
            offset=10
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
        from semantic_scholar_mcp.server import search_papers, PaperSearchInput

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(500))

        params = PaperSearchInput(query="test")
        result = await search_papers(params)

        assert "Error" in result


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
        from semantic_scholar_mcp.server import get_paper_details, PaperDetailsInput

        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        respx.get(url).mock(return_value=Response(200, json={
            "paperId": paper_id,
            "title": "Test Paper",
            "year": 2024,
            "citationCount": 10,
            "influentialCitationCount": 2,
        }))

        params = PaperDetailsInput(paper_id=paper_id)
        result = await get_paper_details(params)

        assert "Test Paper" in result
        assert "Paper Details" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_details_with_citations(self, reset_client):
        """get_paper_details should include citations when requested."""
        from semantic_scholar_mcp.server import get_paper_details, PaperDetailsInput

        paper_id = "a" * 40
        base_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        cit_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}/citations"

        respx.get(base_url).mock(return_value=Response(200, json={
            "paperId": paper_id,
            "title": "Main Paper",
            "year": 2024,
            "citationCount": 100,
            "influentialCitationCount": 10,
        }))
        respx.get(cit_url).mock(return_value=Response(200, json={
            "data": [{"citingPaper": {"title": "Citing Paper", "year": 2024, "citationCount": 5}}]
        }))

        params = PaperDetailsInput(paper_id=paper_id, include_citations=True)
        result = await get_paper_details(params)

        assert "Main Paper" in result
        assert "Citing Paper" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_details_with_references(self, reset_client):
        """get_paper_details should include references when requested."""
        from semantic_scholar_mcp.server import get_paper_details, PaperDetailsInput

        paper_id = "a" * 40
        base_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        ref_url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}/references"

        respx.get(base_url).mock(return_value=Response(200, json={
            "paperId": paper_id,
            "title": "Main Paper",
            "year": 2024,
            "citationCount": 100,
            "influentialCitationCount": 10,
        }))
        respx.get(ref_url).mock(return_value=Response(200, json={
            "data": [{"citedPaper": {"title": "Referenced Paper", "year": 2020, "citationCount": 500}}]
        }))

        params = PaperDetailsInput(paper_id=paper_id, include_references=True)
        result = await get_paper_details(params)

        assert "Main Paper" in result
        assert "Referenced Paper" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_details_invalid_id(self, reset_client):
        """get_paper_details should validate paper ID."""
        from semantic_scholar_mcp.server import get_paper_details, PaperDetailsInput

        params = PaperDetailsInput(paper_id="invalid-id")
        result = await get_paper_details(params)

        assert "Error" in result


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
            get_recommendations, PaperRecommendationsInput, RECOMMENDATIONS_BASE
        )

        paper_id = "a" * 40
        url = f"{RECOMMENDATIONS_BASE}/papers/forpaper/{paper_id}"
        respx.get(url).mock(return_value=Response(200, json={
            "recommendedPapers": [{
                "paperId": "b" * 40,
                "title": "Recommended Paper",
                "year": 2024,
                "citationCount": 50,
                "influentialCitationCount": 5,
            }]
        }))

        params = PaperRecommendationsInput(paper_id=paper_id)
        result = await get_recommendations(params)

        assert "Recommended Paper" in result
        assert "Recommendations" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_recommendations_invalid_id(self, reset_client):
        """get_recommendations should validate paper ID."""
        from semantic_scholar_mcp.server import get_recommendations, PaperRecommendationsInput

        params = PaperRecommendationsInput(paper_id="invalid-id")
        result = await get_recommendations(params)

        assert "Error" in result


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
        from semantic_scholar_mcp.server import search_authors, AuthorSearchInput

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/search"
        respx.get(url).mock(return_value=Response(200, json={
            "total": 1,
            "data": [{
                "authorId": "123",
                "name": "John Doe",
                "hIndex": 50,
                "paperCount": 100,
                "citationCount": 5000,
            }]
        }))

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
        from semantic_scholar_mcp.server import get_author_details, AuthorDetailsInput

        author_id = "123"
        base_url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}"
        papers_url = f"{SEMANTIC_SCHOLAR_API_BASE}/author/{author_id}/papers"

        respx.get(base_url).mock(return_value=Response(200, json={
            "authorId": author_id,
            "name": "Jane Smith",
            "hIndex": 45,
            "paperCount": 80,
            "citationCount": 4000,
        }))
        respx.get(papers_url).mock(return_value=Response(200, json={
            "data": [{"title": "Author Paper", "year": 2024, "citationCount": 100}]
        }))

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
        from semantic_scholar_mcp.server import get_bulk_papers, BulkPaperInput

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"
        respx.post(url).mock(return_value=Response(200, json=[
            {"paperId": "a" * 40, "title": "Paper 1", "citationCount": 10},
            {"paperId": "b" * 40, "title": "Paper 2", "citationCount": 20},
        ]))

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
        from semantic_scholar_mcp.server import get_bulk_papers, BulkPaperInput

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"
        respx.post(url).mock(return_value=Response(200, json=[
            {"paperId": "a" * 40, "title": "Paper 1", "citationCount": 10},
            None,  # Paper not found
        ]))

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
        from semantic_scholar_mcp.server import get_bulk_papers, BulkPaperInput

        params = BulkPaperInput(paper_ids=["invalid-id-1", "invalid-id-2"])
        result = await get_bulk_papers(params)

        assert "Error" in result
        assert "Invalid paper ID" in result


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
        from semantic_scholar_mcp.server import _make_request, RECOMMENDATIONS_BASE

        url = f"{RECOMMENDATIONS_BASE}/papers/forpaper/123"
        respx.get(url).mock(return_value=Response(200, json={"recommendedPapers": []}))

        result = await _make_request(
            "GET", "papers/forpaper/123", base_url=RECOMMENDATIONS_BASE
        )
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
        # x-api-key should only be present if env var is set
        # Don't check for x-api-key absence as env might have it

    def test_headers_with_api_key(self):
        """_get_headers should include API key when provided."""
        from semantic_scholar_mcp.server import _get_headers

        headers = _get_headers(api_key="test-api-key")
        assert headers["x-api-key"] == "test-api-key"


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
