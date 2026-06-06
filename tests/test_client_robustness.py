"""Tests covering the hardened paths in :mod:`semantic_scholar_mcp.client`.

Each test pins a specific failure mode that previously crashed the tool
execution or surfaced a misleading error to the MCP client.
"""

from __future__ import annotations

from email.utils import format_datetime

import httpx
import pytest
import respx
from httpx import Response

from semantic_scholar_mcp import client
from semantic_scholar_mcp.client import _parse_retry_after, make_request
from semantic_scholar_mcp.errors import SemanticScholarError

# ──────────────────────────────────────────────────────────────────────────────
# _parse_retry_after — RFC 9110 supports delay-seconds OR HTTP-date
# ──────────────────────────────────────────────────────────────────────────────


class TestParseRetryAfter:
    def test_missing_header_returns_default(self):
        assert _parse_retry_after(None, default=42.0) == 42.0
        assert _parse_retry_after("", default=7.5) == 7.5

    def test_delay_seconds(self):
        assert _parse_retry_after("30", default=99.0) == 30.0
        assert _parse_retry_after("0.5", default=99.0) == 0.5

    def test_http_date_in_future(self):
        from datetime import datetime, timedelta, timezone

        future = datetime.now(timezone.utc) + timedelta(seconds=60)
        header = format_datetime(future, usegmt=True)
        result = _parse_retry_after(header, default=99.0)
        # Allow a generous tolerance for execution time.
        assert 50.0 <= result <= 65.0

    def test_http_date_in_past_returns_zero(self):
        from datetime import datetime, timedelta, timezone

        past = datetime.now(timezone.utc) - timedelta(seconds=300)
        header = format_datetime(past, usegmt=True)
        assert _parse_retry_after(header, default=99.0) == 0.0

    def test_malformed_header_falls_back(self):
        assert _parse_retry_after("not a number or date", default=12.5) == 12.5

    def test_non_finite_header_falls_back(self):
        # float() accepts "nan"/"inf"; a non-finite delay must not leak through,
        # or it would serialize to invalid JSON (NaN/Infinity) downstream.
        assert _parse_retry_after("nan", default=8.0) == 8.0
        assert _parse_retry_after("inf", default=8.0) == 8.0
        assert _parse_retry_after("-inf", default=8.0) == 8.0
        assert _parse_retry_after("Infinity", default=8.0) == 8.0


# ──────────────────────────────────────────────────────────────────────────────
# Bad JSON response — surface a typed error, don't crash
# ──────────────────────────────────────────────────────────────────────────────


class TestNonJsonResponse:
    @respx.mock
    @pytest.mark.asyncio
    async def test_html_body_raises_typed_error(self, reset_client, reset_rate_limit):
        # Corporate-proxy HTML page returned with a 200 status.
        url = f"{client.SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                content=b"<html><body>Login required</body></html>",
                headers={"Content-Type": "text/html"},
            )
        )

        with pytest.raises(SemanticScholarError, match="non-JSON response"):
            await make_request("GET", "paper/search", params={"query": "x"})


# ──────────────────────────────────────────────────────────────────────────────
# Transient network errors — retry, don't crash
# ──────────────────────────────────────────────────────────────────────────────


class TestNetworkErrorRetry:
    @respx.mock
    @pytest.mark.asyncio
    async def test_connect_error_retries_then_succeeds(
        self, monkeypatch, reset_client, reset_rate_limit
    ):
        # First call raises ConnectError, second returns 200.
        url = f"{client.SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(
            side_effect=[
                httpx.ConnectError("DNS resolution failed"),
                Response(200, json={"total": 0, "data": []}),
            ]
        )
        # Skip the backoff sleep so the test runs fast.
        import semantic_scholar_mcp.client as client_mod

        async def _no_sleep(_):
            return None

        monkeypatch.setattr(client_mod.asyncio, "sleep", _no_sleep)

        result = await make_request("GET", "paper/search", params={"query": "x"})
        assert result == {"total": 0, "data": []}

    @respx.mock
    @pytest.mark.asyncio
    async def test_connect_error_exhausts_retries(
        self, monkeypatch, reset_client, reset_rate_limit
    ):
        url = f"{client.SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(side_effect=httpx.ConnectError("connection refused"))
        import semantic_scholar_mcp.client as client_mod

        async def _no_sleep(_):
            return None

        monkeypatch.setattr(client_mod.asyncio, "sleep", _no_sleep)

        with pytest.raises(SemanticScholarError, match="Network error after .* retries"):
            await make_request("GET", "paper/search", params={"query": "x"})


# ──────────────────────────────────────────────────────────────────────────────
# Redirect following — 3xx must transparently follow
# ──────────────────────────────────────────────────────────────────────────────


class TestRedirectFollowing:
    @respx.mock
    @pytest.mark.asyncio
    async def test_307_redirect_is_followed(self, reset_client, reset_rate_limit):
        original = f"{client.SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        redirected = f"{client.SEMANTIC_SCHOLAR_API_BASE}/paper/search/v2"
        respx.get(original).mock(return_value=Response(307, headers={"Location": redirected}))
        respx.get(redirected).mock(
            return_value=Response(200, json={"total": 1, "data": [{"paperId": "abc"}]})
        )

        result = await make_request("GET", "paper/search", params={"query": "x"})
        assert isinstance(result, dict)
        assert result["total"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Per-request api_key deprecation warning
# ──────────────────────────────────────────────────────────────────────────────


class TestApiKeyDeprecationWarning:
    def test_warns_when_api_key_provided(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            from semantic_scholar_mcp.client import get_headers

            get_headers(api_key="test-key")

    def test_no_warning_when_api_key_is_none(self):
        import warnings

        from semantic_scholar_mcp.client import get_headers

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            get_headers(api_key=None)


# ──────────────────────────────────────────────────────────────────────────────
# is_valid_paper_id — injection-character rejection parity with validate_paper_id
# ──────────────────────────────────────────────────────────────────────────────


class TestIsValidPaperIdInjection:
    """Verify is_valid_paper_id rejects the same injection characters as validate_paper_id."""

    def test_rejects_null_byte(self):
        from semantic_scholar_mcp.validators import is_valid_paper_id

        assert not is_valid_paper_id("DOI:10.1234\x00injection")

    def test_rejects_query_string(self):
        from semantic_scholar_mcp.validators import is_valid_paper_id

        assert not is_valid_paper_id("DOI:10.1234?q=1")

    def test_rejects_fragment(self):
        from semantic_scholar_mcp.validators import is_valid_paper_id

        assert not is_valid_paper_id("URL:https://example.com#frag")

    def test_rejects_path_traversal(self):
        from semantic_scholar_mcp.validators import is_valid_paper_id

        assert not is_valid_paper_id("DOI:../etc/passwd")

    def test_accepts_valid_doi(self):
        from semantic_scholar_mcp.validators import is_valid_paper_id

        assert is_valid_paper_id("DOI:10.1038/s41586-021-03819-2")

    def test_accepts_valid_arxiv(self):
        from semantic_scholar_mcp.validators import is_valid_paper_id

        assert is_valid_paper_id("ARXIV:1706.03762")

    def test_accepts_valid_hex_id(self):
        from semantic_scholar_mcp.validators import is_valid_paper_id

        assert is_valid_paper_id("649def34f8be52c8b66281af98ae884c09aef38b")
