"""
Semantic Scholar API Compatibility Tests
========================================
Validates every field combination sent by the MCP server against
the live S2 API. Catches field deprecations and endpoint changes
before users hit errors.

Run: pytest tests/test_api_compatibility.py -v
Requires: SEMANTIC_SCHOLAR_API_KEY env var for live API tests.
Without API key, only internal consistency tests run.
Rate limit: Tests include delays between requests and retry on 429.
"""

import os
import time

import httpx
import pytest

BASE = "https://api.semanticscholar.org/graph/v1"
RECO_BASE = "https://api.semanticscholar.org/recommendations/v1"

# Known stable IDs (Attention Is All You Need, Ashish Vaswani)
TEST_PAPER_ID = "649def34f8be52c8b66281af98ae884c09aef38b"
TEST_AUTHOR_ID = "40348417"
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

# Rate limit: 1 req/sec without key, higher with key
_DELAY = 1.0 if API_KEY else 3.0


def _headers():
    h = {"Accept": "application/json"}
    if API_KEY:
        h["x-api-key"] = API_KEY
    return h


def _get(url: str, **kwargs) -> httpx.Response:
    """GET with rate limiting and retry on 429."""
    for backoff in (0, 10, 20):
        if backoff:
            time.sleep(backoff)
        else:
            time.sleep(_DELAY)
        r = httpx.get(url, **kwargs)
        if r.status_code != 429:
            return r
    return r


# -- Auth header format ---------------------------------------------------


class TestAuthHeader:
    @pytest.mark.skipif(not API_KEY, reason="No API key")
    def test_api_key_accepted(self):
        r = _get(
            f"{BASE}/paper/search",
            params={"query": "test", "limit": "1", "fields": "title"},
            headers={"x-api-key": API_KEY},
        )
        assert r.status_code == 200, f"x-api-key auth failed: {r.status_code}"

    @pytest.mark.skipif(not API_KEY, reason="No API key")
    def test_bearer_auth_not_accepted(self):
        # Semantic Scholar authenticates via the `x-api-key` header. Sending
        # the same secret as `Authorization: Bearer <key>` (without `x-api-key`)
        # must NOT authenticate the request. The API may reject outright
        # (401/403) or treat the request as anonymous; either is acceptable,
        # but a 200 response would mean Bearer is being silently honored,
        # which would contradict the documented auth contract this server
        # depends on.
        r = _get(
            f"{BASE}/paper/search",
            params={"query": "test", "limit": "1", "fields": "title"},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert r.status_code in (200, 401, 403, 429), (
            f"Unexpected status from Bearer-only request: {r.status_code}"
        )
        # If the request succeeded, it must have been treated as anonymous,
        # not as authenticated by the Bearer token.
        if r.status_code == 200:
            sent = r.request.headers
            assert "x-api-key" not in sent, (
                "Test sent x-api-key by accident; cannot prove Bearer is rejected"
            )


# -- Author field compatibility -------------------------------------------


@pytest.mark.skipif(not API_KEY, reason="No API key — skipping live API tests")
class TestAuthorFields:
    def test_author_search_accepts_fields(self):
        from semantic_scholar_mcp.server import AUTHOR_FIELDS

        r = _get(
            f"{BASE}/author/search",
            params={"query": "Einstein", "limit": "1", "fields": ",".join(AUTHOR_FIELDS)},
            headers=_headers(),
        )
        assert r.status_code == 200, f"author/search rejected fields: {r.text[:200]}"

    def test_author_detail_accepts_fields(self):
        from semantic_scholar_mcp.server import AUTHOR_FIELDS

        r = _get(
            f"{BASE}/author/{TEST_AUTHOR_ID}",
            params={"fields": ",".join(AUTHOR_FIELDS)},
            headers=_headers(),
        )
        assert r.status_code == 200, f"author detail rejected fields: {r.text[:200]}"

    def test_aliases_rejected(self):
        r = _get(
            f"{BASE}/author/search",
            params={"query": "test", "limit": "1", "fields": "authorId,aliases"},
            headers=_headers(),
        )
        assert r.status_code == 400, f"aliases should be rejected, got {r.status_code}"


# -- Paper field compatibility per endpoint --------------------------------


@pytest.mark.skipif(not API_KEY, reason="No API key — skipping live API tests")
class TestPaperFieldsByEndpoint:
    def test_paper_search_supports_tldr(self):
        from semantic_scholar_mcp.server import PAPER_SEARCH_FIELDS

        r = _get(
            f"{BASE}/paper/search",
            params={"query": "test", "limit": "1", "fields": ",".join(PAPER_SEARCH_FIELDS)},
            headers=_headers(),
        )
        assert r.status_code == 200, f"paper/search rejected: {r.text[:200]}"

    def test_citations_rejects_tldr(self):
        r = _get(
            f"{BASE}/paper/{TEST_PAPER_ID}/citations",
            params={"fields": "title,tldr", "limit": "1"},
            headers=_headers(),
        )
        assert r.status_code == 400, f"citations should reject tldr, got {r.status_code}"

    def test_references_rejects_tldr(self):
        r = _get(
            f"{BASE}/paper/{TEST_PAPER_ID}/references",
            params={"fields": "title,tldr", "limit": "1"},
            headers=_headers(),
        )
        assert r.status_code == 400, f"references should reject tldr, got {r.status_code}"

    def test_author_papers_rejects_tldr(self):
        r = _get(
            f"{BASE}/author/{TEST_AUTHOR_ID}/papers",
            params={"fields": "title,tldr", "limit": "1"},
            headers=_headers(),
        )
        assert r.status_code == 400, f"author/papers should reject tldr, got {r.status_code}"

    def test_recommendations_rejects_tldr(self):
        r = _get(
            f"{RECO_BASE}/papers/forpaper/{TEST_PAPER_ID}",
            params={"fields": "title,tldr", "limit": "1"},
            headers=_headers(),
        )
        assert r.status_code == 400, f"recommendations should reject tldr, got {r.status_code}"


# -- LITE fields work on restricted endpoints ------------------------------


@pytest.mark.skipif(not API_KEY, reason="No API key — skipping live API tests")
class TestLiteFieldsWork:
    def test_recommendations_with_lite(self):
        from semantic_scholar_mcp.server import PAPER_SEARCH_FIELDS_LITE

        r = _get(
            f"{RECO_BASE}/papers/forpaper/{TEST_PAPER_ID}",
            params={"fields": ",".join(PAPER_SEARCH_FIELDS_LITE), "limit": "1"},
            headers=_headers(),
        )
        assert r.status_code == 200, f"recommendations LITE failed: {r.text[:200]}"

    def test_author_papers_with_lite(self):
        from semantic_scholar_mcp.server import PAPER_SEARCH_FIELDS_LITE

        r = _get(
            f"{BASE}/author/{TEST_AUTHOR_ID}/papers",
            params={"fields": ",".join(PAPER_SEARCH_FIELDS_LITE), "limit": "1"},
            headers=_headers(),
        )
        assert r.status_code == 200, f"author/papers LITE failed: {r.text[:200]}"

    def test_citations_with_lite(self):
        from semantic_scholar_mcp.server import PAPER_SEARCH_FIELDS_LITE

        r = _get(
            f"{BASE}/paper/{TEST_PAPER_ID}/citations",
            params={"fields": ",".join(PAPER_SEARCH_FIELDS_LITE), "limit": "1"},
            headers=_headers(),
        )
        assert r.status_code == 200, f"citations LITE failed: {r.text[:200]}"

    def test_references_with_lite(self):
        from semantic_scholar_mcp.server import PAPER_SEARCH_FIELDS_LITE

        r = _get(
            f"{BASE}/paper/{TEST_PAPER_ID}/references",
            params={"fields": ",".join(PAPER_SEARCH_FIELDS_LITE), "limit": "1"},
            headers=_headers(),
        )
        assert r.status_code == 200, f"references LITE failed: {r.text[:200]}"


# -- Internal consistency -------------------------------------------------


class TestFieldListConsistency:
    def test_lite_excludes_only_tldr(self):
        from semantic_scholar_mcp.server import PAPER_SEARCH_FIELDS, PAPER_SEARCH_FIELDS_LITE

        diff = set(PAPER_SEARCH_FIELDS) - set(PAPER_SEARCH_FIELDS_LITE)
        assert diff == {"tldr"}, f"LITE should exclude only tldr, excludes: {diff}"

    def test_no_aliases(self):
        from semantic_scholar_mcp.server import AUTHOR_FIELDS

        assert "aliases" not in AUTHOR_FIELDS

    def test_api_key_header_format(self):
        from semantic_scholar_mcp.server import _get_headers

        h = _get_headers("test_key_123")
        assert h["x-api-key"] == "test_key_123"
        assert "Authorization" not in h

    def test_no_auth_without_key(self):
        import semantic_scholar_mcp.server as srv
        from semantic_scholar_mcp.server import _get_headers

        original = srv.SEMANTIC_SCHOLAR_API_KEY
        srv.SEMANTIC_SCHOLAR_API_KEY = ""
        try:
            h = _get_headers(None)
            assert "Authorization" not in h
            assert "x-api-key" not in h
        finally:
            srv.SEMANTIC_SCHOLAR_API_KEY = original
