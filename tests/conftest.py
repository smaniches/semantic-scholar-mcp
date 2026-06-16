"""
Shared test fixtures for semantic-scholar-mcp.

Centralizes common setup/teardown patterns to keep test files DRY.
"""

from __future__ import annotations

import pytest

from semantic_scholar_mcp import cache, client


@pytest.fixture
def reset_client():
    """Reset the global HTTP client before/after each test.

    Ensures test isolation by restoring original client state.
    """
    old_client = client._client
    client._client = None
    yield
    # Close any client created during test
    if client._client is not None and not client._client.is_closed:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(client._client.aclose())
            else:
                loop.run_until_complete(client._client.aclose())
        except RuntimeError:
            pass
    client._client = old_client


@pytest.fixture
def reset_rate_limit():
    """Reset rate-limiting state and semaphore before/after each test.

    The asyncio.Semaphore is bound to an event loop on first use.
    pytest-asyncio creates a new loop per test, so we must recreate the semaphore.

    Also neutralizes any ambient ``SEMANTIC_SCHOLAR_API_KEY`` so the public-tier
    timing assertions are hermetic regardless of the developer's shell
    environment. The constant is captured at import time into both ``client``
    (used by the rate limiter) and ``server`` (which does ``from .client import
    SEMANTIC_SCHOLAR_API_KEY``, binding its own name), so both are reset. Tests
    that exercise the keyed interval pass an explicit ``api_key`` argument and
    are unaffected.
    """
    import asyncio

    from semantic_scholar_mcp import server

    old_time = client._last_request_time
    old_semaphore = client._rate_semaphore
    old_client_key = client.SEMANTIC_SCHOLAR_API_KEY
    old_server_key = server.SEMANTIC_SCHOLAR_API_KEY
    client._last_request_time = 0.0
    client._rate_semaphore = asyncio.Semaphore(1)
    client.SEMANTIC_SCHOLAR_API_KEY = ""
    server.SEMANTIC_SCHOLAR_API_KEY = ""
    yield
    client._last_request_time = old_time
    client._rate_semaphore = old_semaphore
    client.SEMANTIC_SCHOLAR_API_KEY = old_client_key
    server.SEMANTIC_SCHOLAR_API_KEY = old_server_key


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the TTL cache before/after each test (autouse)."""
    cache.cache_clear()
    yield
    cache.cache_clear()


@pytest.fixture
def reset_all(reset_client, reset_rate_limit, reset_cache):
    """Reset HTTP client, rate-limiting state, and cache."""
    yield


@pytest.fixture
def sample_paper():
    """Return a complete paper dict for testing."""
    return {
        "paperId": "a" * 40,
        "corpusId": 12345,
        "url": "https://semanticscholar.org/paper/" + "a" * 40,
        "title": "Attention Is All You Need",
        "venue": "NeurIPS",
        "year": 2017,
        "citationCount": 50000,
        "influentialCitationCount": 5000,
        "isOpenAccess": True,
        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        "fieldsOfStudy": ["Computer Science", "Machine Learning"],
        "authors": [
            {"authorId": "1", "name": "Author One"},
            {"authorId": "2", "name": "Author Two"},
        ],
        "externalIds": {"DOI": "10.1234/test", "ArXiv": "1706.03762"},
        "tldr": {"text": "Transformers are great."},
        "abstract": "We propose a new architecture based on attention mechanisms.",
    }


@pytest.fixture
def sample_author():
    """Return a complete author dict for testing."""
    return {
        "authorId": "12345",
        "externalIds": {"ORCID": "0000-0001-2345-6789"},
        "url": "https://semanticscholar.org/author/12345",
        "name": "Jane Researcher",
        "aliases": ["J. Researcher", "Jane R."],
        "affiliations": ["MIT", "Stanford"],
        "homepage": "https://janeresearcher.com",
        "paperCount": 150,
        "citationCount": 10000,
        "hIndex": 45,
    }
