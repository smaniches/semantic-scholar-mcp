"""
Shared test fixtures for semantic-scholar-mcp.

Centralizes common setup/teardown patterns to keep test files DRY.
"""

from __future__ import annotations

import pytest

import semantic_scholar_mcp.server as server


@pytest.fixture
def reset_client():
    """Reset the global HTTP client before/after each test.

    Ensures test isolation by restoring original client state.
    """
    old_client = server._client
    server._client = None
    yield
    # Close any client created during test
    if server._client is not None and not server._client.is_closed:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(server._client.aclose())
            else:
                loop.run_until_complete(server._client.aclose())
        except RuntimeError:
            pass
    server._client = old_client


@pytest.fixture
def reset_rate_limit():
    """Reset rate-limiting state and semaphore before/after each test.

    The asyncio.Semaphore is bound to an event loop on first use.
    pytest-asyncio creates a new loop per test, so we must recreate the semaphore.
    """
    import asyncio

    old_time = server._last_request_time
    old_semaphore = server._rate_semaphore
    server._last_request_time = 0.0
    server._rate_semaphore = asyncio.Semaphore(1)
    yield
    server._last_request_time = old_time
    server._rate_semaphore = old_semaphore


@pytest.fixture
def reset_all(reset_client, reset_rate_limit):
    """Reset both HTTP client and rate-limiting state."""
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
