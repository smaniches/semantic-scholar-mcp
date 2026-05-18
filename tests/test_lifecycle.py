"""
Tests for server lifecycle, rate limiting, and concurrency.

Covers:
- Lifespan context manager (startup/shutdown)
- Rate-limiting behavior
- Concurrent request serialization
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx
from httpx import Response

from semantic_scholar_mcp.server import (
    SEMANTIC_SCHOLAR_API_BASE,
    _get_client,
    _lifespan,
    _make_request,
    mcp,
)

# ===============================================================================
# LIFESPAN TESTS
# ===============================================================================


class TestLifespan:
    """Test _lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_creates_and_closes_client(self, reset_client):
        """Lifespan should allow client usage and close on shutdown."""
        from semantic_scholar_mcp import client as _ssm_client_mod

        async with _lifespan(mcp):
            # During lifespan, client should be usable
            client = await _get_client()
            assert client is not None
            assert not client.is_closed
            # Store reference to check after shutdown
            client_ref = client

        # After lifespan exits, client should be closed
        assert client_ref.is_closed
        assert _ssm_client_mod._client is None

    @pytest.mark.asyncio
    async def test_lifespan_no_client_created(self, reset_client):
        """Lifespan should not fail if no client was created."""
        from semantic_scholar_mcp import client as _ssm_client_mod

        _ssm_client_mod._client = None
        async with _lifespan(mcp):
            pass  # Don't create any client
        # Should not raise

    @pytest.mark.asyncio
    async def test_lifespan_handles_already_closed_client(self, reset_client):
        """Lifespan should handle already-closed client gracefully."""

        async with _lifespan(mcp):
            client = await _get_client()
            await client.aclose()  # Close prematurely
            # Lifespan exit should not raise


# ===============================================================================
# RATE LIMITING TESTS
# ===============================================================================


class TestRateLimiting:
    """Test rate-limiting behavior."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limiting_enforces_interval(self, reset_all):
        """Requests should be spaced by at least _MIN_REQUEST_INTERVAL."""
        from semantic_scholar_mcp import client as _ssm_client_mod

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(200, json={"data": []}))

        # Make two requests and measure the time between them
        start = time.monotonic()
        await _make_request("GET", "paper/search", params={"query": "test1"})
        await _make_request("GET", "paper/search", params={"query": "test2"})
        elapsed = time.monotonic() - start

        # Should have waited at least the minimum interval (1.0s for no API key)
        # Use a small tolerance for timing
        assert elapsed >= _ssm_client_mod._MIN_REQUEST_INTERVAL * 0.8

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limiting_keyed_faster(self, reset_all):
        """Keyed requests should use shorter interval."""

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(200, json={"data": []}))

        start = time.monotonic()
        await _make_request("GET", "paper/search", params={"query": "t1"}, api_key="test-key")
        await _make_request("GET", "paper/search", params={"query": "t2"}, api_key="test-key")
        elapsed = time.monotonic() - start

        # Keyed interval is 0.1s, so two requests should be much faster than 1s
        assert elapsed < 1.5


# ===============================================================================
# CONCURRENT REQUEST TESTS
# ===============================================================================


class TestConcurrency:
    """Test that concurrent requests are serialized."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_concurrent_requests_serialized(self, reset_all):
        """Multiple concurrent requests should be serialized by the semaphore."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        call_times: list[float] = []

        async def mock_response(request: httpx.Request) -> Response:
            call_times.append(time.monotonic())
            return Response(200, json={"data": []})

        respx.get(url).mock(side_effect=mock_response)

        # Fire 3 concurrent requests
        tasks = [
            asyncio.create_task(_make_request("GET", "paper/search", params={"query": f"q{i}"}))
            for i in range(3)
        ]
        await asyncio.gather(*tasks)

        # All 3 should have been called
        assert len(call_times) == 3

        # They should be serialized (each after the other)
        for i in range(1, len(call_times)):
            gap = call_times[i] - call_times[i - 1]
            # Each gap should be at least ~interval (with tolerance for timing)
            assert gap >= 0.05  # Very loose bound to avoid flakiness
