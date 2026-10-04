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

    def test_public_interval_ignores_authenticated_override(self, monkeypatch):
        """Public requests keep their conservative local interval."""
        from semantic_scholar_mcp import client as _ssm_client_mod

        monkeypatch.setenv("SEMANTIC_SCHOLAR_MIN_SECONDS_BETWEEN_REQUESTS", "not-a-number")
        assert _ssm_client_mod.get_min_request_interval(False) == 1.0

    def test_authenticated_interval_defaults_below_one_rps(self, monkeypatch):
        """Introductory API keys default below Semantic Scholar's 1 RPS ceiling."""
        from semantic_scholar_mcp import client as _ssm_client_mod

        monkeypatch.delenv("SEMANTIC_SCHOLAR_MIN_SECONDS_BETWEEN_REQUESTS", raising=False)
        assert _ssm_client_mod.get_min_request_interval(True) == 1.1

    def test_authenticated_interval_can_be_configured(self, monkeypatch):
        """Higher reviewed quotas can opt into a shorter interval."""
        from semantic_scholar_mcp import client as _ssm_client_mod

        monkeypatch.setenv("SEMANTIC_SCHOLAR_MIN_SECONDS_BETWEEN_REQUESTS", "0.25")
        assert _ssm_client_mod.get_min_request_interval(True) == 0.25

    @pytest.mark.parametrize("value", ["not-a-number", "inf", "0", "-0.5"])
    def test_authenticated_interval_rejects_invalid_values(self, monkeypatch, value):
        """Invalid rate-limit configuration must fail explicitly."""
        from semantic_scholar_mcp import client as _ssm_client_mod
        from semantic_scholar_mcp.errors import SemanticScholarError

        monkeypatch.setenv("SEMANTIC_SCHOLAR_MIN_SECONDS_BETWEEN_REQUESTS", value)
        with pytest.raises(
            SemanticScholarError,
            match="SEMANTIC_SCHOLAR_MIN_SECONDS_BETWEEN_REQUESTS",
        ):
            _ssm_client_mod.get_min_request_interval(True)

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_attempts_obey_configured_interval(self, reset_all, monkeypatch):
        """Retries must not bypass the assigned client-side request interval."""
        from semantic_scholar_mcp import client as _ssm_client_mod

        monkeypatch.setenv("SEMANTIC_SCHOLAR_MIN_SECONDS_BETWEEN_REQUESTS", "0.05")
        monkeypatch.setattr(_ssm_client_mod.random, "uniform", lambda *_: 0.0)

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        call_times: list[float] = []

        async def mock_response(request: httpx.Request) -> Response:
            call_times.append(time.monotonic())
            if len(call_times) == 1:
                return Response(429, headers={"Retry-After": "0"})
            return Response(200, json={"data": []})

        respx.get(url).mock(side_effect=mock_response)

        result = await _make_request(
            "GET", "paper/search", params={"query": "retry"}, api_key="test-key"
        )

        assert result == {"data": []}
        assert len(call_times) == 2
        assert call_times[1] - call_times[0] >= 0.04

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limiting_keyed_uses_configured_interval(self, reset_all, monkeypatch):
        """Authenticated requests honor the configured key-specific interval."""
        monkeypatch.setenv("SEMANTIC_SCHOLAR_MIN_SECONDS_BETWEEN_REQUESTS", "0.05")

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(200, json={"data": []}))

        start = time.monotonic()
        await _make_request("GET", "paper/search", params={"query": "t1"}, api_key="test-key")
        await _make_request("GET", "paper/search", params={"query": "t2"}, api_key="test-key")
        elapsed = time.monotonic() - start

        assert elapsed >= 0.04
        assert elapsed < 1.0


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
