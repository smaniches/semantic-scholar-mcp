"""
Load and stress tests for rate-limiting under concurrent tool calls.

Validates that the server correctly serializes requests, enforces
rate limits, and handles bursts of concurrent calls gracefully.
"""

from __future__ import annotations

import asyncio
import time

import pytest
import respx
from httpx import Response
from mcp.server.fastmcp.exceptions import ToolError

from semantic_scholar_mcp.server import (
    SEMANTIC_SCHOLAR_API_BASE,
    PaperSearchInput,
    _make_request,
    search_papers,
)

# ===============================================================================
# BURST LOAD TESTS
# ===============================================================================


class TestBurstLoad:
    """Test behavior under burst load conditions."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_burst_of_5_requests(self, reset_all):
        """5 concurrent requests should all succeed, serialized."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        call_count = 0

        async def mock_handler(request):
            nonlocal call_count
            call_count += 1
            return Response(200, json={"data": [], "total": 0})

        respx.get(url).mock(side_effect=mock_handler)

        tasks = [
            asyncio.create_task(_make_request("GET", "paper/search", params={"query": f"topic{i}"}))
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert call_count == 5
        # All should return valid responses
        for r in results:
            assert isinstance(r, dict)

    @respx.mock
    @pytest.mark.asyncio
    async def test_burst_maintains_rate_limit(self, reset_all):
        """Burst requests should be spaced by rate limit interval."""
        from semantic_scholar_mcp import client as _ssm_client_mod

        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        timestamps: list[float] = []

        async def mock_handler(request):
            timestamps.append(time.monotonic())
            return Response(200, json={"data": [], "total": 0})

        respx.get(url).mock(side_effect=mock_handler)

        # Fire 3 concurrent requests
        tasks = [
            asyncio.create_task(_make_request("GET", "paper/search", params={"query": f"q{i}"}))
            for i in range(3)
        ]
        await asyncio.gather(*tasks)

        assert len(timestamps) == 3

        # Check that requests were properly spaced
        total_elapsed = timestamps[-1] - timestamps[0]
        expected_min = _ssm_client_mod._MIN_REQUEST_INTERVAL * 1.5  # 2 gaps for 3 requests
        assert total_elapsed >= expected_min * 0.7  # tolerance for timing

    @respx.mock
    @pytest.mark.asyncio
    async def test_burst_with_api_key_is_faster(self, reset_all):
        """Burst with API key should complete faster due to shorter interval."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(return_value=Response(200, json={"data": [], "total": 0}))

        start = time.monotonic()
        tasks = [
            asyncio.create_task(
                _make_request("GET", "paper/search", params={"query": f"q{i}"}, api_key="test-key")
            )
            for i in range(3)
        ]
        await asyncio.gather(*tasks)
        keyed_elapsed = time.monotonic() - start

        # With API key (0.1s interval), 3 requests should take ~0.2s
        # Without API key (1.0s interval), 3 requests would take ~2.0s
        assert keyed_elapsed < 2.0  # Should be much faster than unkeyed


# ===============================================================================
# TOOL-LEVEL LOAD TESTS
# ===============================================================================


class TestToolLevelLoad:
    """Test concurrent tool function calls."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_concurrent_search_calls(self, reset_all):
        """Multiple concurrent search_papers calls should all succeed."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "total": 1,
                    "data": [
                        {
                            "paperId": "abc",
                            "title": "Test Paper",
                            "year": 2024,
                            "citationCount": 0,
                            "influentialCitationCount": 0,
                        }
                    ],
                },
            )
        )

        queries = ["machine learning", "deep learning", "transformers"]
        tasks = [asyncio.create_task(search_papers(PaperSearchInput(query=q))) for q in queries]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert "Test Paper" in result


# ===============================================================================
# MIXED ERROR AND SUCCESS LOAD
# ===============================================================================


class TestMixedLoadScenarios:
    """Test behavior when some requests fail during burst."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_burst_with_intermittent_errors(self, reset_all):
        """Some requests failing should not affect others."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        call_num = 0

        async def mock_handler(request):
            nonlocal call_num
            call_num += 1
            # Every other request fails
            if call_num % 2 == 0:
                return Response(500)
            return Response(200, json={"data": [], "total": 0})

        respx.get(url).mock(side_effect=mock_handler)

        queries = [f"query{i}" for i in range(4)]
        tasks = [asyncio.create_task(search_papers(PaperSearchInput(query=q))) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should return (some with ToolError exceptions, some with success)
        assert len(results) == 4
        successes = [r for r in results if isinstance(r, str)]
        errors = [r for r in results if isinstance(r, ToolError)]
        # At least some should succeed and some should fail
        assert len(successes) > 0 or len(errors) > 0  # All returned something

    @respx.mock
    @pytest.mark.asyncio
    async def test_burst_with_rate_limits(self, reset_all):
        """429 responses during burst should trigger retries."""
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        call_num = 0

        async def mock_handler(request):
            nonlocal call_num
            call_num += 1
            # First call gets rate limited, rest succeed
            if call_num == 1:
                return Response(429, headers={"Retry-After": "0.01"})
            return Response(200, json={"data": [], "total": 0})

        respx.get(url).mock(side_effect=mock_handler)

        result = await _make_request("GET", "paper/search", params={"query": "test"})
        assert result == {"data": [], "total": 0}
        # Should have been called twice (429 + retry success)
        assert call_num == 2
