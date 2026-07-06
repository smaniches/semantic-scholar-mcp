"""
Tests for the TTL cache module.

Covers cache get/set, TTL expiry, LRU eviction, and integration
with paper/author detail lookups.
"""

from __future__ import annotations

import time

import pytest
import respx
from httpx import Response

from semantic_scholar_mcp.server import (
    SEMANTIC_SCHOLAR_API_BASE,
    PaperDetailsInput,
    _cache,
    _cache_clear,
    _cache_get,
    _cache_set,
    get_paper_details,
)

# ===============================================================================
# CACHE UNIT TESTS
# ===============================================================================


class TestCacheOperations:
    """Test basic cache get/set/clear operations."""

    def setup_method(self):
        """Clear cache before each test."""
        _cache_clear()

    def teardown_method(self):
        """Clear cache after each test."""
        _cache_clear()

    def test_cache_set_and_get(self):
        """Should store and retrieve values."""
        _cache_set("key1", {"data": "value1"})
        assert _cache_get("key1") == {"data": "value1"}

    def test_cache_get_missing(self):
        """Should return None for missing keys."""
        assert _cache_get("nonexistent") is None

    def test_cache_clear(self):
        """Should remove all entries."""
        _cache_set("key1", "val1")
        _cache_set("key2", "val2")
        _cache_clear()
        assert _cache_get("key1") is None
        assert _cache_get("key2") is None

    def test_cache_overwrite(self):
        """Should overwrite existing entries."""
        _cache_set("key1", "old")
        _cache_set("key1", "new")
        assert _cache_get("key1") == "new"

    def test_cache_ttl_expiry(self):
        """Expired entries should return None."""
        _cache_set("key1", "value")
        # Manually set timestamp to the past
        _cache["key1"] = (time.monotonic() - 400, "value")
        assert _cache_get("key1") is None
        # Entry should be removed
        assert "key1" not in _cache

    def test_cache_not_expired(self):
        """Non-expired entries should return value."""
        _cache_set("key1", "value")
        assert _cache_get("key1") == "value"

    def test_cache_lru_eviction(self):
        """Should evict oldest entry when full."""
        from semantic_scholar_mcp.server import _CACHE_MAX_SIZE

        # Fill cache to max
        for i in range(_CACHE_MAX_SIZE):
            _cache_set(f"key{i}", f"val{i}")

        # Add one more — oldest should be evicted
        _cache_set("new_key", "new_val")

        assert _cache_get("new_key") == "new_val"
        assert len(_cache) == _CACHE_MAX_SIZE

    def test_cache_eviction_removes_the_oldest_entry(self):
        """Eviction must drop the entry with the OLDEST timestamp specifically.

        Surfaced by a mutation spot-check: flipping min() to max() in the
        eviction (evicting the newest instead) left the whole suite green,
        because only the cache size and the new key were asserted.
        """
        from semantic_scholar_mcp.server import _CACHE_MAX_SIZE

        for i in range(_CACHE_MAX_SIZE):
            _cache_set(f"key{i}", f"val{i}")
        # Backdate one entry (within TTL) so it is unambiguously the oldest.
        _cache["key7"] = (time.monotonic() - 200, "val7")

        _cache_set("new_key", "new_val")

        assert _cache_get("key7") is None, "the oldest entry should have been evicted"
        assert _cache_get("new_key") == "new_val"
        assert _cache_get("key0") == "val0", "newer entries must survive eviction"
        assert len(_cache) == _CACHE_MAX_SIZE

    def test_cache_stores_various_types(self):
        """Should handle dicts, lists, strings, None."""
        _cache_set("dict", {"a": 1})
        _cache_set("list", [1, 2, 3])
        _cache_set("str", "hello")
        _cache_set("none", None)

        assert _cache_get("dict") == {"a": 1}
        assert _cache_get("list") == [1, 2, 3]
        assert _cache_get("str") == "hello"
        assert _cache_get("none") is None  # Can't distinguish from miss


# ===============================================================================
# CACHE INTEGRATION WITH TOOLS
# ===============================================================================


class TestCacheIntegration:
    """Test that tools use the cache correctly."""

    def setup_method(self):
        _cache_clear()

    def teardown_method(self):
        _cache_clear()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper_caches_result(self, reset_all):
        """First call should cache, second should use cache."""
        paper_id = "a" * 40
        url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
        route = respx.get(url).mock(
            return_value=Response(
                200,
                json={
                    "paperId": paper_id,
                    "title": "Cached Paper",
                    "year": 2024,
                    "citationCount": 10,
                    "influentialCitationCount": 1,
                },
            )
        )

        # First call — hits API
        params = PaperDetailsInput(paper_id=paper_id)
        result1 = await get_paper_details(params)
        assert "Cached Paper" in result1
        assert route.call_count == 1

        # Second call — should use cache, not hit API again
        result2 = await get_paper_details(params)
        assert "Cached Paper" in result2
        assert route.call_count == 1  # Still 1 — cache hit

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_key_includes_paper_id(self, reset_all):
        """Different paper IDs should have different cache entries."""
        paper_id_a = "a" * 40
        paper_id_b = "b" * 40

        url_a = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id_a}"
        url_b = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id_b}"

        respx.get(url_a).mock(
            return_value=Response(
                200,
                json={
                    "paperId": paper_id_a,
                    "title": "Paper A",
                    "year": 2024,
                    "citationCount": 0,
                    "influentialCitationCount": 0,
                },
            )
        )
        respx.get(url_b).mock(
            return_value=Response(
                200,
                json={
                    "paperId": paper_id_b,
                    "title": "Paper B",
                    "year": 2024,
                    "citationCount": 0,
                    "influentialCitationCount": 0,
                },
            )
        )

        result_a = await get_paper_details(PaperDetailsInput(paper_id=paper_id_a))
        result_b = await get_paper_details(PaperDetailsInput(paper_id=paper_id_b))

        assert "Paper A" in result_a
        assert "Paper B" in result_b
