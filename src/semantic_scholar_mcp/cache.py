"""In-memory TTL cache for paper/author lookups within a session.

Avoids redundant API calls when the same entity is fetched multiple times
(e.g. ``get_paper_details`` followed by ``export_citation`` for the same DOI).
Bounded by ``_MAX_SIZE`` with oldest-first eviction.
"""

from __future__ import annotations

import time
from typing import Any

_TTL = 300.0  # 5 minutes
_MAX_SIZE = 200

_cache: dict[str, tuple[float, Any]] = {}


def cache_get(key: str) -> Any | None:
    """Return a cached value, or ``None`` if missing or expired."""
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.monotonic() - ts > _TTL:
        del _cache[key]
        return None
    return value


def cache_set(key: str, value: Any) -> None:
    """Store ``value`` under ``key`` with oldest-first eviction at capacity."""
    if len(_cache) >= _MAX_SIZE and key not in _cache:
        oldest_key = min(_cache, key=lambda k: _cache[k][0])
        del _cache[oldest_key]
    _cache[key] = (time.monotonic(), value)


def cache_clear() -> None:
    """Drop every cached entry. Invoked on server shutdown."""
    _cache.clear()


__all__ = ["cache_clear", "cache_get", "cache_set"]
