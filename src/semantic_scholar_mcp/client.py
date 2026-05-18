"""Shared HTTP client, rate limiter, and retry loop for the Semantic Scholar API.

A single :class:`httpx.AsyncClient` is reused for the process lifetime to
amortize connection setup across tool invocations. Requests are serialized
through a semaphore so the per-second rate limit (1 req/s public, 10 req/s
keyed) is enforced even when the MCP host issues tool calls in parallel.
Retries cover ``429`` and ``503`` with exponential backoff + jitter, capped
at 30 s, honoring the ``Retry-After`` header when present.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any, cast

import httpx

from .errors import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    SemanticScholarError,
    ServerError,
    ValidationError,
)
from .logging_config import get_logger

# Module-level config.
SEMANTIC_SCHOLAR_API_KEY: str = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_API_BASE: str = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_BASE: str = "https://api.semanticscholar.org/recommendations/v1"

# Rate-limit state.
_rate_semaphore = asyncio.Semaphore(1)
_last_request_time: float = 0.0
_MIN_REQUEST_INTERVAL = 1.0  # public tier: 1 req/sec
_MIN_REQUEST_INTERVAL_KEYED = 0.1  # keyed tier: 10 req/sec

# Retry config.
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds

# Shared client (lazy singleton).
_client: httpx.AsyncClient | None = None

logger = get_logger()


async def get_client() -> httpx.AsyncClient:
    """Return the shared :class:`httpx.AsyncClient`, creating it if needed."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=10, max_keepalive_connections=5, keepalive_expiry=30
            ),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
    return _client


async def close_client() -> None:
    """Close the shared client. Called from the FastMCP lifespan teardown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
        logger.info("HTTP client closed")


def get_headers(api_key: str | None = None) -> dict[str, str]:
    """Build request headers. The per-call ``api_key`` overrides the env var."""
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    effective_key = api_key or SEMANTIC_SCHOLAR_API_KEY
    if effective_key:
        headers["x-api-key"] = effective_key
    return headers


async def make_request(
    method: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | list[Any]:
    """Issue an HTTP request to the Semantic Scholar API.

    Serialized through ``_rate_semaphore`` and gated by the per-tier minimum
    interval, then dispatched to :func:`_execute_request_with_retry`.
    """
    global _last_request_time

    url = f"{base_url or SEMANTIC_SCHOLAR_API_BASE}/{endpoint}"
    headers = get_headers(api_key)
    effective_key = api_key or SEMANTIC_SCHOLAR_API_KEY

    async with _rate_semaphore:
        now = time.monotonic()
        elapsed = now - _last_request_time
        interval = _MIN_REQUEST_INTERVAL_KEYED if effective_key else _MIN_REQUEST_INTERVAL
        if elapsed < interval:
            await asyncio.sleep(interval - elapsed)
        _last_request_time = time.monotonic()

        return await _execute_request_with_retry(method, url, params, json_body, headers, api_key)


async def _execute_request_with_retry(
    method: str,
    url: str,
    params: dict[str, Any] | None,
    json_body: dict[str, Any] | None,
    headers: dict[str, str],
    api_key: str | None,
) -> dict[str, Any] | list[Any]:
    """Execute one request with exponential-backoff retry for 429/503/timeout."""
    client = await get_client()

    for attempt in range(MAX_RETRIES + 1):
        try:
            if method == "GET":
                resp = await client.get(url, params=params, headers=headers)
            else:
                resp = await client.post(url, params=params, json=json_body, headers=headers)
            resp.raise_for_status()
            return cast(dict[str, Any] | list[Any], resp.json())
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # Retriable: 429, 503.
            if status in (429, 503) and attempt < MAX_RETRIES:
                if status == 429:
                    retry_after = float(
                        e.response.headers.get("Retry-After", RETRY_BACKOFF_BASE * (2**attempt))
                    )
                else:
                    retry_after = RETRY_BACKOFF_BASE * (2**attempt)
                jitter = random.uniform(0, 0.5)
                wait = min(retry_after + jitter, 30.0)
                logger.warning(
                    "HTTP %d. Retry %d/%d after %.1fs", status, attempt + 1, MAX_RETRIES, wait
                )
                await asyncio.sleep(wait)
                continue
            # Non-retriable or exhausted: raise typed exception.
            retry_after_header = e.response.headers.get("Retry-After")
            handle_error(
                status,
                api_key,
                retry_after=float(retry_after_header) if retry_after_header else None,
            )
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (2**attempt) + random.uniform(0, 0.5)
                logger.warning("Timeout. Retry %d/%d after %.1fs", attempt + 1, MAX_RETRIES, wait)
                await asyncio.sleep(wait)
                continue
            raise SemanticScholarError("Request timed out after all retries") from None

    raise SemanticScholarError("Request failed: no response received")  # pragma: no cover


def handle_error(
    status: int,
    api_key: str | None = None,
    retry_after: float | None = None,
) -> None:
    """Map an HTTP status code onto a typed exception with an actionable message."""
    if status == 400:
        raise ValidationError("Bad request. Check syntax.", status_code=400)
    if status == 401:
        if api_key:
            msg = "Auth failed. Check your provided API key."
        else:
            msg = "Auth failed. Set SEMANTIC_SCHOLAR_API_KEY env var or provide api_key parameter."
        raise AuthenticationError(msg, status_code=401)
    if status == 403:
        if api_key:
            msg = "Forbidden. Your provided API key may be invalid or expired."
        else:
            msg = "Forbidden. Check SEMANTIC_SCHOLAR_API_KEY env var or provide api_key parameter."
        raise AuthenticationError(msg, status_code=403)
    if status == 404:
        raise NotFoundError("Not found. Check ID format.", status_code=404)
    if status == 429:
        if api_key:
            msg = f"Rate limited. Retry in {retry_after}s." if retry_after else "Rate limited."
        else:
            msg = (
                "Rate limited. Get a free API key for faster access: "
                "https://www.semanticscholar.org/product/api"
            )
        raise RateLimitError(msg, retry_after=retry_after)
    if status in (500, 502, 503):
        msg = "Service unavailable." if status == 503 else "Server error. Try later."
        raise ServerError(msg, status_code=status)
    raise SemanticScholarError(f"Unknown error (HTTP {status})", status_code=status)


__all__ = [
    "MAX_RETRIES",
    "RECOMMENDATIONS_BASE",
    "RETRY_BACKOFF_BASE",
    "SEMANTIC_SCHOLAR_API_BASE",
    "SEMANTIC_SCHOLAR_API_KEY",
    "close_client",
    "get_client",
    "get_headers",
    "handle_error",
    "make_request",
]
