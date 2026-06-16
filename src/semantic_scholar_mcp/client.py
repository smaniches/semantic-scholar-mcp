"""Shared HTTP client, rate limiter, and retry loop for the Semantic Scholar API.

A single :class:`httpx.AsyncClient` is reused for the process lifetime to
amortize connection setup across tool invocations. Requests are serialized
through a semaphore so the per-second rate limit (1 req/s public, 10 req/s
keyed) is enforced even when the MCP host issues tool calls in parallel.
Retries cover ``429`` and ``503`` with exponential backoff + jitter, capped
at 30 s, honoring the ``Retry-After`` header when present.

API-key resolution order (highest precedence first): the deprecated per-call
``api_key`` tool parameter, the request-scoped key bound by the Streamable
HTTP transport (see :mod:`semantic_scholar_mcp.transport`), then the
``SEMANTIC_SCHOLAR_API_KEY`` environment variable.
"""

from __future__ import annotations

import asyncio
import json as _json
import math
import os
import random
import time
import warnings
from contextvars import ContextVar
from email.utils import parsedate_to_datetime
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

# Request-scoped API key, bound per HTTP request by the Streamable HTTP
# transport middleware. Contextvars are copied into the per-request server
# task, so concurrent remote users can never observe each other's keys.
# Empty string means "no key for this request" and falls through to the env var.
_request_api_key: ContextVar[str] = ContextVar("semantic_scholar_request_api_key", default="")


def get_request_api_key() -> str:
    """Return the API key bound to the current request context ('' if none)."""
    return _request_api_key.get()


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
            # Follow 3xx so an endpoint move or HTTP→HTTPS upgrade doesn't
            # surface as a phantom error from raise_for_status().
            follow_redirects=True,
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
    if api_key is not None:
        warnings.warn(
            "Per-request api_key is deprecated and will be removed in v2.0.0. "
            "Set the SEMANTIC_SCHOLAR_API_KEY environment variable instead. "
            "See SECURITY.md for transcript-exposure risk.",
            DeprecationWarning,
            stacklevel=2,
        )
    effective_key = api_key or _request_api_key.get() or SEMANTIC_SCHOLAR_API_KEY
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
    effective_key = api_key or _request_api_key.get() or SEMANTIC_SCHOLAR_API_KEY

    async with _rate_semaphore:
        now = time.monotonic()
        elapsed = now - _last_request_time
        interval = _MIN_REQUEST_INTERVAL_KEYED if effective_key else _MIN_REQUEST_INTERVAL
        if elapsed < interval:
            await asyncio.sleep(interval - elapsed)
        _last_request_time = time.monotonic()

        return await _execute_request_with_retry(method, url, params, json_body, headers, api_key)


def _parse_retry_after(header_value: str | None, default: float) -> float:
    """Parse a ``Retry-After`` header per RFC 9110 (delay-seconds OR HTTP-date).

    Falls back to ``default`` for missing, malformed, or past-dated values —
    so a CDN that emits a date string can never crash the retry loop.
    """
    if not header_value:
        return default
    try:
        seconds = float(header_value)
    except ValueError:
        pass
    else:
        # ``float()`` accepts "nan"/"inf"; a non-finite delay would serialize to
        # invalid JSON (NaN/Infinity) downstream, so treat it as malformed.
        if math.isfinite(seconds):
            return seconds
    try:
        target = parsedate_to_datetime(header_value)
    except (TypeError, ValueError):
        return default
    # Defensive: on Python >=3.10 parsedate_to_datetime raises (never returns
    # None) for bad input, so this arm is unreachable on supported versions.
    if target is None:  # pragma: no cover
        return default
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    delta = (target - now).total_seconds()
    return max(delta, 0.0)


async def _execute_request_with_retry(
    method: str,
    url: str,
    params: dict[str, Any] | None,
    json_body: dict[str, Any] | None,
    headers: dict[str, str],
    api_key: str | None,
) -> dict[str, Any] | list[Any]:
    """Execute one request with exponential-backoff retry for transient errors.

    Retries: 429, 503, and any transport-level :class:`httpx.RequestError`
    (timeouts, connect/read errors, DNS hiccups, remote-protocol errors).
    Non-retriable status codes raise a typed exception via :func:`handle_error`.
    """
    client = await get_client()

    def backoff(n: int) -> float:
        # Jitter spreads retry timing to avoid thundering-herd; it is not a
        # security primitive, so the stdlib PRNG is correct here (not secrets).
        return float(RETRY_BACKOFF_BASE * (2**n) + random.uniform(0, 0.5))  # nosec B311

    for attempt in range(MAX_RETRIES + 1):
        try:
            if method == "GET":
                resp = await client.get(url, params=params, headers=headers)
            else:
                resp = await client.post(url, params=params, json=json_body, headers=headers)
            resp.raise_for_status()
            try:
                return cast(dict[str, Any] | list[Any], resp.json())
            except _json.JSONDecodeError as e:
                # 2xx but the body isn't JSON: corporate-proxy HTML page, an
                # unexpected 204 No Content, or an API regression. Surface a
                # typed error rather than crashing the tool execution.
                preview = (resp.text or "")[:120].replace("\n", " ")
                raise SemanticScholarError(
                    f"API returned non-JSON response (Content-Type: "
                    f"{resp.headers.get('Content-Type', 'unknown')}). Body preview: {preview!r}"
                ) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (429, 503) and attempt < MAX_RETRIES:
                default = RETRY_BACKOFF_BASE * (2**attempt)
                retry_after = (
                    _parse_retry_after(e.response.headers.get("Retry-After"), default)
                    if status == 429
                    else default
                )
                # Jitter (non-security) again; see backoff() above.
                wait = min(retry_after + random.uniform(0, 0.5), 30.0)  # nosec B311
                logger.warning(
                    "HTTP %d. Retry %d/%d after %.1fs", status, attempt + 1, MAX_RETRIES, wait
                )
                await asyncio.sleep(wait)
                continue
            # Non-retriable or exhausted: raise typed exception.
            handle_error(
                status,
                api_key,
                retry_after=_parse_retry_after(e.response.headers.get("Retry-After"), 0.0) or None,
            )
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                wait = backoff(attempt)
                logger.warning("Timeout. Retry %d/%d after %.1fs", attempt + 1, MAX_RETRIES, wait)
                await asyncio.sleep(wait)
                continue
            raise SemanticScholarError("Request timed out after all retries") from None
        except httpx.RequestError as e:
            # Transport-level transient errors: ConnectError, ReadError,
            # RemoteProtocolError, etc. (TimeoutException is a sibling and is
            # already handled above.) Retry with backoff before giving up.
            if attempt < MAX_RETRIES:
                wait = backoff(attempt)
                logger.warning(
                    "Network error %s. Retry %d/%d after %.1fs",
                    type(e).__name__,
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            raise SemanticScholarError(
                f"Network error after {MAX_RETRIES} retries: {type(e).__name__}: {e}"
            ) from e

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
    "get_request_api_key",
    "handle_error",
    "make_request",
]
