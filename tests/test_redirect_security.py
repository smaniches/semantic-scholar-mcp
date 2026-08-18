"""Regression tests for outbound-origin enforcement across HTTP redirects."""

from __future__ import annotations

import httpx
import pytest
import respx

from semantic_scholar_mcp import client
from semantic_scholar_mcp.errors import SemanticScholarError


@pytest.mark.asyncio
async def test_trusted_semantic_scholar_origin_is_allowed():
    request = httpx.Request(
        "GET", "https://api.semanticscholar.org/graph/v1/paper/search"
    )

    await client._enforce_trusted_api_origin(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://api.semanticscholar.org/graph/v1/paper/search",
        "https://evil.example/collect",
        "https://api.semanticscholar.org:444/graph/v1/paper/search",
    ],
)
async def test_untrusted_origin_is_rejected(url: str):
    request = httpx.Request("GET", url, headers={"x-api-key": "test-secret"})

    with pytest.raises(
        SemanticScholarError,
        match="outside the trusted Semantic Scholar HTTPS origin",
    ):
        await client._enforce_trusted_api_origin(request)


@respx.mock
@pytest.mark.asyncio
async def test_cross_origin_redirect_is_blocked_before_api_key_forwarding(
    reset_client, reset_rate_limit
):
    client.SEMANTIC_SCHOLAR_API_KEY = "test-secret"
    original = f"{client.SEMANTIC_SCHOLAR_API_BASE}/paper/search"
    escaped = "https://evil.example/collect"

    original_route = respx.get(original).mock(
        return_value=httpx.Response(307, headers={"Location": escaped})
    )
    escaped_route = respx.get(escaped).mock(
        return_value=httpx.Response(200, json={"unexpected": True})
    )

    with pytest.raises(
        SemanticScholarError,
        match="outside the trusted Semantic Scholar HTTPS origin",
    ):
        await client.make_request("GET", "paper/search", params={"query": "x"})

    assert original_route.called
    assert not escaped_route.called
