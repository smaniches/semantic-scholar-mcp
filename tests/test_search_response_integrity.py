"""Offline contracts: malformed provider responses are not zero search results."""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response
from mcp.server.fastmcp.exceptions import ToolError

from semantic_scholar_mcp.server import (
    SEMANTIC_SCHOLAR_API_BASE,
    AuthorSearchInput,
    PaperSearchInput,
    ResponseFormat,
    _require_search_page,
    search_authors,
    search_papers,
)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"error": "upstream unavailable"},
        {"total": 0},
        {"data": []},
        {"total": None, "data": []},
        {"total": True, "data": []},
        {"total": -1, "data": []},
        {"total": "0", "data": []},
        {"total": 0, "data": None},
        {"total": 0, "data": {}},
        {"total": 1, "data": [None]},
        {"total": 1, "data": [{}]},
        {"total": 1, "data": [{"paperId": None}]},
        {"total": 1, "data": [{"paperId": "  "}]},
    ],
)
def test_invalid_paper_search_envelope_is_not_zero_matches(payload: object) -> None:
    """All malformed successful-HTTP search bodies must fail explicitly."""
    with pytest.raises(ToolError, match="Invalid Semantic Scholar paper search response"):
        _require_search_page(payload, record_type="paper", identifier_key="paperId")


@pytest.mark.parametrize(
    ("kind", "id_key", "sample"),
    [
        ("paper", "paperId", "sample-id"),
        ("author", "authorId", "12345"),
    ],
)
def test_real_empty_and_valid_records_preserve_search_semantics(
    kind: str, id_key: str, sample: str
) -> None:
    empty = {"total": 0, "data": []}
    assert _require_search_page(empty, record_type=kind, identifier_key=id_key) == (0, [])
    record = {id_key: sample}
    valid = {"total": 1, "data": [record]}
    assert _require_search_page(valid, record_type=kind, identifier_key=id_key) == (
        1,
        [record],
    )


@pytest.mark.parametrize(
    ("kind", "path", "id_key"),
    [
        ("paper", "paper/search", "paperId"),
        ("author", "author/search", "authorId"),
    ],
)
@respx.mock
@pytest.mark.asyncio
async def test_malformed_http_200_fails_before_reporting_no_matches(
    kind: str, path: str, id_key: str, reset_all: object
) -> None:
    """An API error envelope must not be rendered as a successful search."""
    route = respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/{path}").mock(
        return_value=Response(200, json={"error": "temporarily unavailable"})
    )
    if kind == "paper":
        with pytest.raises(ToolError, match="Invalid Semantic Scholar paper search response"):
            await search_papers(PaperSearchInput(query="biomedical"))
    else:
        with pytest.raises(ToolError, match="Invalid Semantic Scholar author search response"):
            await search_authors(AuthorSearchInput(query="researcher"))
    assert route.called


@pytest.mark.parametrize(
    ("kind", "path"),
    [("paper", "paper/search"), ("author", "author/search")],
)
@respx.mock
@pytest.mark.asyncio
async def test_valid_empty_provider_response_remains_zero(
    kind: str, path: str, reset_all: object
) -> None:
    respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/{path}").mock(
        return_value=Response(200, json={"total": 0, "data": []})
    )
    if kind == "paper":
        raw = await search_papers(
            PaperSearchInput(query="nonexistent", response_format=ResponseFormat.JSON)
        )
        assert json.loads(raw) == {"query": "nonexistent", "total": 0, "papers": []}
        text = await search_papers(PaperSearchInput(query="nonexistent"))
        assert "showing 0)" in text
    else:
        raw = await search_authors(
            AuthorSearchInput(query="nonexistent", response_format=ResponseFormat.JSON)
        )
        assert json.loads(raw) == {"query": "nonexistent", "total": 0, "authors": []}
