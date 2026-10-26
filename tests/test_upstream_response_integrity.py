"""Malformed successful-HTTP responses must not masquerade as empty research results.

All fixtures are synthetic and exercise provider-response validation only.
They make no claims about actual Semantic Scholar papers or authors.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from semantic_scholar_mcp import server as s


@pytest.fixture(autouse=True)
def clear_test_cache() -> None:
    s.cache_clear()


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        {"data": None},
        {"data": "not a list"},
        {"data": [None]},
        {"data": [{"paperId": "ok"}, False]},
    ],
)
def test_invalid_result_list_cannot_silently_become_empty(response: Any) -> None:
    with pytest.raises(ToolError, match="Invalid Semantic Scholar"):
        s._required_provider_items(response, "data", "paper/search")


@pytest.mark.parametrize("total", [None, True, -1, "0", 1.5])
def test_missing_or_invalid_upstream_total_is_not_zero(total: Any) -> None:
    with pytest.raises(ToolError, match="missing valid total"):
        s._required_provider_total({"total": total}, "paper/search")


def test_invalid_total_response_envelope() -> None:
    with pytest.raises(ToolError, match="expected JSON object"):
        s._required_provider_total([], "paper/search")


def test_empty_result_list_and_exact_zero_total_are_valid() -> None:
    assert s._required_provider_items({"data": []}, "data", "paper/search") == []
    assert s._required_provider_items({"data": [{}]}, "data", "paper/search") == [{}]
    assert (
        s._required_provider_items(
            {"recommendedPapers": []}, "recommendedPapers", "recommendations"
        )
        == []
    )
    assert s._required_provider_total({"total": 0}, "paper/search") == 0
    assert s._required_provider_total({"total": 17}, "paper/search") == 17


_PAPER = "a" * 40


@pytest.mark.parametrize(
    ("tool", "params"),
    [
        ("search_papers", s.PaperSearchInput(query="evidence")),
        ("search_authors", s.AuthorSearchInput(query="Researcher")),
        ("bulk_search", s.BulkSearchInput(query="evidence")),
        ("get_recommendations", s.PaperRecommendationsInput(paper_id=_PAPER)),
        ("multi_recommend", s.MultiRecommendInput(positive_paper_ids=[_PAPER])),
        ("match_paper", s.PaperMatchInput(query="Research title")),
        ("get_paper_authors", s.PaperAuthorsInput(paper_id=_PAPER)),
        ("snippet_search", s.SnippetSearchInput(query="evidence")),
        ("get_author_details", s.AuthorDetailsInput(author_id="123", include_papers=True)),
        (
            "get_paper_details",
            s.PaperDetailsInput(paper_id=_PAPER, include_citations=True),
        ),
        (
            "get_paper_details",
            s.PaperDetailsInput(paper_id=_PAPER, include_references=True),
        ),
    ],
)
@pytest.mark.asyncio
async def test_malformed_successful_http_body_is_not_reported_as_no_results(
    monkeypatch: pytest.MonkeyPatch, tool: str, params: Any
) -> None:
    async def invalid_response(_method: str, endpoint: str, **_kwargs: Any) -> Any:
        if endpoint == f"paper/{_PAPER}":
            return {"paperId": _PAPER, "title": "Synthetic paper"}
        if endpoint == "author/123":
            return {"authorId": "123", "name": "Synthetic author"}
        # An HTTP-200 body missing mandatory result lists/metadata.
        return {}

    monkeypatch.setattr(s, "make_request", invalid_response)
    with pytest.raises(ToolError, match="Invalid Semantic Scholar"):
        await getattr(s, tool)(params)


@pytest.mark.parametrize(
    ("tool", "params", "response"),
    [
        (
            "search_papers",
            s.PaperSearchInput(query="absent", response_format=s.ResponseFormat.JSON),
            {"total": 0, "data": []},
        ),
        (
            "search_authors",
            s.AuthorSearchInput(query="absent", response_format=s.ResponseFormat.JSON),
            {"total": 0, "data": []},
        ),
        (
            "bulk_search",
            s.BulkSearchInput(query="absent", response_format=s.ResponseFormat.JSON),
            {"total": 0, "data": []},
        ),
        (
            "match_paper",
            s.PaperMatchInput(query="absent"),
            {"data": []},
        ),
        (
            "get_recommendations",
            s.PaperRecommendationsInput(paper_id=_PAPER, response_format=s.ResponseFormat.JSON),
            {"recommendedPapers": []},
        ),
        (
            "multi_recommend",
            s.MultiRecommendInput(
                positive_paper_ids=[_PAPER], response_format=s.ResponseFormat.JSON
            ),
            {"recommendedPapers": []},
        ),
        (
            "get_paper_authors",
            s.PaperAuthorsInput(paper_id=_PAPER, response_format=s.ResponseFormat.JSON),
            {"data": []},
        ),
        (
            "snippet_search",
            s.SnippetSearchInput(query="absent", response_format=s.ResponseFormat.JSON),
            {"data": []},
        ),
    ],
)
@pytest.mark.asyncio
async def test_genuine_empty_upstream_pages_remain_successful(
    monkeypatch: pytest.MonkeyPatch, tool: str, params: Any, response: Any
) -> None:
    async def good_response(_method: str, _endpoint: str, **_kwargs: Any) -> Any:
        return response

    monkeypatch.setattr(s, "make_request", good_response)
    result = await getattr(s, tool)(params)
    assert result
    assert "Invalid Semantic Scholar" not in result


@pytest.mark.parametrize(
    "score",
    [None, True, "0", math.nan, math.inf, -math.inf, 10**400, {}, []],
)
@pytest.mark.asyncio
async def test_match_score_never_invents_zero_or_accepts_malformed_value(
    monkeypatch: pytest.MonkeyPatch, score: Any
) -> None:
    async def malformed_match(_method: str, _endpoint: str, **_kwargs: Any) -> Any:
        return {"data": [{"paperId": _PAPER, "matchScore": score}]}

    monkeypatch.setattr(s, "make_request", malformed_match)
    with pytest.raises(ToolError, match="missing or invalid matchScore"):
        await s.match_paper(s.PaperMatchInput(query="Synthetic"))


@pytest.mark.asyncio
async def test_missing_match_score_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing_score(_method: str, _endpoint: str, **_kwargs: Any) -> Any:
        return {"data": [{"paperId": _PAPER, "title": "Synthetic match without score"}]}

    monkeypatch.setattr(s, "make_request", missing_score)
    with pytest.raises(ToolError, match="missing or invalid matchScore"):
        await s.match_paper(s.PaperMatchInput(query="Synthetic"))


@pytest.mark.asyncio
async def test_invalid_bulk_continuation_token_is_not_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def malformed_bulk(_method: str, _endpoint: str, **_kwargs: Any) -> Any:
        return {"total": 0, "data": [], "token": ["bad token"]}

    monkeypatch.setattr(s, "make_request", malformed_bulk)
    with pytest.raises(ToolError, match="continuation token"):
        await s.bulk_search(s.BulkSearchInput(query="Synthetic"))

@pytest.mark.parametrize(
    ("tool", "params", "identity_field"),
    [
        ("search_papers", s.PaperSearchInput(query="Synthetic"), "paperId"),
        ("search_authors", s.AuthorSearchInput(query="Synthetic"), "authorId"),
        ("bulk_search", s.BulkSearchInput(query="Synthetic"), "paperId"),
        ("match_paper", s.PaperMatchInput(query="Synthetic"), "paperId"),
    ],
)
@pytest.mark.parametrize("bad_identity", [None, "", "   ", 17, False])
@pytest.mark.asyncio
async def test_unattributable_search_result_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    params: Any,
    identity_field: str,
    bad_identity: Any,
) -> None:
    """A counted search hit must not be displayed without its provider ID."""

    async def missing_identity(_method: str, _endpoint: str, **_kwargs: Any) -> Any:
        return {
            "total": 1,
            "data": [
                {identity_field: bad_identity, "title": "Synthetic", "matchScore": 174.2298}
            ],
        }

    monkeypatch.setattr(s, "make_request", missing_identity)
    with pytest.raises(ToolError, match=f"missing valid {identity_field}"):
        await getattr(s, tool)(params)


@pytest.mark.parametrize(
    ("tool", "params", "identity_field"),
    [
        (
            "search_papers",
            s.PaperSearchInput(query="Synthetic", response_format=s.ResponseFormat.JSON),
            "paperId",
        ),
        (
            "search_authors",
            s.AuthorSearchInput(query="Synthetic", response_format=s.ResponseFormat.JSON),
            "authorId",
        ),
        (
            "bulk_search",
            s.BulkSearchInput(query="Synthetic", response_format=s.ResponseFormat.JSON),
            "paperId",
        ),
    ],
)
@pytest.mark.asyncio
async def test_counted_search_preserves_valid_upstream_identity(
    monkeypatch: pytest.MonkeyPatch, tool: str, params: Any, identity_field: str
) -> None:
    async def valid_response(_method: str, _endpoint: str, **_kwargs: Any) -> Any:
        return {
            "total": 1,
            "data": [{identity_field: "VALID_ID", "title": "Synthetic", "name": "Synthetic"}],
        }

    monkeypatch.setattr(s, "make_request", valid_response)
    result = await getattr(s, tool)(params)
    assert "VALID_ID" in result


@pytest.mark.asyncio
async def test_provider_match_score_above_one_is_not_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The official S2 example uses a matchScore above 100, not a probability."""

    async def valid_match(_method: str, _endpoint: str, **_kwargs: Any) -> Any:
        return {"data": [{"paperId": _PAPER, "title": "Synthetic", "matchScore": 174.2298}]}

    monkeypatch.setattr(s, "make_request", valid_match)
    json_result = await s.match_paper(
        s.PaperMatchInput(query="Synthetic", response_format=s.ResponseFormat.JSON)
    )
    import json

    assert json.loads(json_result)["matchScore"] == 174.2298
    markdown_result = await s.match_paper(s.PaperMatchInput(query="Synthetic"))
    assert "174.2" in markdown_result
