"""Regression tests for independent citation and reference pagination."""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response
from pydantic import ValidationError as PydanticValidationError

from semantic_scholar_mcp.models import PaperDetailsInput, ResponseFormat
from semantic_scholar_mcp.server import SEMANTIC_SCHOLAR_API_BASE, get_paper_details

PAPER_ID = "a" * 40
PAPER_URL = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{PAPER_ID}"
CITATIONS_URL = f"{PAPER_URL}/citations"
REFERENCES_URL = f"{PAPER_URL}/references"


def _respond(url, payload):
    return respx.get(url).mock(return_value=Response(200, json=payload))


@pytest.mark.parametrize("field", ["citations_offset", "references_offset"])
def test_offsets_reject_negative_values(field):
    with pytest.raises(PydanticValidationError):
        PaperDetailsInput(paper_id=PAPER_ID, **{field: -1})


def test_offsets_exposed_in_input_schema():
    props = PaperDetailsInput.model_json_schema()["properties"]
    assert props["citations_offset"]["default"] == 0
    assert props["references_offset"]["default"] == 0
    assert PaperDetailsInput(paper_id=PAPER_ID).citations_offset == 0
    assert PaperDetailsInput(paper_id=PAPER_ID).references_offset == 0


@respx.mock
@pytest.mark.asyncio
async def test_json_paginates_both_directions_independently(reset_all):
    _respond(PAPER_URL, {"paperId": PAPER_ID, "title": "Seed"})
    citing = _respond(
        CITATIONS_URL,
        {"data": [{"citingPaper": {"paperId": "citation-7", "title": "Citing"}}], "next": 9},
    )
    cited = _respond(
        REFERENCES_URL,
        {"data": [{"citedPaper": {"paperId": "reference-13", "title": "Cited"}}], "next": 15},
    )
    params = PaperDetailsInput(
        paper_id=PAPER_ID,
        include_citations=True,
        include_references=True,
        citations_offset=7,
        citations_limit=2,
        references_offset=13,
        references_limit=2,
        response_format=ResponseFormat.JSON,
    )
    result = json.loads(await get_paper_details(params))
    assert result["paper"]["title"] == "Seed"
    assert result["citations"][0]["citingPaper"]["paperId"] == "citation-7"
    assert result["references"][0]["citedPaper"]["paperId"] == "reference-13"
    assert result["citations_page"] == {"offset": 7, "next": 9}
    assert result["references_page"] == {"offset": 13, "next": 15}
    assert citing.calls[0].request.url.params["offset"] == "7"
    assert citing.calls[0].request.url.params["limit"] == "2"
    assert cited.calls[0].request.url.params["offset"] == "13"
    assert cited.calls[0].request.url.params["limit"] == "2"


@respx.mock
@pytest.mark.asyncio
async def test_markdown_indicates_continuation_for_both_directions(reset_all):
    _respond(PAPER_URL, {"paperId": PAPER_ID, "title": "Seed"})
    _respond(CITATIONS_URL, {"data": [{"citingPaper": {"title": "Citing"}}], "next": 4})
    _respond(REFERENCES_URL, {"data": [{"citedPaper": {"title": "Cited"}}], "next": 5})
    params = PaperDetailsInput(paper_id=PAPER_ID, include_citations=True, include_references=True)
    result = await get_paper_details(params)
    assert "Citing" in result
    assert "Cited" in result
    assert "citations_offset=4" in result
    assert "references_offset=5" in result


@respx.mock
@pytest.mark.asyncio
async def test_missing_citation_counts_are_unknown_not_zero(reset_all):
    """Missing upstream citation counts cannot be presented as observed zero."""
    _respond(PAPER_URL, {"paperId": PAPER_ID, "title": "Synthetic Seed"})
    _respond(
        CITATIONS_URL,
        {"data": [{"citingPaper": {"paperId": "c1", "title": "Missing count"}}]},
    )
    _respond(
        REFERENCES_URL,
        {
            "data": [
                {"citedPaper": {"paperId": "r1", "title": "Null count", "citationCount": None}},
                {"citedPaper": {"paperId": "r2", "title": "Reported zero", "citationCount": 0}},
            ]
        },
    )
    params = PaperDetailsInput(paper_id=PAPER_ID, include_citations=True, include_references=True)
    output = await get_paper_details(params)
    assert "**Missing count** () - N/A citations" in output
    assert "**Null count** () - N/A citations" in output
    assert "**Reported zero** () - 0 citations" in output


@respx.mock
@pytest.mark.asyncio
async def test_end_of_pages_does_not_advertise_more(reset_all):
    _respond(PAPER_URL, {"paperId": PAPER_ID, "title": "Seed"})
    _respond(CITATIONS_URL, {"data": [], "next": None})
    _respond(REFERENCES_URL, {"data": []})
    params = PaperDetailsInput(paper_id=PAPER_ID, include_citations=True, include_references=True)
    result = await get_paper_details(params)
    assert "citations_offset=" not in result
    assert "references_offset=" not in result


@respx.mock
@pytest.mark.asyncio
async def test_non_object_graph_response_is_explicit_upstream_failure(reset_all):
    from mcp.server.fastmcp.exceptions import ToolError

    _respond(PAPER_URL, {"paperId": PAPER_ID, "title": "Seed"})
    _respond(CITATIONS_URL, [])
    params = PaperDetailsInput(
        paper_id=PAPER_ID, include_citations=True, response_format=ResponseFormat.JSON
    )
    with pytest.raises(ToolError, match="paper/citations"):
        await get_paper_details(params)


@respx.mock
@pytest.mark.asyncio
async def test_failed_direction_cancels_sibling_request(reset_all):
    """A failing citations fetch must not leave the references fetch in flight.

    ``asyncio.gather`` propagates the first exception but keeps sibling tasks
    running: before the fix the references request was still dispatched after
    the tool had already raised, burning a rate-limit slot for a result nobody
    could observe.
    """
    import asyncio

    from mcp.server.fastmcp.exceptions import ToolError

    from semantic_scholar_mcp import client as _client

    _respond(PAPER_URL, {"paperId": PAPER_ID, "title": "Seed"})
    respx.get(CITATIONS_URL).mock(return_value=Response(404))
    refs = _respond(REFERENCES_URL, {"data": []})
    params = PaperDetailsInput(paper_id=PAPER_ID, include_citations=True, include_references=True)

    with pytest.raises(ToolError, match="Not found"):
        await get_paper_details(params)

    # No orphaned make_request task may survive the tool's failure...
    leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert not any("make_request" in repr(t) for t in leftovers), leftovers
    # ...and the sibling request must never reach the wire, even after the
    # rate-limit interval that previously let it slip out.
    await asyncio.sleep(_client.get_min_request_interval(False) + 0.2)
    assert not refs.called


@pytest.mark.asyncio
async def test_gather_or_cancel_preserves_order_and_cancels_on_failure():
    import asyncio

    from semantic_scholar_mcp.server import _gather_or_cancel

    async def ok(value, delay=0.0):
        await asyncio.sleep(delay)
        return value

    assert await _gather_or_cancel([ok("a", 0.01), ok("b")]) == ["a", "b"]

    cancelled = asyncio.Event()

    async def slow():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await _gather_or_cancel([slow(), boom()])
    assert cancelled.is_set()
