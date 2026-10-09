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
    respx.get(PAPER_URL).mock(
        return_value=Response(200, json={"paperId": PAPER_ID, "title": "Seed"})
    )
    citing = respx.get(CITATIONS_URL).mock(
        return_value=Response(
            200,
            json={
                "data": [{"citingPaper": {"paperId": "citation-7", "title": "Citing"}}],
                "next": 9,
            },
        )
    )
    cited = respx.get(REFERENCES_URL).mock(
        return_value=Response(
            200,
            json={
                "data": [{"citedPaper": {"paperId": "reference-13", "title": "Referenced"}}],
                "next": 15,
            },
        )
    )
    result = json.loads(
        await get_paper_details(
            PaperDetailsInput(
                paper_id=PAPER_ID,
                include_citations=True,
                include_references=True,
                citations_offset=7,
                citations_limit=2,
                references_offset=13,
                references_limit=2,
                response_format=ResponseFormat.JSON,
            )
        )
    )
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
    respx.get(PAPER_URL).mock(
        return_value=Response(200, json={"paperId": PAPER_ID, "title": "Seed"})
    )
    respx.get(CITATIONS_URL).mock(
        return_value=Response(
            200, json={"data": [{"citingPaper": {"title": "Citing"}}], "next": 4}
        )
    )
    respx.get(REFERENCES_URL).mock(
        return_value=Response(
            200, json={"data": [{"citedPaper": {"title": "Referenced"}}], "next": 5}
        )
    )
    text = await get_paper_details(
        PaperDetailsInput(
            paper_id=PAPER_ID, include_citations=True, include_references=True
        )
    )
    assert "Citing" in text
    assert "Referenced" in text
    assert "citations_offset=4" in text
    assert "references_offset=5" in text


@respx.mock
@pytest.mark.asyncio
async def test_end_of_pages_does_not_advertise_more(reset_all):
    respx.get(PAPER_URL).mock(
        return_value=Response(200, json={"paperId": PAPER_ID, "title": "Seed"})
    )
    respx.get(CITATIONS_URL).mock(
        return_value=Response(200, json={"data": [], "next": None})
    )
    respx.get(REFERENCES_URL).mock(
        return_value=Response(200, json={"data": []})
    )
    text = await get_paper_details(
        PaperDetailsInput(
            paper_id=PAPER_ID, include_citations=True, include_references=True
        )
    )
    assert "citations_offset=" not in text
    assert "references_offset=" not in text


@respx.mock
@pytest.mark.asyncio
async def test_non_object_graph_response_retains_legacy_empty_array(reset_all):
    respx.get(PAPER_URL).mock(
        return_value=Response(200, json={"paperId": PAPER_ID, "title": "Seed"})
    )
    respx.get(CITATIONS_URL).mock(return_value=Response(200, json=[]))
    result = json.loads(
        await get_paper_details(
            PaperDetailsInput(
                paper_id=PAPER_ID,
                include_citations=True,
                response_format=ResponseFormat.JSON,
            )
        )
    )
    assert result["citations"] == []
    assert result["citations_page"] == {"offset": 0, "next": None}
    assert "references_page" not in result
