"""Integration tests for optional DOI mode against the actual MCP tool function.

These tests require the full s2-mcp-server repository and its dev extras.
All upstream responses are mocked; no outbound API request is made.
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from semantic_scholar_mcp import server
from semantic_scholar_mcp.models import CitationExportInput

PAPER_ID = "a" * 40
DOI = "10.1145/3065386"
RAW = "@article{krizhevsky2012,\n title={ImageNet Classification},\n year={2012}\n}"


@pytest.fixture
def upstream(monkeypatch):
    calls: list[dict] = []
    paper = {
        "paperId": PAPER_ID,
        "title": "ImageNet Classification",
        "externalIds": {"DOI": DOI},
        "citationStyles": {"bibtex": RAW},
    }

    async def request(method, endpoint, *, params, api_key=None):
        calls.append({"method": method, "endpoint": endpoint, "params": params, "api_key": api_key})
        return paper

    monkeypatch.setattr(server, "cache_get", lambda _: None)
    monkeypatch.setattr(server, "make_request", request)
    return paper, calls


@pytest.mark.asyncio
async def test_verbatim_default_remains_byte_for_byte(upstream):
    paper, calls = upstream
    paper["externalIds"] = {"DOI": "10.5555/different"}
    assert await server.export_citation(CitationExportInput(paper_id=PAPER_ID)) == RAW
    assert len(calls) == 1
    assert calls[0]["params"]["fields"] == "paperId,corpusId,externalIds,url,title,citationStyles"


@pytest.mark.asyncio
async def test_opt_in_adds_only_structured_doi(upstream):
    _, calls = upstream
    output = await server.export_citation(
        CitationExportInput(paper_id=PAPER_ID, format="bibtex+doi", api_key="staging-test-key")
    )
    assert "doi = {10.1145/3065386}" in output
    assert "year={2012}" in output  # deliberately NOT silently 'corrected' to 2017
    assert len(calls) == 1
    assert calls[0]["api_key"] == "staging-test-key"


@pytest.mark.asyncio
async def test_opt_in_does_not_rewrite_matching_existing_doi(upstream):
    paper, _ = upstream
    paper["citationStyles"]["bibtex"] = RAW[:-1] + ", doi={10.1145/3065386}\n}"
    actual = await server.export_citation(
        CitationExportInput(paper_id=PAPER_ID, format="bibtex+doi")
    )
    assert actual == paper["citationStyles"]["bibtex"]


@pytest.mark.asyncio
async def test_opt_in_rejects_conflicting_existing_doi(upstream):
    paper, _ = upstream
    paper["citationStyles"]["bibtex"] = RAW[:-1] + ", doi={10.5555/wrong}\n}"
    with pytest.raises(ToolError, match="conflicts"):
        await server.export_citation(CitationExportInput(paper_id=PAPER_ID, format="bibtex+doi"))


@pytest.mark.asyncio
async def test_opt_in_rejects_missing_structured_doi(upstream):
    paper, _ = upstream
    paper["externalIds"] = {"ArXiv": "1201.00001"}
    with pytest.raises(ToolError, match="Missing or invalid"):
        await server.export_citation(CitationExportInput(paper_id=PAPER_ID, format="bibtex+doi"))


@pytest.mark.asyncio
async def test_opt_in_rejects_missing_external_ids_map(upstream):
    paper, _ = upstream
    paper["externalIds"] = None
    with pytest.raises(ToolError, match="Missing or invalid"):
        await server.export_citation(CitationExportInput(paper_id=PAPER_ID, format="bibtex+doi"))


@pytest.mark.asyncio
async def test_opt_in_rejects_malformed_upstream_bibtex(upstream):
    paper, _ = upstream
    paper["citationStyles"]["bibtex"] = "@article{x, title={never closed}"
    with pytest.raises(ToolError, match="Unclosed"):
        await server.export_citation(CitationExportInput(paper_id=PAPER_ID, format="bibtex+doi"))


@pytest.mark.asyncio
async def test_unknown_format_still_errors(upstream):
    _, calls = upstream
    with pytest.raises(ToolError, match="Unsupported citation format"):
        await server.export_citation(CitationExportInput(paper_id=PAPER_ID, format="apa"))
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_bibtex_missing_still_errors(upstream):
    paper, _ = upstream
    paper["citationStyles"] = {}
    with pytest.raises(ToolError, match="No BibTeX"):
        await server.export_citation(CitationExportInput(paper_id=PAPER_ID, format="bibtex+doi"))


@pytest.mark.asyncio
async def test_requested_doi_must_match_returned_alias(upstream):
    paper, _ = upstream
    paper["externalIds"]["DOI"] = "10.5555/not-requested"
    with pytest.raises(ToolError, match="identity mismatch"):
        await server.export_citation(
            CitationExportInput(paper_id="DOI:" + DOI, format="bibtex+doi")
        )
