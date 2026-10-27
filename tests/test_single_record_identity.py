"""Source-identity contracts for individually retrieved S2 papers/authors.

All examples are synthetic. No research paper, author, or biomedical inference
is asserted by these fixtures.
"""

from __future__ import annotations

import pytest
import respx
from httpx import Response
from mcp.server.fastmcp.exceptions import ToolError

from semantic_scholar_mcp.server import (
    SEMANTIC_SCHOLAR_API_BASE,
    AuthorDetailsInput,
    CitationExportInput,
    PaperDetailsInput,
    _require_author_record_identity,
    _require_paper_record_identity,
    cache_get,
    cache_set,
    export_citation,
    get_author_details,
    get_paper_details,
)

_PAPER = "a" * 40
_OTHER_PAPER = "b" * 40


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        {},
        {"paperId": None},
        {"paperId": 42},
        {"paperId": ""},
        {"paperId": "   "},
    ],
)
def test_missing_paper_identity_cannot_be_treated_as_retrieved(record: object) -> None:
    with pytest.raises(ToolError, match="Unexpected"):
        _require_paper_record_identity(record, _PAPER)


def test_wrong_canonical_paper_id_is_not_attributed_to_requested_record() -> None:
    with pytest.raises(ToolError, match="Paper identity mismatch"):
        _require_paper_record_identity({"paperId": _OTHER_PAPER}, _PAPER)


@pytest.mark.parametrize(
    ("requested", "returned"),
    [
        (_PAPER, _PAPER),
        (_PAPER.upper(), _PAPER),
    ],
)
def test_paper_identity_accepts_canonical_ids(requested: str, returned: str) -> None:
    paper = {"paperId": returned, "title": "Synthetic paper"}
    assert _require_paper_record_identity(paper, requested) is paper


@pytest.mark.parametrize(
    ("alias", "evidence"),
    [
        ("DOI:10.1234/example", {"externalIds": {"DOI": "10.1234/EXAMPLE"}}),
        ("ARXIV:2106.15928", {"externalIds": {"ArXiv": "2106.15928"}}),
        ("ARXIV:2106.15928v2", {"externalIds": {"ArXiv": "2106.15928"}}),
        ("ARXIV:hep-th/0406185v3", {"externalIds": {"ArXiv": "hep-th/0406185"}}),
        ("PMID:19872477", {"externalIds": {"PubMed": "19872477"}}),
        ("ACL:W12-3903", {"externalIds": {"ACL": "W12-3903"}}),
        ("CorpusId:00215416146", {"corpusId": 215416146}),
        ("CorpusId:215416146", {"externalIds": {"CorpusId": "215416146"}}),
        (
            "URL:https://arxiv.org/abs/2106.15928v1",
            {"externalIds": {"ArXiv": "2106.15928"}},
        ),
        (
            "URL:https://arxiv.org/abs/hep-th/0406185",
            {"externalIds": {"ArXiv": "hep-th/0406185"}},
        ),
        ("URL:https://doi.org/10.1234/example", {"externalIds": {"DOI": "10.1234/example"}}),
        (
            "URL:https://pubmed.ncbi.nlm.nih.gov/19872477/",
            {"externalIds": {"PubMed": "19872477"}},
        ),
        (
            "URL:https://www.semanticscholar.org/paper/" + _PAPER,
            {"url": "https://www.semanticscholar.org/paper/" + _PAPER},
        ),
        (
            "URL:https://www.aclweb.org/anthology/W12-3903",
            {"externalIds": {"ACL": "W12-3903"}},
        ),
        (
            "URL:https://aclweb.org/anthology/P19-1285",
            {"externalIds": {"ACL": "P19-1285"}},
        ),
        (
            "URL:https://dl.acm.org/doi/10.1145/12345",
            {"externalIds": {"DOI": "10.1145/12345"}},
        ),
        (
            "URL:https://www.biorxiv.org/content/10.1101/2020.01.01.12345v2",
            {"externalIds": {"DOI": "10.1101/2020.01.01.12345"}},
        ),
        (
            "URL:https://www.biorxiv.org/content/10.1101/2023.01.25.525531v1.full",
            {"externalIds": {"DOI": "10.1101/2023.01.25.525531"}},
        ),
        (
            "URL:https://www.biorxiv.org/content/10.1101/2023.01.25.525531v1.full.pdf",
            {"externalIds": {"DOI": "10.1101/2023.01.25.525531"}},
        ),
    ],
)
def test_documented_paper_alias_is_bound_to_provider_evidence(
    alias: str, evidence: dict[str, object]
) -> None:
    paper: dict[str, object] = {"paperId": _PAPER, **evidence}
    assert _require_paper_record_identity(paper, alias) is paper


@pytest.mark.parametrize(
    "alias",
    [
        "DOI:10.1234/other",
        "ARXIV:2106.15929",
        "PMID:19872478",
        "ACL:W12-3904",
        "CorpusId:215416147",
        "URL:https://arxiv.org/abs/2106.15929",
        "URL:https://biorxiv.org/content/10.1101/elsewhere",
        "URL:https://example.com/unrelated",
        "ARXIV:2106.15929v2",
        "URL:https://arxiv.org/abs/hep-th/0406186",
        "URL:https://aclweb.org/anthology/P19-1285",
        "URL:https://www.biorxiv.org/content/10.1101/2023.01.25.525531v1.full.pdf",
    ],
)
def test_wrong_or_unverifiable_alias_is_rejected(alias: str) -> None:
    paper = {
        "paperId": _PAPER,
        "externalIds": {
            "DOI": "10.1234/example",
            "ArXiv": "2106.15928",
            "PubMed": "19872477",
            "ACL": "W12-3903",
        },
        "corpusId": 215416146,
        "url": "https://www.semanticscholar.org/paper/" + _PAPER,
    }
    with pytest.raises(ToolError, match="Paper identity mismatch"):
        _require_paper_record_identity(paper, alias)


def test_s2_url_with_wrong_embedded_id_rejected_even_if_url_repeated() -> None:
    requested_url = "https://www.semanticscholar.org/paper/" + _PAPER
    wrong_record = {"paperId": _OTHER_PAPER, "url": requested_url}
    with pytest.raises(ToolError, match="Paper identity mismatch"):
        _require_paper_record_identity(wrong_record, "URL:" + requested_url)


@pytest.mark.parametrize(
    ("alias", "record"),
    [
        ("UNSUPPORTED:unknown", {"paperId": _PAPER}),
        ("URL:https://[malformed", {"paperId": _PAPER}),
        (
            "URL:http://doi.org/10.1234/example",
            {"paperId": _PAPER, "externalIds": {"DOI": "10.1234/example"}},
        ),
        ("URL:https://www.semanticscholar.org/other/" + _PAPER, {"paperId": _PAPER}),
        ("URL:https://www.semanticscholar.org/paper/" + _OTHER_PAPER, {"paperId": _PAPER}),
        ("CorpusId:abc", {"paperId": _PAPER, "corpusId": 123}),
        ("CorpusId:123", {"paperId": _PAPER, "corpusId": True}),
        (
            "DOI:10.1234/example",
            {"paperId": _PAPER, "externalIds": {"DOI": None}},
        ),
    ],
)
def test_unverifiable_aliases_fail_closed(alias: str, record: dict[str, object]) -> None:
    """Defensive verifier rejects unsupported, malformed and mismatched IDs."""
    with pytest.raises(ToolError, match="Paper identity mismatch"):
        _require_paper_record_identity(record, alias)


@pytest.mark.asyncio
async def test_poisoned_doi_cache_is_rejected_before_return(reset_all: object) -> None:
    alias = "DOI:10.1234/wanted"
    cache_set(
        f"paper:{alias}",
        {"paperId": _PAPER, "externalIds": {"DOI": "10.1234/wrong"}},
    )
    with pytest.raises(ToolError, match="Paper identity mismatch"):
        await get_paper_details(PaperDetailsInput(paper_id=alias))


@respx.mock
@pytest.mark.asyncio
async def test_wrong_doi_http_200_fails_and_is_not_cached(reset_all: object) -> None:
    alias = "DOI:10.1234/wanted"
    respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{alias}").mock(
        return_value=Response(
            200, json={"paperId": _PAPER, "externalIds": {"DOI": "10.1234/wrong"}}
        )
    )
    with pytest.raises(ToolError, match="Paper identity mismatch"):
        await get_paper_details(PaperDetailsInput(paper_id=alias))
    assert cache_get(f"paper:{alias}") is None


@respx.mock
@pytest.mark.asyncio
async def test_citation_export_checks_doi_identity_and_requests_evidence(
    reset_all: object,
) -> None:
    alias = "DOI:10.1234/wanted"
    route = respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{alias}").mock(
        return_value=Response(
            200,
            json={
                "paperId": _PAPER,
                "externalIds": {"DOI": "10.1234/wrong"},
                "citationStyles": {"bibtex": "@article{wrong}"},
            },
        )
    )
    with pytest.raises(ToolError, match="Paper identity mismatch"):
        await export_citation(CitationExportInput(paper_id=alias))
    assert "externalIds" in route.calls[0].request.url.params["fields"]
    assert "corpusId" in route.calls[0].request.url.params["fields"]


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        {},
        {"authorId": None},
        {"authorId": 43},
        {"authorId": ""},
        {"authorId": "fullwidth-\uff11\uff12\uff13"},
        {"authorId": "\uff11\uff12\uff13"},
        {"authorId": "124"},
    ],
)
def test_invalid_or_wrong_author_identity_rejected(record: object) -> None:
    with pytest.raises(ToolError, match=r"(Unexpected|identity mismatch)"):
        _require_author_record_identity(record, "123")


@pytest.mark.parametrize(
    ("requested", "returned"),
    [("123", "123"), ("00123", "123"), ("123", "00123"), ("0", "00")],
)
def test_author_numeric_alias_retains_true_identity(requested: str, returned: str) -> None:
    author = {"authorId": returned, "name": "Synthetic researcher"}
    assert _require_author_record_identity(author, requested) is author


@respx.mock
@pytest.mark.asyncio
async def test_wrong_paper_http_200_fails_and_is_not_cached(reset_all: object) -> None:
    respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{_PAPER}").mock(
        return_value=Response(200, json={"paperId": _OTHER_PAPER, "title": "Wrong paper"})
    )
    with pytest.raises(ToolError, match="Paper identity mismatch"):
        await get_paper_details(PaperDetailsInput(paper_id=_PAPER))
    assert cache_get(f"paper:{_PAPER}") is None


@respx.mock
@pytest.mark.asyncio
async def test_wrong_author_http_200_fails_and_is_not_cached(reset_all: object) -> None:
    respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/author/123").mock(
        return_value=Response(200, json={"authorId": "124", "name": "Wrong person"})
    )
    with pytest.raises(ToolError, match="Author identity mismatch"):
        await get_author_details(AuthorDetailsInput(author_id="123", include_papers=False))
    assert cache_get("author:123") is None


@pytest.mark.asyncio
async def test_poisoned_paper_cache_does_not_bypass_identity_check(reset_all: object) -> None:
    cache_set(f"paper:{_PAPER}", {"paperId": _OTHER_PAPER, "title": "Wrong paper"})
    with pytest.raises(ToolError, match="Paper identity mismatch"):
        await get_paper_details(PaperDetailsInput(paper_id=_PAPER))


@pytest.mark.asyncio
async def test_poisoned_author_cache_does_not_bypass_identity_check(reset_all: object) -> None:
    cache_set("author:123", {"authorId": "124", "name": "Wrong researcher"})
    with pytest.raises(ToolError, match="Author identity mismatch"):
        await get_author_details(AuthorDetailsInput(author_id="123", include_papers=False))


@respx.mock
@pytest.mark.asyncio
async def test_citation_export_rejects_wrong_paper_response(reset_all: object) -> None:
    respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{_PAPER}").mock(
        return_value=Response(
            200,
            json={
                "paperId": _OTHER_PAPER,
                "title": "Wrong paper",
                "citationStyles": {"bibtex": "@article{wrong, title={Wrong}}"},
            },
        )
    )
    with pytest.raises(ToolError, match="Paper identity mismatch"):
        await export_citation(CitationExportInput(paper_id=_PAPER))


@respx.mock
@pytest.mark.asyncio
async def test_citation_export_requires_provider_paper_id(reset_all: object) -> None:
    respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{_PAPER}").mock(
        return_value=Response(
            200, json={"citationStyles": {"bibtex": "@article{missing, title={Missing}}"}}
        )
    )
    with pytest.raises(ToolError, match="missing paperId"):
        await export_citation(CitationExportInput(paper_id=_PAPER))
