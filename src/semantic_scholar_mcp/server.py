"""Semantic Scholar MCP Server — FastMCP entry point.

Architecture (see ``README.md`` ▸ Architecture):
    ┌────────────┐   stdio   ┌──────────┐   models    ┌──────────┐
    │ MCP client │ ────────▶│ FastMCP  │ ──────────▶│  client  │
    │ (Claude…)  │           │ (server) │  (Pydantic) │  (httpx) │
    └────────────┘           └──────────┘             └──────────┘
                                  │                        │
                                  │ TTL cache              │ rate-limit + retry
                                  │ formatters             │ Semantic Scholar
                                  ▼                        ▼  Graph / Recs API

Concerns live in focused modules; this file is the public surface — the 14
``@mcp.tool`` registrations, the FastMCP lifespan, and the ``main()`` entry —
plus back-compat re-exports so ``from semantic_scholar_mcp.server import X``
keeps working for downstream callers and the existing test suite.

Author: Santiago Maniches (ORCID: 0009-0005-6480-1987)
Organization: TOPOLOGICA LLC — https://topologica.ai
License: MIT
Repository: https://github.com/smaniches/semantic-scholar-mcp
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, get_type_hints

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ContentBlock, ToolAnnotations
from pydantic import ValidationError as PydanticValidationError

from . import __version__

# Re-exports for back-compat: tests and downstream callers import these names
# from ``semantic_scholar_mcp.server``. The leading-underscore aliases preserve
# the old "private" surface while the implementations live in focused modules.
from .cache import _MAX_SIZE as _CACHE_MAX_SIZE  # noqa: F401
from .cache import _TTL as _CACHE_TTL  # noqa: F401
from .cache import _cache, cache_clear, cache_get, cache_set  # noqa: F401
from .client import (
    MAX_RETRIES,  # noqa: F401
    RECOMMENDATIONS_BASE,
    RETRY_BACKOFF_BASE,  # noqa: F401
    SEMANTIC_SCHOLAR_API_BASE,  # noqa: F401
    SEMANTIC_SCHOLAR_API_KEY,
    _execute_request_with_retry,  # noqa: F401
    close_client,
    get_client,
    get_headers,
    handle_error,
    make_request,
)
from .errors import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    SemanticScholarError,
    ServerError,
    ValidationError,
)
from .formatters import format_author_markdown, format_paper_markdown
from .logging_config import get_logger
from .models import (
    AUTHOR_FIELDS,
    PAPER_BULK_SEARCH_FIELDS,
    PAPER_DETAIL_FIELDS,
    PAPER_SEARCH_FIELDS,
    PAPER_SEARCH_FIELDS_LITE,
    AuthorBatchInput,
    AuthorDetailsInput,
    AuthorSearchInput,
    BulkPaperInput,
    BulkSearchInput,
    CitationExportInput,
    MultiRecommendInput,
    PaperAuthorsInput,
    PaperDetailsInput,
    PaperMatchInput,
    PaperRecommendationsInput,
    PaperSearchInput,
    ResponseFormat,
    SnippetSearchInput,
)
from .validators import is_valid_paper_id, validate_paper_id

# Back-compat aliases so legacy imports
# (``from semantic_scholar_mcp.server import _make_request``) keep resolving.
_make_request = make_request
_get_headers = get_headers
_get_client = get_client
_handle_error = handle_error
_validate_paper_id = validate_paper_id
_is_valid_paper_id = is_valid_paper_id
_format_paper_markdown = format_paper_markdown
_format_author_markdown = format_author_markdown
_cache_get = cache_get
_cache_set = cache_set
_cache_clear = cache_clear

logger = get_logger()


@asynccontextmanager
async def _lifespan(app: FastMCP) -> AsyncIterator[None]:
    """Manage the shared HTTP client and cache across the server lifetime."""
    logger.info("Starting semantic-scholar-mcp v%s", __version__)
    try:
        yield
    finally:
        cache_clear()
        await close_client()
        logger.info("Server shutdown complete")


def _get_accepted_params(tool_name: str, tool_manager: Any) -> list[str]:
    """Extract accepted parameter names from a tool's input Pydantic model."""
    tool = tool_manager.get_tool(tool_name)
    if not tool:
        return []
    try:
        hints = get_type_hints(tool.fn)
    except Exception:
        return []
    for name, param_type in hints.items():
        if name == "return":
            continue
        if isinstance(param_type, type) and hasattr(param_type, "model_fields"):
            return list(param_type.model_fields.keys())
    return []


def _friendly_validation_message(
    tool_name: str,
    exc: PydanticValidationError,
    tool_manager: Any,
) -> str:
    """Convert a raw pydantic ValidationError into a user-friendly message."""
    errors = exc.errors()

    extra_fields: list[str] = []
    other_messages: list[str] = []

    for err in errors:
        loc_parts = [str(p) for p in err.get("loc", ()) if p != "params"]
        loc_str = ".".join(loc_parts) if loc_parts else "input"
        if err.get("type") == "extra_forbidden":
            extra_fields.append(loc_str)
        else:
            other_messages.append(f"{loc_str}: {err['msg']}")

    accepted = _get_accepted_params(tool_name, tool_manager)
    accepted_str = f" Accepted parameters: {accepted}." if accepted else ""

    if extra_fields and not other_messages:
        parts = [f"Invalid parameter(s): {extra_fields}."]
        if "fields" in extra_fields:
            parts.append(
                "This tool does not expose a 'fields' parameter;"
                " field selection is managed internally."
            )
        parts.append(accepted_str.strip())
        return " ".join(p for p in parts if p)

    if other_messages and not extra_fields:
        return f"Validation error: {'; '.join(other_messages)}.{accepted_str}"

    all_msgs: list[str] = []
    if extra_fields:
        all_msgs.append(f"Invalid parameter(s): {extra_fields}")
        if "fields" in extra_fields:
            all_msgs.append(
                "This tool does not expose a 'fields' parameter;"
                " field selection is managed internally"
            )
    all_msgs.extend(other_messages)
    return f"Validation error: {'; '.join(all_msgs)}.{accepted_str}"


class _S2FastMCP(FastMCP):
    """FastMCP subclass that intercepts pydantic input-validation errors."""

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        try:
            return await super().call_tool(name, arguments)
        except ToolError as e:
            if isinstance(e.__cause__, PydanticValidationError):
                raise ToolError(
                    _friendly_validation_message(name, e.__cause__, self._tool_manager)
                ) from e.__cause__
            raise


mcp = _S2FastMCP(
    "semantic_scholar_mcp",
    instructions="""
    Semantic Scholar MCP Server - Access 200M+ academic papers.
    Created by Santiago Maniches (ORCID: 0009-0005-6480-1987)
    TOPOLOGICA LLC - https://topologica.ai

    Supports DOI, ArXiv, PubMed, ACL, and Semantic Scholar IDs.
    """,
    lifespan=_lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool(
    name="semantic_scholar_search_papers",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def search_papers(params: PaperSearchInput) -> str:
    """Search for academic papers.

    Supports boolean operators (AND, OR, NOT), phrase search with quotes.
    """
    logger.info("Searching: %s", params.query)

    api_params: dict[str, Any] = {
        "query": params.query,
        "offset": params.offset,
        "limit": params.limit,
        "fields": ",".join(PAPER_SEARCH_FIELDS),
    }
    if params.year:
        api_params["year"] = params.year
    if params.fields_of_study:
        api_params["fieldsOfStudy"] = ",".join(params.fields_of_study)
    if params.publication_types:
        api_params["publicationTypes"] = ",".join(params.publication_types)
    if params.open_access_only:
        api_params["openAccessPdf"] = ""
    if params.min_citation_count:
        api_params["minCitationCount"] = params.min_citation_count

    try:
        response = await make_request(
            "GET", "paper/search", params=api_params, api_key=params.api_key
        )
        total = response.get("total", 0) if isinstance(response, dict) else 0
        papers = response.get("data", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"query": params.query, "total": total, "papers": papers}, indent=2)

    lines = [
        f'## Search Results: "{params.query}"',
        f"**Found:** {total} papers (showing {params.offset + 1}-{params.offset + len(papers)})",
        "",
    ]
    for paper in papers:
        lines.append(format_paper_markdown(paper))
    if total > params.offset + len(papers):
        lines.append(f"*Use offset={params.offset + params.limit} to see more results*")
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_get_paper",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def get_paper_details(params: PaperDetailsInput) -> str:
    """Get paper details. Accepts: S2 ID, DOI:xxx, ARXIV:xxx, PMID:xxx, CorpusId:xxx"""
    logger.info("Getting paper: %s", params.paper_id)

    try:
        validate_paper_id(params.paper_id)

        cache_key = f"paper:{params.paper_id}"
        paper = cache_get(cache_key)
        if paper is None:
            paper = await make_request(
                "GET",
                f"paper/{params.paper_id}",
                params={"fields": ",".join(PAPER_DETAIL_FIELDS)},
                api_key=params.api_key,
            )
            if isinstance(paper, dict):
                cache_set(cache_key, paper)

        if not isinstance(paper, dict):
            raise ToolError("Unexpected response format from Semantic Scholar API")
        result: dict[str, Any] = {"paper": paper}

        # Fetch citations and references in parallel when both requested.
        sub_tasks: dict[str, Any] = {}
        if params.include_citations:
            sub_tasks["citations"] = make_request(
                "GET",
                f"paper/{params.paper_id}/citations",
                params={
                    "fields": ",".join(PAPER_SEARCH_FIELDS_LITE),
                    "limit": params.citations_limit,
                },
                api_key=params.api_key,
            )
        if params.include_references:
            sub_tasks["references"] = make_request(
                "GET",
                f"paper/{params.paper_id}/references",
                params={
                    "fields": ",".join(PAPER_SEARCH_FIELDS_LITE),
                    "limit": params.references_limit,
                },
                api_key=params.api_key,
            )
        if sub_tasks:
            keys = list(sub_tasks.keys())
            responses = await asyncio.gather(*sub_tasks.values())
            for key, resp in zip(keys, responses):
                result[key] = resp.get("data", []) if isinstance(resp, dict) else []
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    lines = ["## Paper Details", "", format_paper_markdown(paper)]
    if result.get("citations"):
        lines.extend(["---", f"### Citing Papers ({len(result['citations'])} shown)", ""])
        for c in result["citations"]:
            p = c.get("citingPaper", {})
            if p:
                lines.append(
                    f"- **{p.get('title', '?')}** ({p.get('year', '')}) "
                    f"- {p.get('citationCount', 0)} citations"
                )
    if result.get("references"):
        lines.extend(["---", f"### References ({len(result['references'])} shown)", ""])
        for r in result["references"]:
            p = r.get("citedPaper", {})
            if p:
                lines.append(
                    f"- **{p.get('title', '?')}** ({p.get('year', '')}) "
                    f"- {p.get('citationCount', 0)} citations"
                )
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_search_authors",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def search_authors(params: AuthorSearchInput) -> str:
    """Search for academic authors by name."""
    logger.info("Searching authors: %s", params.query)

    try:
        response = await make_request(
            "GET",
            "author/search",
            params={
                "query": params.query,
                "offset": params.offset,
                "limit": params.limit,
                "fields": ",".join(AUTHOR_FIELDS),
            },
            api_key=params.api_key,
        )
        total = response.get("total", 0) if isinstance(response, dict) else 0
        authors = response.get("data", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"query": params.query, "total": total, "authors": authors}, indent=2)

    lines = [f'## Author Search: "{params.query}"', f"**Found:** {total} authors", ""]
    for author in authors:
        lines.append(format_author_markdown(author))
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_get_author",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def get_author_details(params: AuthorDetailsInput) -> str:
    """Get author profile with optional publications list."""
    logger.info("Getting author: %s", params.author_id)

    try:
        cache_key = f"author:{params.author_id}"
        author = cache_get(cache_key)
        if author is None:
            author = await make_request(
                "GET",
                f"author/{params.author_id}",
                params={"fields": ",".join(AUTHOR_FIELDS)},
                api_key=params.api_key,
            )
            if isinstance(author, dict):
                cache_set(cache_key, author)

        if not isinstance(author, dict):
            raise ToolError("Unexpected response format from Semantic Scholar API")
        result: dict[str, Any] = {"author": author}

        if params.include_papers:
            papers = await make_request(
                "GET",
                f"author/{params.author_id}/papers",
                params={"fields": ",".join(PAPER_SEARCH_FIELDS_LITE), "limit": params.papers_limit},
                api_key=params.api_key,
            )
            result["papers"] = papers.get("data", []) if isinstance(papers, dict) else []
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    lines = ["## Author Profile", "", format_author_markdown(author)]
    if result.get("papers"):
        lines.extend(["---", f"### Publications ({len(result['papers'])} shown)", ""])
        for p in result["papers"]:
            lines.append(
                f"- **{p.get('title', '?')}** ({p.get('year', '')}) "
                f"- {p.get('citationCount', 0)} citations"
            )
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_recommendations",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def get_recommendations(params: PaperRecommendationsInput) -> str:
    """Get paper recommendations based on a seed paper."""
    logger.info("Recommendations for: %s", params.paper_id)

    try:
        validate_paper_id(params.paper_id)
        response = await make_request(
            "GET",
            f"papers/forpaper/{params.paper_id}",
            params={
                "fields": ",".join(PAPER_SEARCH_FIELDS_LITE),
                "limit": params.limit,
                "from": params.from_pool,
            },
            api_key=params.api_key,
            base_url=RECOMMENDATIONS_BASE,
        )
        papers = response.get("recommendedPapers", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"seed": params.paper_id, "recommendations": papers}, indent=2)

    lines = ["## Recommendations", f"**Seed:** {params.paper_id}", f"**Found:** {len(papers)}", ""]
    for paper in papers:
        lines.append(format_paper_markdown(paper))
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_bulk_papers",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def get_bulk_papers(params: BulkPaperInput) -> str:
    """Retrieve multiple papers in a single request (max 500)."""
    logger.info("Bulk retrieval: %d papers", len(params.paper_ids))

    invalid_ids = [pid for pid in params.paper_ids if not is_valid_paper_id(pid)]
    if invalid_ids:
        msg = f"Invalid paper ID format(s): {', '.join(invalid_ids[:10])}"
        if len(invalid_ids) > 10:
            msg += f" ... +{len(invalid_ids) - 10} more"
        raise ToolError(msg)

    try:
        response = await make_request(
            "POST",
            "paper/batch",
            params={"fields": ",".join(PAPER_SEARCH_FIELDS)},
            json_body={"ids": params.paper_ids},
            api_key=params.api_key,
        )
        papers = response if isinstance(response, list) else response.get("data", [])
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    # Null entries == not-found IDs; track and surface them so users can react.
    succeeded = [p for p in papers if p]
    failed_indices = [i for i, p in enumerate(papers) if not p]
    failed_ids = [params.paper_ids[i] for i in failed_indices if i < len(params.paper_ids)]

    if failed_ids:
        logger.warning("Bulk retrieval: %d papers not found: %s", len(failed_ids), failed_ids[:10])

    if params.response_format == ResponseFormat.JSON:
        result = {
            "requested": len(params.paper_ids),
            "retrieved": len(succeeded),
            "papers": succeeded,
        }
        if failed_ids:
            result["not_found"] = failed_ids
        return json.dumps(result, indent=2)

    lines = [
        "## Bulk Retrieval",
        f"**Requested:** {len(params.paper_ids)} | **Retrieved:** {len(succeeded)}",
        "",
    ]
    if failed_ids:
        display_ids = failed_ids[:20]
        lines.append(f"**Not found ({len(failed_ids)}):** {', '.join(display_ids)}")
        if len(failed_ids) > 20:
            lines[-1] += f" ... +{len(failed_ids) - 20} more"
        lines.append("")
    for paper in succeeded:
        lines.append(format_paper_markdown(paper))
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_bulk_search",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def bulk_search(params: BulkSearchInput) -> str:
    """Search papers with sorting and cursor-based pagination for large result sets.

    Unlike regular search, supports sorting (e.g., by citation count) and
    returns a continuation token for paging through all results.
    """
    logger.info("Bulk searching: %s", params.query)

    api_params: dict[str, Any] = {
        "query": params.query,
        "limit": params.limit,
        "fields": ",".join(PAPER_BULK_SEARCH_FIELDS),
    }
    if params.sort:
        api_params["sort"] = params.sort
    if params.token:
        api_params["token"] = params.token
    if params.year:
        api_params["year"] = params.year
    if params.fields_of_study:
        api_params["fieldsOfStudy"] = ",".join(params.fields_of_study)
    if params.publication_types:
        api_params["publicationTypes"] = ",".join(params.publication_types)
    if params.min_citation_count:
        api_params["minCitationCount"] = params.min_citation_count

    try:
        response = await make_request(
            "GET", "paper/search/bulk", params=api_params, api_key=params.api_key
        )
        total = response.get("total", 0) if isinstance(response, dict) else 0
        token = response.get("token") if isinstance(response, dict) else None
        papers = response.get("data", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    if params.response_format == ResponseFormat.JSON:
        result: dict[str, Any] = {"query": params.query, "total": total, "papers": papers}
        if token:
            result["token"] = token
        return json.dumps(result, indent=2)

    lines = [
        f'## Bulk Search: "{params.query}"',
        f"**Found:** {total} papers (showing {len(papers)})",
        "",
    ]
    for paper in papers:
        lines.append(format_paper_markdown(paper))
    if token:
        lines.append(f"*Next page token: `{token}`*")
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_export_citation",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def export_citation(params: CitationExportInput) -> str:
    """Export a citation for a paper in BibTeX format."""
    logger.info("Exporting citation for: %s", params.paper_id)

    try:
        validate_paper_id(params.paper_id)

        # Check cache first (paper-detail fetch already includes citationStyles).
        cache_key = f"paper:{params.paper_id}"
        paper = cache_get(cache_key)
        if paper is None:
            paper = await make_request(
                "GET",
                f"paper/{params.paper_id}",
                params={"fields": "title,citationStyles"},
                api_key=params.api_key,
            )

        if not isinstance(paper, dict):
            raise ToolError("Unexpected response format from Semantic Scholar API")

        citation_styles = paper.get("citationStyles") or {}
        fmt = params.format.lower()

        if fmt != "bibtex":
            raise ToolError(f"Unsupported citation format: '{fmt}'. Supported: bibtex")

        citation = citation_styles.get("bibtex")
        if not citation:
            title = paper.get("title", params.paper_id)
            raise ToolError(f"No BibTeX citation available for '{title}'")

        return str(citation)

    except SemanticScholarError as e:
        raise ToolError(str(e)) from e


@mcp.tool(
    name="semantic_scholar_match_paper",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def match_paper(params: PaperMatchInput) -> str:
    """Find the single best paper matching a title string. Returns match score."""
    logger.info("Matching paper: %s", params.query)

    try:
        response = await make_request(
            "GET",
            "paper/search/match",
            params={"query": params.query, "fields": ",".join(PAPER_SEARCH_FIELDS)},
            api_key=params.api_key,
        )
        papers = response.get("data", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    if not papers:
        return "No matching paper found."

    paper = papers[0]
    match_score = paper.get("matchScore", 0)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"matchScore": match_score, "paper": paper}, indent=2)

    lines = [
        "## Paper Match",
        f"**Match Score:** {match_score:.1f}",
        "",
        format_paper_markdown(paper),
    ]
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_paper_authors",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def get_paper_authors(params: PaperAuthorsInput) -> str:
    """Get full author profiles for a paper's authors."""
    logger.info("Getting authors for paper: %s", params.paper_id)

    try:
        validate_paper_id(params.paper_id)
        response = await make_request(
            "GET",
            f"paper/{params.paper_id}/authors",
            params={"fields": ",".join(AUTHOR_FIELDS), "limit": params.limit},
            api_key=params.api_key,
        )
        authors = response.get("data", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"paper_id": params.paper_id, "authors": authors}, indent=2)

    lines = [f"## Authors of {params.paper_id}", f"**Found:** {len(authors)} authors", ""]
    for author in authors:
        lines.append(format_author_markdown(author))
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_author_batch",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def get_author_batch(params: AuthorBatchInput) -> str:
    """Retrieve multiple authors in a single request (max 1000)."""
    logger.info("Batch author retrieval: %d authors", len(params.author_ids))

    try:
        response = await make_request(
            "POST",
            "author/batch",
            params={"fields": ",".join(AUTHOR_FIELDS)},
            json_body={"ids": params.author_ids},
            api_key=params.api_key,
        )
        authors = response if isinstance(response, list) else response.get("data", [])
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    succeeded = [a for a in authors if a]
    failed_indices = [i for i, a in enumerate(authors) if not a]
    failed_ids = [params.author_ids[i] for i in failed_indices if i < len(params.author_ids)]

    if params.response_format == ResponseFormat.JSON:
        result: dict[str, Any] = {
            "requested": len(params.author_ids),
            "retrieved": len(succeeded),
            "authors": succeeded,
        }
        if failed_ids:
            result["not_found"] = failed_ids
        return json.dumps(result, indent=2)

    lines = [
        "## Batch Author Retrieval",
        f"**Requested:** {len(params.author_ids)} | **Retrieved:** {len(succeeded)}",
        "",
    ]
    if failed_ids:
        lines.append(f"**Not found ({len(failed_ids)}):** {', '.join(failed_ids[:20])}")
        lines.append("")
    for author in succeeded:
        lines.append(format_author_markdown(author))
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_multi_recommend",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def multi_recommend(params: MultiRecommendInput) -> str:
    """Get recommendations using multiple positive and negative example papers."""
    logger.info(
        "Multi-recommend: %d positive, %d negative",
        len(params.positive_paper_ids),
        len(params.negative_paper_ids),
    )

    all_ids = params.positive_paper_ids + params.negative_paper_ids
    invalid_ids = [pid for pid in all_ids if not is_valid_paper_id(pid)]
    if invalid_ids:
        raise ToolError(f"Invalid paper ID format(s): {', '.join(invalid_ids[:10])}")

    try:
        response = await make_request(
            "POST",
            "papers/",
            params={"fields": ",".join(PAPER_SEARCH_FIELDS_LITE), "limit": params.limit},
            json_body={
                "positivePaperIds": params.positive_paper_ids,
                "negativePaperIds": params.negative_paper_ids,
            },
            api_key=params.api_key,
            base_url=RECOMMENDATIONS_BASE,
        )
        papers = response.get("recommendedPapers", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(
            {
                "positive": params.positive_paper_ids,
                "negative": params.negative_paper_ids,
                "recommendations": papers,
            },
            indent=2,
        )

    lines = [
        "## Multi-Paper Recommendations",
        f"**Positive seeds:** {len(params.positive_paper_ids)}",
        f"**Negative seeds:** {len(params.negative_paper_ids)}",
        f"**Found:** {len(papers)}",
        "",
    ]
    for paper in papers:
        lines.append(format_paper_markdown(paper))
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_snippet_search",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def snippet_search(params: SnippetSearchInput) -> str:
    """Search within paper full text. Returns text snippets with context.

    Note: This endpoint is heavily rate-limited without an API key.
    """
    logger.info("Snippet search: %s", params.query)

    api_params: dict[str, Any] = {
        "query": params.query,
        "limit": params.limit,
        "fields": "snippet.text,snippet.snippetKind,snippet.section",
    }
    if params.paper_ids:
        api_params["paperIds"] = ",".join(params.paper_ids)
    if params.year:
        api_params["year"] = params.year
    if params.fields_of_study:
        api_params["fieldsOfStudy"] = ",".join(params.fields_of_study)
    if params.min_citation_count:
        api_params["minCitationCount"] = params.min_citation_count

    try:
        response = await make_request(
            "GET", "snippet/search", params=api_params, api_key=params.api_key
        )
        results = response.get("data", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        raise ToolError(str(e)) from e

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"query": params.query, "results": results}, indent=2)

    lines = [f'## Snippet Search: "{params.query}"', f"**Found:** {len(results)} snippets", ""]
    for item in results:
        paper = item.get("paper", {})
        snippet = item.get("snippet", {})
        title = paper.get("title", "Unknown")
        section = snippet.get("section", "")
        text = snippet.get("text", "")
        lines.append(f"### {title}")
        if section:
            lines.append(f"**Section:** {section}")
        lines.append(f"> {text}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_status",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def server_status() -> str:
    """Check server health, API connectivity, and key status."""
    has_key = bool(SEMANTIC_SCHOLAR_API_KEY)
    status: dict[str, Any] = {
        "server": "semantic-scholar-mcp",
        "version": __version__,
        "api_key_configured": has_key,
        "rate_tier": "authenticated (10 req/sec)" if has_key else "public (1 req/sec)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not has_key:
        status["tip"] = (
            "Get a free API key for 10x speed: https://www.semanticscholar.org/product/api"
        )
    try:
        # Route health check through make_request for retry/rate-limit protection.
        await make_request(
            "GET",
            "paper/search",
            params={"query": "test", "limit": 1, "fields": "paperId"},
        )
        status["api_reachable"] = True
    except SemanticScholarError as e:
        status["api_reachable"] = False
        status["error"] = str(e)
    except (httpx.HTTPError, OSError, RuntimeError) as e:
        status["api_reachable"] = False
        status["error"] = str(e)

    return json.dumps(status, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run the MCP server over stdio."""
    if not SEMANTIC_SCHOLAR_API_KEY:
        logger.warning(
            "SEMANTIC_SCHOLAR_API_KEY not set. "
            "You can provide api_key per-request or use rate-limited public access (1 req/sec)."
        )
    mcp.run()


if __name__ == "__main__":
    main()


# Public surface for back-compat. Anything tests or downstream callers import
# from ``semantic_scholar_mcp.server`` is re-exported here so the modularization
# is invisible at the import boundary.
__all__ = [
    "AUTHOR_FIELDS",
    "AuthenticationError",
    "AuthorBatchInput",
    "AuthorDetailsInput",
    "AuthorSearchInput",
    "BulkPaperInput",
    "BulkSearchInput",
    "CitationExportInput",
    "MAX_RETRIES",
    "MultiRecommendInput",
    "NotFoundError",
    "PAPER_BULK_SEARCH_FIELDS",
    "PAPER_DETAIL_FIELDS",
    "PAPER_SEARCH_FIELDS",
    "PAPER_SEARCH_FIELDS_LITE",
    "PaperAuthorsInput",
    "PaperDetailsInput",
    "PaperMatchInput",
    "PaperRecommendationsInput",
    "PaperSearchInput",
    "RECOMMENDATIONS_BASE",
    "RETRY_BACKOFF_BASE",
    "RateLimitError",
    "ResponseFormat",
    "SEMANTIC_SCHOLAR_API_BASE",
    "SEMANTIC_SCHOLAR_API_KEY",
    "SemanticScholarError",
    "ServerError",
    "SnippetSearchInput",
    "ValidationError",
    "__version__",
    "_cache_clear",
    "_cache_get",
    "_cache_set",
    "_format_author_markdown",
    "_format_paper_markdown",
    "_get_headers",
    "_handle_error",
    "_is_valid_paper_id",
    "_make_request",
    "_validate_paper_id",
    "bulk_search",
    "export_citation",
    "get_author_batch",
    "get_author_details",
    "get_bulk_papers",
    "get_paper_authors",
    "get_paper_details",
    "get_recommendations",
    "main",
    "match_paper",
    "mcp",
    "multi_recommend",
    "search_authors",
    "search_papers",
    "server_status",
    "snippet_search",
]
