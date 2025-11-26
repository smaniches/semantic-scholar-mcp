"""
Semantic Scholar MCP Server
===========================

Production MCP server providing direct access to Semantic Scholar's 
database of 200M+ academic papers within Claude Desktop.

Tools Provided:
    - semantic_scholar_search_papers: Advanced paper search with filters
    - semantic_scholar_get_paper: Full paper details with citations/references
    - semantic_scholar_search_authors: Find researchers by name
    - semantic_scholar_get_author: Author profiles and publications
    - semantic_scholar_recommendations: AI-powered related paper discovery
    - semantic_scholar_bulk_papers: Batch retrieval (up to 500 papers)

Configuration:
    Set SEMANTIC_SCHOLAR_API_KEY environment variable with your API key.
    Get a free key at: https://www.semanticscholar.org/product/api

Author: Santiago Maniches
    - ORCID: https://orcid.org/0009-0005-6480-1987
    - LinkedIn: https://www.linkedin.com/in/santiago-maniches/

Organization: TOPOLOGICA LLC
    - Website: https://topologica.ai
    - Email: santiago@topologica.ai

License: MIT
Repository: https://github.com/smaniches/semantic-scholar-mcp

Copyright (c) 2025 TOPOLOGICA LLC. All rights reserved.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API Key: Set via environment variable (user provides their own key)
# Get free key at: https://www.semanticscholar.org/product/api
SEMANTIC_SCHOLAR_API_KEY: str = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

SEMANTIC_SCHOLAR_API_BASE: str = "https://api.semanticscholar.org/graph/v1"
DEFAULT_TIMEOUT: float = 30.0

# Field sets for comprehensive paper metadata
PAPER_FIELDS: List[str] = [
    "paperId", "corpusId", "url", "title", "abstract", "venue", "publicationVenue",
    "year", "referenceCount", "citationCount", "influentialCitationCount",
    "isOpenAccess", "openAccessPdf", "fieldsOfStudy", "s2FieldsOfStudy",
    "publicationTypes", "publicationDate", "journal", "citationStyles",
    "authors", "externalIds", "tldr"
]

AUTHOR_FIELDS: List[str] = [
    "authorId", "externalIds", "url", "name", "aliases", "affiliations",
    "homepage", "paperCount", "citationCount", "hIndex"
]

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("semantic_scholar_mcp")

# ═══════════════════════════════════════════════════════════════════════════════
# MCP SERVER
# ═══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    "semantic_scholar_mcp",
    instructions="""
    Semantic Scholar MCP Server - Access 200M+ academic papers.
    Created by Santiago Maniches (ORCID: 0009-0005-6480-1987)
    TOPOLOGICA LLC - https://topologica.ai
    
    Supports DOI, ArXiv, PubMed, ACL, and Semantic Scholar IDs.
    """
)

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC INPUT MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class PaperSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    year: Optional[str] = Field(default=None, description="Year filter: '2024', '2020-2024', '2020-'")
    fields_of_study: Optional[List[str]] = Field(default=None, description="Filter by fields: ['Computer Science', 'Biology']")
    publication_types: Optional[List[str]] = Field(default=None, description="Filter: 'Review', 'JournalArticle'")
    open_access_only: bool = Field(default=False, description="Only return open access papers")
    min_citation_count: Optional[int] = Field(default=None, description="Minimum citations", ge=0)
    limit: int = Field(default=10, description="Max results (1-100)", ge=1, le=100)
    offset: int = Field(default=0, description="Pagination offset", ge=0)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class PaperDetailsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_id: str = Field(..., description="Paper ID: S2 ID, DOI:xxx, ARXIV:xxx, PMID:xxx, CorpusId:xxx", min_length=1)
    include_citations: bool = Field(default=False, description="Include citing papers")
    include_references: bool = Field(default=False, description="Include referenced papers")
    citations_limit: int = Field(default=10, description="Max citations to return", ge=1, le=100)
    references_limit: int = Field(default=10, description="Max references to return", ge=1, le=100)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class AuthorSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Author name to search", min_length=1, max_length=200)
    limit: int = Field(default=10, description="Max results", ge=1, le=100)
    offset: int = Field(default=0, description="Pagination offset", ge=0)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class AuthorDetailsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    author_id: str = Field(..., description="Semantic Scholar author ID", min_length=1)
    include_papers: bool = Field(default=True, description="Include publications")
    papers_limit: int = Field(default=20, description="Max papers to return", ge=1, le=100)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class PaperRecommendationsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_id: str = Field(..., description="Seed paper ID for recommendations", min_length=1)
    limit: int = Field(default=10, description="Max recommendations", ge=1, le=100)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class BulkPaperInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_ids: List[str] = Field(..., description="List of paper IDs (max 500)", min_length=1, max_length=500)
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON, description="Output format")


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def _get_headers() -> Dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return headers


async def _make_request(
    method: str, endpoint: str, params: Optional[Dict] = None, json_body: Optional[Dict] = None
) -> Dict[str, Any]:
    url = f"{SEMANTIC_SCHOLAR_API_BASE}/{endpoint}"
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            if method == "GET":
                resp = await client.get(url, params=params, headers=_get_headers())
            else:
                resp = await client.post(url, params=params, json=json_body, headers=_get_headers())
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            _handle_error(e.response.status_code)
        except httpx.TimeoutException:
            raise Exception("Request timed out")
    return {}


def _handle_error(status: int) -> None:
    errors = {
        400: "Bad request. Check syntax.",
        401: "Auth failed. Set SEMANTIC_SCHOLAR_API_KEY env var.",
        403: "Forbidden. Check API key.",
        404: "Not found. Check ID format.",
        429: "Rate limited. Wait and retry.",
        500: "Server error. Try later.",
        503: "Service unavailable."
    }
    raise Exception(f"API Error ({status}): {errors.get(status, 'Unknown')}")


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _format_paper_markdown(paper: Dict[str, Any]) -> str:
    lines = []
    title = paper.get("title", "Unknown Title")
    year = paper.get("year", "N/A")
    lines.append(f"### {title} ({year})")
    
    authors = paper.get("authors", [])
    if authors:
        names = [a.get("name", "?") for a in authors[:5]]
        if len(authors) > 5:
            names.append(f"... +{len(authors)-5} more")
        lines.append(f"**Authors:** {', '.join(names)}")
    
    venue = paper.get("venue") or (paper.get("publicationVenue") or {}).get("name")
    if venue:
        lines.append(f"**Venue:** {venue}")
    
    citations = paper.get("citationCount", 0)
    influential = paper.get("influentialCitationCount", 0)
    lines.append(f"**Citations:** {citations} ({influential} influential)")
    
    pdf_info = paper.get("openAccessPdf") or {}
    if pdf_info.get("url"):
        lines.append(f"**Open Access:** [PDF]({pdf_info['url']})")

    fields = paper.get("fieldsOfStudy") or []
    if fields:
        lines.append(f"**Fields:** {', '.join(fields[:5])}")
    
    tldr = paper.get("tldr") or {}
    if tldr.get("text"):
        lines.append(f"**TL;DR:** {tldr['text']}")
    
    abstract = paper.get("abstract")
    if abstract:
        lines.append(f"**Abstract:** {abstract[:500]}..." if len(abstract) > 500 else f"**Abstract:** {abstract}")
    
    ext_ids = paper.get("externalIds") or {}
    ids = []
    if ext_ids.get("DOI"): ids.append(f"DOI: {ext_ids['DOI']}")
    if ext_ids.get("ArXiv"): ids.append(f"ArXiv: {ext_ids['ArXiv']}")
    if ext_ids.get("PubMed"): ids.append(f"PMID: {ext_ids['PubMed']}")
    if ids:
        lines.append(f"**IDs:** {', '.join(ids)}")
    
    if paper.get("url"):
        lines.append(f"**Link:** [{paper.get('paperId')}]({paper['url']})")
    
    lines.append("")
    return "\n".join(lines)


def _format_author_markdown(author: Dict[str, Any]) -> str:
    lines = [f"### {author.get('name', 'Unknown')}"]
    
    affiliations = author.get("affiliations") or []
    if affiliations:
        lines.append(f"**Affiliations:** {', '.join(affiliations[:3])}")
    
    lines.append(f"**h-index:** {author.get('hIndex')} | **Papers:** {author.get('paperCount', 0)} | **Citations:** {author.get('citationCount', 0)}")
    
    if author.get("homepage"):
        lines.append(f"**Homepage:** {author['homepage']}")
    if author.get("url"):
        lines.append(f"**Profile:** [{author.get('authorId')}]({author['url']})")
    
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(name="semantic_scholar_search_papers")
async def search_papers(params: PaperSearchInput) -> str:
    """Search for academic papers. Supports boolean operators (AND, OR, NOT), phrase search with quotes."""
    logger.info(f"Searching: {params.query}")
    
    api_params = {"query": params.query, "offset": params.offset, "limit": params.limit, "fields": ",".join(PAPER_FIELDS)}
    if params.year: api_params["year"] = params.year
    if params.fields_of_study: api_params["fieldsOfStudy"] = ",".join(params.fields_of_study)
    if params.publication_types: api_params["publicationTypes"] = ",".join(params.publication_types)
    if params.open_access_only: api_params["openAccessPdf"] = ""
    if params.min_citation_count: api_params["minCitationCount"] = params.min_citation_count

    response = await _make_request("GET", "paper/search", params=api_params)
    total, papers = response.get("total", 0), response.get("data", [])
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"query": params.query, "total": total, "papers": papers}, indent=2)
    
    lines = [f"## Search Results: \"{params.query}\"", f"**Found:** {total} papers (showing {params.offset + 1}-{params.offset + len(papers)})", ""]
    for paper in papers:
        lines.append(_format_paper_markdown(paper))
    if total > params.offset + len(papers):
        lines.append(f"*Use offset={params.offset + params.limit} to see more results*")
    return "\n".join(lines)


@mcp.tool(name="semantic_scholar_get_paper")
async def get_paper_details(params: PaperDetailsInput) -> str:
    """Get paper details. Accepts: S2 ID, DOI:xxx, ARXIV:xxx, PMID:xxx, CorpusId:xxx"""
    logger.info(f"Getting paper: {params.paper_id}")
    
    paper = await _make_request("GET", f"paper/{params.paper_id}", params={"fields": ",".join(PAPER_FIELDS)})
    result = {"paper": paper}
    
    if params.include_citations:
        cit = await _make_request("GET", f"paper/{params.paper_id}/citations", params={"fields": ",".join(PAPER_FIELDS), "limit": params.citations_limit})
        result["citations"] = cit.get("data", [])
    if params.include_references:
        ref = await _make_request("GET", f"paper/{params.paper_id}/references", params={"fields": ",".join(PAPER_FIELDS), "limit": params.references_limit})
        result["references"] = ref.get("data", [])
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    lines = ["## Paper Details", "", _format_paper_markdown(paper)]
    if result.get("citations"):
        lines.extend(["---", f"### Citing Papers ({len(result['citations'])} shown)", ""])
        for c in result["citations"]:
            p = c.get("citingPaper", {})
            if p: lines.append(f"- **{p.get('title', '?')}** ({p.get('year', '')}) - {p.get('citationCount', 0)} citations")
    if result.get("references"):
        lines.extend(["---", f"### References ({len(result['references'])} shown)", ""])
        for r in result["references"]:
            p = r.get("citedPaper", {})
            if p: lines.append(f"- **{p.get('title', '?')}** ({p.get('year', '')}) - {p.get('citationCount', 0)} citations")
    return "\n".join(lines)


@mcp.tool(name="semantic_scholar_search_authors")
async def search_authors(params: AuthorSearchInput) -> str:
    """Search for academic authors by name."""
    logger.info(f"Searching authors: {params.query}")
    
    response = await _make_request("GET", "author/search", params={"query": params.query, "offset": params.offset, "limit": params.limit, "fields": ",".join(AUTHOR_FIELDS)})
    total, authors = response.get("total", 0), response.get("data", [])
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"query": params.query, "total": total, "authors": authors}, indent=2)
    
    lines = [f"## Author Search: \"{params.query}\"", f"**Found:** {total} authors", ""]
    for author in authors:
        lines.append(_format_author_markdown(author))
    return "\n".join(lines)


@mcp.tool(name="semantic_scholar_get_author")
async def get_author_details(params: AuthorDetailsInput) -> str:
    """Get author profile with optional publications list."""
    logger.info(f"Getting author: {params.author_id}")

    author = await _make_request("GET", f"author/{params.author_id}", params={"fields": ",".join(AUTHOR_FIELDS)})
    result = {"author": author}
    
    if params.include_papers:
        papers = await _make_request("GET", f"author/{params.author_id}/papers", params={"fields": ",".join(PAPER_FIELDS), "limit": params.papers_limit})
        result["papers"] = papers.get("data", [])
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    lines = ["## Author Profile", "", _format_author_markdown(author)]
    if result.get("papers"):
        lines.extend(["---", f"### Publications ({len(result['papers'])} shown)", ""])
        for p in result["papers"]:
            lines.append(f"- **{p.get('title', '?')}** ({p.get('year', '')}) - {p.get('citationCount', 0)} citations")
    return "\n".join(lines)


@mcp.tool(name="semantic_scholar_recommendations")
async def get_recommendations(params: PaperRecommendationsInput) -> str:
    """Get paper recommendations based on a seed paper."""
    logger.info(f"Recommendations for: {params.paper_id}")

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.post(
            f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{params.paper_id}",
            params={"fields": ",".join(PAPER_FIELDS), "limit": params.limit},
            json={"positivePaperIds": [params.paper_id]},
            headers=_get_headers()
        )
        resp.raise_for_status()
        data = resp.json()
    
    papers = data.get("recommendedPapers", [])
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"seed": params.paper_id, "recommendations": papers}, indent=2)
    
    lines = [f"## Recommendations", f"**Seed:** {params.paper_id}", f"**Found:** {len(papers)}", ""]
    for paper in papers:
        lines.append(_format_paper_markdown(paper))
    return "\n".join(lines)


@mcp.tool(name="semantic_scholar_bulk_papers")
async def get_bulk_papers(params: BulkPaperInput) -> str:
    """Retrieve multiple papers in a single request (max 500)."""
    logger.info(f"Bulk retrieval: {len(params.paper_ids)} papers")
    
    response = await _make_request("POST", "paper/batch", params={"fields": ",".join(PAPER_FIELDS)}, json_body={"ids": params.paper_ids})
    papers = response if isinstance(response, list) else response.get("data", [])
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"requested": len(params.paper_ids), "retrieved": len(papers), "papers": papers}, indent=2)
    
    lines = [f"## Bulk Retrieval", f"**Requested:** {len(params.paper_ids)} | **Retrieved:** {len(papers)}", ""]
    for paper in papers:
        if paper: lines.append(_format_paper_markdown(paper))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the MCP server."""
    if not SEMANTIC_SCHOLAR_API_KEY:
        logger.warning("SEMANTIC_SCHOLAR_API_KEY not set. Using rate-limited public access (1 req/sec).")
    mcp.run()


if __name__ == "__main__":
    main()
