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
    - semantic_scholar_status: Health check and API connectivity status

Configuration:
    API Key (choose one):
    - Environment variable: Set SEMANTIC_SCHOLAR_API_KEY
    - Per-request: Pass api_key parameter to any tool (takes priority over env var)
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

import asyncio
import json
import logging
import os
import random
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

# ═══════════════════════════════════════════════════════════════════════════════
# VERSION
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "1.1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticScholarError(Exception):
    """Base exception for Semantic Scholar MCP."""
    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(SemanticScholarError):
    """API key invalid or missing (401/403)."""
    pass


class RateLimitError(SemanticScholarError):
    """Rate limit exceeded (429)."""
    def __init__(self, message: str, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(message, status_code=429)


class NotFoundError(SemanticScholarError):
    """Paper/author not found (404)."""
    pass


class ValidationError(SemanticScholarError):
    """Bad request — invalid parameters (400)."""
    pass


class ServerError(SemanticScholarError):
    """Semantic Scholar server error (500/502/503)."""
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API Key: Set via environment variable (user provides their own key)
# Get free key at: https://www.semanticscholar.org/product/api
SEMANTIC_SCHOLAR_API_KEY: str = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

SEMANTIC_SCHOLAR_API_BASE: str = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_BASE: str = "https://api.semanticscholar.org/recommendations/v1"

# Field sets for paper metadata (tiered for efficiency)
# Lightweight: for search results, recommendations, bulk, and citation/reference sublists
PAPER_SEARCH_FIELDS: List[str] = [
    "paperId", "corpusId", "url", "title", "venue", "year",
    "citationCount", "influentialCitationCount", "isOpenAccess",
    "openAccessPdf", "fieldsOfStudy", "authors", "externalIds", "tldr"
]

# Comprehensive: for single paper detail views only
PAPER_DETAIL_FIELDS: List[str] = [
    *PAPER_SEARCH_FIELDS,
    "abstract", "publicationVenue", "referenceCount",
    "s2FieldsOfStudy", "publicationTypes", "publicationDate",
    "journal", "citationStyles"
]

AUTHOR_FIELDS: List[str] = [
    "authorId", "externalIds", "url", "name", "aliases", "affiliations",
    "homepage", "paperCount", "citationCount", "hIndex"
]

# Structured JSON logging
class _StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry)


_handler = logging.StreamHandler()
_handler.setFormatter(_StructuredFormatter())
logger = logging.getLogger("semantic_scholar_mcp")
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# ═══════════════════════════════════════════════════════════════════════════════
# MCP SERVER LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def _lifespan(app: FastMCP) -> AsyncIterator[None]:
    """Lifespan context manager for proper HTTP client cleanup on shutdown."""
    global _client
    logger.info(f"Starting semantic-scholar-mcp v{__version__}")
    try:
        yield
    finally:
        # Close the shared HTTP client on shutdown
        if _client is not None and not _client.is_closed:
            await _client.aclose()
            _client = None
            logger.info("HTTP client closed")
        logger.info("Server shutdown complete")


mcp = FastMCP(
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
    api_key: Optional[str] = Field(default=None, description="API key (overrides SEMANTIC_SCHOLAR_API_KEY env var)")


class PaperDetailsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_id: str = Field(..., description="Paper ID: S2 ID, DOI:xxx, ARXIV:xxx, PMID:xxx, CorpusId:xxx", min_length=1)
    include_citations: bool = Field(default=False, description="Include citing papers")
    include_references: bool = Field(default=False, description="Include referenced papers")
    citations_limit: int = Field(default=10, description="Max citations to return", ge=1, le=100)
    references_limit: int = Field(default=10, description="Max references to return", ge=1, le=100)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")
    api_key: Optional[str] = Field(default=None, description="API key (overrides SEMANTIC_SCHOLAR_API_KEY env var)")


class AuthorSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Author name to search", min_length=1, max_length=200)
    limit: int = Field(default=10, description="Max results", ge=1, le=100)
    offset: int = Field(default=0, description="Pagination offset", ge=0)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")
    api_key: Optional[str] = Field(default=None, description="API key (overrides SEMANTIC_SCHOLAR_API_KEY env var)")


class AuthorDetailsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    author_id: str = Field(..., description="Semantic Scholar author ID", min_length=1)
    include_papers: bool = Field(default=True, description="Include publications")
    papers_limit: int = Field(default=20, description="Max papers to return", ge=1, le=100)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")
    api_key: Optional[str] = Field(default=None, description="API key (overrides SEMANTIC_SCHOLAR_API_KEY env var)")


class PaperRecommendationsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_id: str = Field(..., description="Seed paper ID for recommendations", min_length=1)
    limit: int = Field(default=10, description="Max recommendations", ge=1, le=100)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")
    api_key: Optional[str] = Field(default=None, description="API key (overrides SEMANTIC_SCHOLAR_API_KEY env var)")


class BulkPaperInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_ids: List[str] = Field(..., description="List of paper IDs (max 500)", min_length=1, max_length=500)
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON, description="Output format")
    api_key: Optional[str] = Field(default=None, description="API key (overrides SEMANTIC_SCHOLAR_API_KEY env var)")


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

# Shared HTTP client (lazy singleton)
_client: httpx.AsyncClient | None = None

# Rate limiting state
_rate_semaphore = asyncio.Semaphore(1)
_last_request_time: float = 0.0
_MIN_REQUEST_INTERVAL = 1.0  # seconds (public tier: 1 req/sec)
_MIN_REQUEST_INTERVAL_KEYED = 0.1  # seconds (keyed tier: 10 req/sec)

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds


async def _get_client() -> httpx.AsyncClient:
    """Get or create shared HTTP client with connection pooling."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30
            ),
            headers={"Accept": "application/json", "Content-Type": "application/json"}
        )
    return _client


def _get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Build request headers. User-provided api_key takes priority over env var."""
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    effective_key = api_key or SEMANTIC_SCHOLAR_API_KEY
    if effective_key:
        headers["x-api-key"] = effective_key
    return headers


async def _make_request(
    method: str,
    endpoint: str,
    params: Optional[Dict] = None,
    json_body: Optional[Dict] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any] | list:
    """Make HTTP request to Semantic Scholar API with rate limiting and retry."""
    global _last_request_time

    url = f"{base_url or SEMANTIC_SCHOLAR_API_BASE}/{endpoint}"
    headers = _get_headers(api_key)
    effective_key = api_key or SEMANTIC_SCHOLAR_API_KEY

    # Rate limiting: serialize requests and enforce minimum interval
    async with _rate_semaphore:
        now = time.monotonic()
        elapsed = now - _last_request_time
        interval = _MIN_REQUEST_INTERVAL_KEYED if effective_key else _MIN_REQUEST_INTERVAL
        if elapsed < interval:
            await asyncio.sleep(interval - elapsed)
        _last_request_time = time.monotonic()

        # Execute request with retry logic
        return await _execute_request_with_retry(method, url, params, json_body, headers, api_key)


async def _execute_request_with_retry(
    method: str,
    url: str,
    params: Optional[Dict],
    json_body: Optional[Dict],
    headers: Dict[str, str],
    api_key: Optional[str],
) -> dict[str, Any] | list:
    """Execute HTTP request with exponential backoff retry for retriable errors."""
    client = await _get_client()

    for attempt in range(MAX_RETRIES + 1):
        try:
            if method == "GET":
                resp = await client.get(url, params=params, headers=headers)
            else:
                resp = await client.post(url, params=params, json=json_body, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # Retriable: 429, 503 only
            if status in (429, 503) and attempt < MAX_RETRIES:
                if status == 429:
                    retry_after = float(e.response.headers.get(
                        "Retry-After", RETRY_BACKOFF_BASE * (2 ** attempt)
                    ))
                else:
                    retry_after = RETRY_BACKOFF_BASE * (2 ** attempt)
                jitter = random.uniform(0, 0.5)
                wait = min(retry_after + jitter, 30.0)
                logger.warning(f"HTTP {status}. Retry {attempt+1}/{MAX_RETRIES} after {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            # Non-retriable or exhausted retries: raise appropriate exception
            retry_after_header = e.response.headers.get("Retry-After")
            _handle_error(
                status,
                api_key,
                retry_after=float(retry_after_header) if retry_after_header else None
            )
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"Timeout. Retry {attempt+1}/{MAX_RETRIES} after {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            raise SemanticScholarError("Request timed out after all retries")

    # This line should be unreachable — every path above either returns or raises
    raise SemanticScholarError("Request failed: no response received")


def _handle_error(
    status: int,
    api_key: Optional[str] = None,
    retry_after: float | None = None,
) -> None:
    """Handle API errors with contextual messages and typed exceptions."""
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
        raise RateLimitError("Rate limited. Wait and retry.", retry_after=retry_after)
    if status in (500, 502, 503):
        msg = "Service unavailable." if status == 503 else "Server error. Try later."
        raise ServerError(msg, status_code=status)
    raise SemanticScholarError(f"Unknown error (HTTP {status})", status_code=status)


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER ID VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

# Regex patterns for valid paper ID formats
_PAPER_ID_PATTERNS = [
    re.compile(r"^[a-f0-9]{40}$", re.IGNORECASE),  # 40-char hex (S2 ID)
    re.compile(r"^DOI:.+$", re.IGNORECASE),         # DOI:xxx
    re.compile(r"^ARXIV:\d+\.\d+.*$", re.IGNORECASE),  # ARXIV:2106.15928
    re.compile(r"^PMID:\d+$", re.IGNORECASE),       # PMID:32908142
    re.compile(r"^CorpusId:\d+$", re.IGNORECASE),   # CorpusId:215416146
    re.compile(r"^URL:.+$", re.IGNORECASE),         # URL:xxx
    re.compile(r"^ACL:.+$", re.IGNORECASE),         # ACL:P19-1285
]


def _validate_paper_id(paper_id: str) -> None:
    """Validate paper ID format before API request.

    Accepts:
        - 40-character hex (Semantic Scholar paper ID)
        - DOI:xxx (e.g., DOI:10.1038/s41586-021-03819-2)
        - ARXIV:xxx (e.g., ARXIV:2106.15928)
        - PMID:xxx (e.g., PMID:32908142)
        - CorpusId:xxx (e.g., CorpusId:215416146)
        - URL:xxx (e.g., URL:https://arxiv.org/abs/2106.15928)
        - ACL:xxx (e.g., ACL:P19-1285)

    Raises:
        ValidationError: If the paper ID does not match any accepted format.
    """
    if not paper_id or not paper_id.strip():
        raise ValidationError("Paper ID cannot be empty.", status_code=400)

    paper_id = paper_id.strip()

    for pattern in _PAPER_ID_PATTERNS:
        if pattern.match(paper_id):
            return

    raise ValidationError(
        f"Invalid paper ID format: '{paper_id}'. "
        "Accepted formats: 40-char hex (S2 ID), DOI:xxx, ARXIV:xxx, PMID:xxx, "
        "CorpusId:xxx, URL:xxx, ACL:xxx",
        status_code=400
    )


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

@mcp.tool(
    name="semantic_scholar_search_papers",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def search_papers(params: PaperSearchInput) -> str:
    """Search for academic papers. Supports boolean operators (AND, OR, NOT), phrase search with quotes."""
    logger.info(f"Searching: {params.query}")

    api_params = {"query": params.query, "offset": params.offset, "limit": params.limit, "fields": ",".join(PAPER_SEARCH_FIELDS)}
    if params.year: api_params["year"] = params.year
    if params.fields_of_study: api_params["fieldsOfStudy"] = ",".join(params.fields_of_study)
    if params.publication_types: api_params["publicationTypes"] = ",".join(params.publication_types)
    if params.open_access_only: api_params["openAccessPdf"] = ""
    if params.min_citation_count: api_params["minCitationCount"] = params.min_citation_count

    try:
        response = await _make_request("GET", "paper/search", params=api_params, api_key=params.api_key)
        total = response.get("total", 0) if isinstance(response, dict) else 0
        papers = response.get("data", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        return f"**Error:** {e}"

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"query": params.query, "total": total, "papers": papers}, indent=2)

    lines = [f"## Search Results: \"{params.query}\"", f"**Found:** {total} papers (showing {params.offset + 1}-{params.offset + len(papers)})", ""]
    for paper in papers:
        lines.append(_format_paper_markdown(paper))
    if total > params.offset + len(papers):
        lines.append(f"*Use offset={params.offset + params.limit} to see more results*")
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_get_paper",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def get_paper_details(params: PaperDetailsInput) -> str:
    """Get paper details. Accepts: S2 ID, DOI:xxx, ARXIV:xxx, PMID:xxx, CorpusId:xxx"""
    logger.info(f"Getting paper: {params.paper_id}")

    try:
        _validate_paper_id(params.paper_id)
        paper = await _make_request("GET", f"paper/{params.paper_id}", params={"fields": ",".join(PAPER_DETAIL_FIELDS)}, api_key=params.api_key)
        if not isinstance(paper, dict):
            return "**Error:** Unexpected response format"
        result: Dict[str, Any] = {"paper": paper}

        if params.include_citations:
            cit = await _make_request("GET", f"paper/{params.paper_id}/citations", params={"fields": ",".join(PAPER_SEARCH_FIELDS), "limit": params.citations_limit}, api_key=params.api_key)
            result["citations"] = cit.get("data", []) if isinstance(cit, dict) else []
        if params.include_references:
            ref = await _make_request("GET", f"paper/{params.paper_id}/references", params={"fields": ",".join(PAPER_SEARCH_FIELDS), "limit": params.references_limit}, api_key=params.api_key)
            result["references"] = ref.get("data", []) if isinstance(ref, dict) else []
    except SemanticScholarError as e:
        return f"**Error:** {e}"

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


@mcp.tool(
    name="semantic_scholar_search_authors",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def search_authors(params: AuthorSearchInput) -> str:
    """Search for academic authors by name."""
    logger.info(f"Searching authors: {params.query}")

    try:
        response = await _make_request("GET", "author/search", params={"query": params.query, "offset": params.offset, "limit": params.limit, "fields": ",".join(AUTHOR_FIELDS)}, api_key=params.api_key)
        total = response.get("total", 0) if isinstance(response, dict) else 0
        authors = response.get("data", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        return f"**Error:** {e}"

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"query": params.query, "total": total, "authors": authors}, indent=2)

    lines = [f"## Author Search: \"{params.query}\"", f"**Found:** {total} authors", ""]
    for author in authors:
        lines.append(_format_author_markdown(author))
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_get_author",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def get_author_details(params: AuthorDetailsInput) -> str:
    """Get author profile with optional publications list."""
    logger.info(f"Getting author: {params.author_id}")

    try:
        author = await _make_request("GET", f"author/{params.author_id}", params={"fields": ",".join(AUTHOR_FIELDS)}, api_key=params.api_key)
        if not isinstance(author, dict):
            return "**Error:** Unexpected response format"
        result: Dict[str, Any] = {"author": author}

        if params.include_papers:
            papers = await _make_request("GET", f"author/{params.author_id}/papers", params={"fields": ",".join(PAPER_SEARCH_FIELDS), "limit": params.papers_limit}, api_key=params.api_key)
            result["papers"] = papers.get("data", []) if isinstance(papers, dict) else []
    except SemanticScholarError as e:
        return f"**Error:** {e}"

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    lines = ["## Author Profile", "", _format_author_markdown(author)]
    if result.get("papers"):
        lines.extend(["---", f"### Publications ({len(result['papers'])} shown)", ""])
        for p in result["papers"]:
            lines.append(f"- **{p.get('title', '?')}** ({p.get('year', '')}) - {p.get('citationCount', 0)} citations")
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_recommendations",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def get_recommendations(params: PaperRecommendationsInput) -> str:
    """Get paper recommendations based on a seed paper."""
    logger.info(f"Recommendations for: {params.paper_id}")

    try:
        _validate_paper_id(params.paper_id)
        response = await _make_request(
            "GET",
            f"papers/forpaper/{params.paper_id}",
            params={"fields": ",".join(PAPER_SEARCH_FIELDS), "limit": params.limit},
            api_key=params.api_key,
            base_url=RECOMMENDATIONS_BASE,
        )
        papers = response.get("recommendedPapers", []) if isinstance(response, dict) else []
    except SemanticScholarError as e:
        return f"**Error:** {e}"

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"seed": params.paper_id, "recommendations": papers}, indent=2)

    lines = [f"## Recommendations", f"**Seed:** {params.paper_id}", f"**Found:** {len(papers)}", ""]
    for paper in papers:
        lines.append(_format_paper_markdown(paper))
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_bulk_papers",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def get_bulk_papers(params: BulkPaperInput) -> str:
    """Retrieve multiple papers in a single request (max 500)."""
    logger.info(f"Bulk retrieval: {len(params.paper_ids)} papers")

    # Validate all paper IDs before making request
    invalid_ids = []
    for paper_id in params.paper_ids:
        try:
            _validate_paper_id(paper_id)
        except ValidationError:
            invalid_ids.append(paper_id)

    if invalid_ids:
        return f"**Error:** Invalid paper ID format(s): {', '.join(invalid_ids[:10])}" + (
            f" ... +{len(invalid_ids) - 10} more" if len(invalid_ids) > 10 else ""
        )

    try:
        response = await _make_request("POST", "paper/batch", params={"fields": ",".join(PAPER_SEARCH_FIELDS)}, json_body={"ids": params.paper_ids}, api_key=params.api_key)
        papers = response if isinstance(response, list) else response.get("data", [])
    except SemanticScholarError as e:
        return f"**Error:** {e}"

    # Track and report failures (null entries for unfound papers)
    succeeded = [p for p in papers if p]
    failed_indices = [i for i, p in enumerate(papers) if not p]
    failed_ids = [params.paper_ids[i] for i in failed_indices if i < len(params.paper_ids)]

    if failed_ids:
        logger.warning(f"Bulk retrieval: {len(failed_ids)} papers not found: {failed_ids[:10]}")

    if params.response_format == ResponseFormat.JSON:
        result = {
            "requested": len(params.paper_ids),
            "retrieved": len(succeeded),
            "papers": succeeded,
        }
        if failed_ids:
            result["not_found"] = failed_ids
        return json.dumps(result, indent=2)

    lines = [f"## Bulk Retrieval", f"**Requested:** {len(params.paper_ids)} | **Retrieved:** {len(succeeded)}", ""]
    if failed_ids:
        display_ids = failed_ids[:20]
        lines.append(f"**Not found ({len(failed_ids)}):** {', '.join(display_ids)}")
        if len(failed_ids) > 20:
            lines[-1] += f" ... +{len(failed_ids) - 20} more"
        lines.append("")
    for paper in succeeded:
        lines.append(_format_paper_markdown(paper))
    return "\n".join(lines)


@mcp.tool(
    name="semantic_scholar_status",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
async def server_status() -> str:
    """Check server health, API connectivity, and key status."""
    status: Dict[str, Any] = {
        "server": "semantic-scholar-mcp",
        "version": __version__,
        "api_key_configured": bool(SEMANTIC_SCHOLAR_API_KEY),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        # Route health check through _make_request for retry/rate-limit protection
        await _make_request(
            "GET",
            "paper/search",
            params={"query": "test", "limit": 1, "fields": "paperId"},
        )
        status["api_reachable"] = True
    except SemanticScholarError as e:
        status["api_reachable"] = False
        status["error"] = str(e)
    except Exception as e:
        status["api_reachable"] = False
        status["error"] = str(e)

    return json.dumps(status, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the MCP server."""
    if not SEMANTIC_SCHOLAR_API_KEY:
        logger.warning("SEMANTIC_SCHOLAR_API_KEY not set. You can provide api_key per-request or use rate-limited public access (1 req/sec).")
    mcp.run()


if __name__ == "__main__":
    main()
