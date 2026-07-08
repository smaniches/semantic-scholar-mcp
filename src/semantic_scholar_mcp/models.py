"""Pydantic input models and field-set constants for the Semantic Scholar API.

Field sets are tiered by the size and shape of the response the API can
return. Sub-endpoints (citations/references/recommendations) silently
ignore ``tldr``; bulk search additionally lacks ``influentialCitationCount``
and ``openAccessPdf``. Carving these into named constants keeps each tool
explicit about what it's asking for without scattering string literals.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# Lightweight: search results, recommendations, bulk, citation/reference sublists.
PAPER_SEARCH_FIELDS: list[str] = [
    "paperId",
    "corpusId",
    "url",
    "title",
    "venue",
    "year",
    "citationCount",
    "influentialCitationCount",
    "isOpenAccess",
    "openAccessPdf",
    "fieldsOfStudy",
    "authors",
    "externalIds",
    "tldr",
]

# Sub-endpoints (recommendations, author/papers, references) don't support tldr.
PAPER_SEARCH_FIELDS_LITE: list[str] = [f for f in PAPER_SEARCH_FIELDS if f != "tldr"]

# Bulk search doesn't support tldr, influentialCitationCount, or openAccessPdf.
PAPER_BULK_SEARCH_FIELDS: list[str] = [
    f for f in PAPER_SEARCH_FIELDS if f not in ("tldr", "influentialCitationCount", "openAccessPdf")
]

# Comprehensive: single-paper detail views only.
PAPER_DETAIL_FIELDS: list[str] = [
    *PAPER_SEARCH_FIELDS,
    "abstract",
    "publicationVenue",
    "referenceCount",
    "s2FieldsOfStudy",
    "publicationTypes",
    "publicationDate",
    "journal",
    "citationStyles",
]

AUTHOR_FIELDS: list[str] = [
    "authorId",
    "externalIds",
    "url",
    "name",
    "affiliations",
    "homepage",
    "paperCount",
    "citationCount",
    "hIndex",
]


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


_API_KEY_FIELD = Field(
    default=None,
    description=(
        "API key (overrides SEMANTIC_SCHOLAR_API_KEY env var). "
        "Deprecated: prefer the environment variable. Removal planned for v2.0.0."
    ),
    json_schema_extra={"deprecated": True},
)
_RESPONSE_FORMAT_FIELD = Field(
    default=ResponseFormat.MARKDOWN,
    description="Output format",
)


class PaperSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    year: str | None = Field(default=None, description="Year filter: '2024', '2020-2024', '2020-'")
    fields_of_study: list[str] | None = Field(
        default=None,
        description="Filter by fields: ['Computer Science', 'Biology']",
    )
    publication_types: list[str] | None = Field(
        default=None, description="Filter: 'Review', 'JournalArticle'"
    )
    open_access_only: bool = Field(default=False, description="Only return open access papers")
    min_citation_count: int | None = Field(default=None, description="Minimum citations", ge=0)
    limit: int = Field(default=10, description="Max results (1-100)", ge=1, le=100)
    offset: int = Field(default=0, description="Pagination offset", ge=0)
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class PaperDetailsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_id: str = Field(
        ...,
        description="Paper ID: S2 ID, DOI:xxx, ARXIV:xxx, PMID:xxx, CorpusId:xxx",
        min_length=1,
    )
    include_citations: bool = Field(default=False, description="Include citing papers")
    include_references: bool = Field(default=False, description="Include referenced papers")
    citations_limit: int = Field(default=10, description="Max citations to return", ge=1, le=100)
    references_limit: int = Field(default=10, description="Max references to return", ge=1, le=100)
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class AuthorSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Author name to search", min_length=1, max_length=200)
    limit: int = Field(default=10, description="Max results", ge=1, le=100)
    offset: int = Field(default=0, description="Pagination offset", ge=0)
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class AuthorDetailsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    author_id: str = Field(..., description="Semantic Scholar author ID", min_length=1)
    include_papers: bool = Field(default=True, description="Include publications")
    papers_limit: int = Field(default=20, description="Max papers to return", ge=1, le=100)
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class PaperRecommendationsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_id: str = Field(..., description="Seed paper ID for recommendations", min_length=1)
    from_pool: str = Field(
        default="recent", description="Paper pool: 'recent' (default) or 'all-cs'"
    )
    limit: int = Field(default=10, description="Max recommendations", ge=1, le=100)
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class BulkPaperInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_ids: list[str] = Field(
        ..., description="List of paper IDs (max 500)", min_length=1, max_length=500
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON, description="Output format"
    )
    api_key: str | None = _API_KEY_FIELD


class BulkSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    sort: str | None = Field(
        default=None,
        description="Sort order: 'citationCount:desc', 'publicationDate:asc', etc.",
    )
    token: str | None = Field(
        default=None,
        description="Continuation token from previous bulk search for pagination",
    )
    year: str | None = Field(default=None, description="Year filter: '2024', '2020-2024', '2020-'")
    fields_of_study: list[str] | None = Field(
        default=None,
        description="Filter by fields: ['Computer Science', 'Biology']",
    )
    publication_types: list[str] | None = Field(
        default=None, description="Filter: 'Review', 'JournalArticle'"
    )
    min_citation_count: int | None = Field(default=None, description="Minimum citations", ge=0)
    limit: int = Field(default=100, description="Max results per page (1-1000)", ge=1, le=1000)
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class CitationExportInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_id: str = Field(
        ...,
        description="Paper ID: S2 ID, DOI:xxx, ARXIV:xxx, PMID:xxx, CorpusId:xxx",
        min_length=1,
    )
    format: str = Field(
        default="bibtex",
        description="Citation format (currently only 'bibtex' supported)",
    )
    api_key: str | None = _API_KEY_FIELD


class PaperMatchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Paper title to match", min_length=1, max_length=500)
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class PaperAuthorsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_id: str = Field(
        ...,
        description="Paper ID: S2 ID, DOI:xxx, ARXIV:xxx, PMID:xxx, CorpusId:xxx",
        min_length=1,
    )
    limit: int = Field(default=100, description="Max authors to return", ge=1, le=1000)
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class AuthorBatchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    author_ids: list[str] = Field(
        ..., description="List of author IDs (max 1000)", min_length=1, max_length=1000
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON, description="Output format"
    )
    api_key: str | None = _API_KEY_FIELD


class MultiRecommendInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    positive_paper_ids: list[str] = Field(
        ..., description="Papers to find similar results for", min_length=1, max_length=100
    )
    negative_paper_ids: list[str] = Field(
        default_factory=list,
        description="Papers to steer recommendations away from",
        max_length=100,
    )
    limit: int = Field(default=10, description="Max recommendations", ge=1, le=500)
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class SnippetSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Search query for paper text", min_length=1, max_length=500)
    paper_ids: list[str] | None = Field(
        default=None, description="Limit search to specific papers", max_length=100
    )
    year: str | None = Field(default=None, description="Year filter: '2024', '2020-2024', '2020-'")
    fields_of_study: list[str] | None = Field(
        default=None,
        description="Filter by fields: ['Computer Science', 'Biology']",
    )
    min_citation_count: int | None = Field(default=None, description="Minimum citations", ge=0)
    limit: int = Field(default=10, description="Max results (1-100)", ge=1, le=100)
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


__all__ = [
    "AUTHOR_FIELDS",
    "PAPER_BULK_SEARCH_FIELDS",
    "PAPER_DETAIL_FIELDS",
    "PAPER_SEARCH_FIELDS",
    "PAPER_SEARCH_FIELDS_LITE",
    "AuthorBatchInput",
    "AuthorDetailsInput",
    "AuthorSearchInput",
    "BulkPaperInput",
    "BulkSearchInput",
    "CitationExportInput",
    "MultiRecommendInput",
    "PaperAuthorsInput",
    "PaperDetailsInput",
    "PaperMatchInput",
    "PaperRecommendationsInput",
    "PaperSearchInput",
    "ResponseFormat",
    "SnippetSearchInput",
]
