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

# Accepted paper-identifier syntaxes, mirrored from validators.py. Shared by
# every parameter that names a paper so the guidance stays identical.
_PAPER_ID_FORMATS = (
    "a 40-character S2 hex ID ('649def34f8be52c8b66281af98ae884c09aef38b'), "
    "'DOI:10.18653/v1/N18-3011', 'ARXIV:1706.03762', 'PMID:19872477', "
    "'CorpusId:215416146', 'ACL:W12-3903', or 'URL:https://...'"
)


class PaperSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(
        ...,
        description=(
            "Search query, e.g. 'protein language models'. Supports AND, OR, NOT "
            "operators and quoted phrases: '\"graph neural network\" AND drug'"
        ),
        min_length=1,
        max_length=500,
    )
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
    limit: int = Field(
        default=10,
        description=(
            "Max results per page (1-100, default 10); use semantic_scholar_bulk_search "
            "for result sets beyond 1000"
        ),
        ge=1,
        le=100,
    )
    offset: int = Field(
        default=0,
        description=(
            "Pagination offset: pass the previous offset + limit to fetch the next page (default 0)"
        ),
        ge=0,
    )
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class PaperDetailsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    paper_id: str = Field(
        ...,
        description=f"Paper ID: {_PAPER_ID_FORMATS}",
        min_length=1,
    )
    include_citations: bool = Field(
        default=False,
        description="Also list papers that cite this one (default false)",
    )
    include_references: bool = Field(
        default=False,
        description="Also list papers this one cites (default false)",
    )
    citations_limit: int = Field(
        default=10,
        description="Max citing papers returned when include_citations=true (1-100, default 10)",
        ge=1,
        le=100,
    )
    references_limit: int = Field(
        default=10,
        description="Max referenced papers returned when include_references=true "
        "(1-100, default 10)",
        ge=1,
        le=100,
    )
    response_format: ResponseFormat = _RESPONSE_FORMAT_FIELD
    api_key: str | None = _API_KEY_FIELD


class AuthorSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(
        ...,
        description="Author name to search, e.g. 'Yoshua Bengio'; partial names also match",
        min_length=1,
        max_length=200,
    )
    limit: int = Field(
        default=10, description="Max results per page (1-100, default 10)", ge=1, le=100
    )
    offset: int = Field(
        default=0,
        description=(
            "Pagination offset: pass the previous offset + limit to fetch the next page (default 0)"
        ),
        ge=0,
    )
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
    paper_id: str = Field(
        ...,
        description=f"Seed paper the recommendations should resemble: {_PAPER_ID_FORMATS}",
        min_length=1,
    )
    from_pool: str = Field(
        default="recent",
        description=(
            "Candidate pool: 'recent' (default; recently published papers from all fields) "
            "or 'all-cs' (computer-science papers of any age)"
        ),
    )
    limit: int = Field(
        default=10, description="Max recommendations (1-100, default 10)", ge=1, le=100
    )
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
        description=f"Paper ID: {_PAPER_ID_FORMATS}",
        min_length=1,
    )
    format: str = Field(
        default="bibtex",
        description="Citation format (currently only 'bibtex' supported; anything else errors)",
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
        description=f"Paper ID: {_PAPER_ID_FORMATS}",
        min_length=1,
    )
    limit: int = Field(
        default=100,
        description="Max authors to return, in listed author order (1-1000, default 100)",
        ge=1,
        le=1000,
    )
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
        ...,
        description=(
            "1-100 example papers the recommendations should resemble; "
            f"each ID may be {_PAPER_ID_FORMATS}"
        ),
        min_length=1,
        max_length=100,
    )
    negative_paper_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 100 papers to steer away from: results similar to these are down-ranked "
            "(same ID formats as positive_paper_ids; default none)"
        ),
        max_length=100,
    )
    limit: int = Field(
        default=10, description="Max recommendations (1-500, default 10)", ge=1, le=500
    )
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
