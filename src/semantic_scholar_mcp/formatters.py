"""Markdown renderers for paper and author dicts.

Optimized for legibility in chat surfaces (Claude Desktop, Cursor, etc.):
- author lists truncate at five with a ``... +N more`` continuation,
- abstracts truncate at 500 chars so the response fits on one screen,
- empty fields are omitted rather than rendered as ``N/A``.
"""

from __future__ import annotations

from typing import Any

_MAX_AUTHORS_INLINE = 5
_MAX_ABSTRACT_CHARS = 500


def format_paper_markdown(paper: dict[str, Any]) -> str:
    """Render a paper dict as a chat-friendly markdown block."""
    lines = []
    # ``.get(default)`` only fires when the key is missing; the S2 API returns
    # explicit nulls for incomplete records, so coalesce with ``or`` instead.
    title = paper.get("title") or "Unknown Title"
    year = paper.get("year") or "N/A"
    lines.append(f"### {title} ({year})")

    authors = paper.get("authors") or []
    if authors:
        names = [a.get("name") or "?" for a in authors[:_MAX_AUTHORS_INLINE]]
        if len(authors) > _MAX_AUTHORS_INLINE:
            names.append(f"... +{len(authors) - _MAX_AUTHORS_INLINE} more")
        lines.append(f"**Authors:** {', '.join(names)}")

    venue = paper.get("venue") or (paper.get("publicationVenue") or {}).get("name")
    if venue:
        lines.append(f"**Venue:** {venue}")

    # A source that omits its count has not reported zero citations.
    raw_citations = paper.get("citationCount")
    citations = str(raw_citations) if raw_citations is not None else "N/A"
    influential = paper.get("influentialCitationCount")
    # Only render an influential count the provider actually supplied (bulk
    # search never returns one); a genuine 0 is kept, absent/null is omitted.
    if influential is None:
        lines.append(f"**Citations:** {citations}")
    else:
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
        if len(abstract) > _MAX_ABSTRACT_CHARS:
            lines.append(f"**Abstract:** {abstract[:_MAX_ABSTRACT_CHARS]}...")
        else:
            lines.append(f"**Abstract:** {abstract}")

    ext_ids = paper.get("externalIds") or {}
    ids = []
    if ext_ids.get("DOI"):
        ids.append(f"DOI: {ext_ids['DOI']}")
    if ext_ids.get("ArXiv"):
        ids.append(f"ArXiv: {ext_ids['ArXiv']}")
    if ext_ids.get("PubMed"):
        ids.append(f"PMID: {ext_ids['PubMed']}")
    if ids:
        lines.append(f"**IDs:** {', '.join(ids)}")

    if paper.get("url"):
        lines.append(f"**Link:** [{paper.get('paperId')}]({paper['url']})")

    lines.append("")
    return "\n".join(lines)


def format_author_markdown(author: dict[str, Any]) -> str:
    """Render an author dict as a chat-friendly markdown block."""
    lines = [f"### {author.get('name') or 'Unknown'}"]

    affiliations = author.get("affiliations") or []
    if affiliations:
        lines.append(f"**Affiliations:** {', '.join(affiliations[:3])}")

    # h-index: 0 is a meaningful value (early-career), so distinguish missing
    # ("N/A") from explicit 0 instead of using ``or 0``.
    h_index = author.get("hIndex")
    h_index_str = str(h_index) if h_index is not None else "N/A"
    paper_count = author.get("paperCount")
    citation_count = author.get("citationCount")
    paper_count_str = str(paper_count) if paper_count is not None else "N/A"
    citation_count_str = str(citation_count) if citation_count is not None else "N/A"
    lines.append(
        f"**h-index:** {h_index_str} | "
        f"**Papers:** {paper_count_str} | "
        f"**Citations:** {citation_count_str}"
    )

    if author.get("homepage"):
        lines.append(f"**Homepage:** {author['homepage']}")
    if author.get("url"):
        lines.append(f"**Profile:** [{author.get('authorId')}]({author['url']})")

    lines.append("")
    return "\n".join(lines)


__all__ = ["format_author_markdown", "format_paper_markdown"]
