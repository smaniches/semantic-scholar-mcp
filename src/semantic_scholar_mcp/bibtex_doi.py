"""Conservative, opt-in DOI enrichment for a single Semantic Scholar BibTeX entry.

Not a citation validator or a general-purpose BibTeX canonicalizer. Preserve the
original entry unchanged when the existing DOI agrees with S2's externalIds.
Refuse to modify unsupported or structurally ambiguous entries.
"""

from __future__ import annotations

import re


class CitationEnrichmentError(ValueError):
    """Safe DOI enrichment was not possible; no citation should be returned."""


_DOI_RE = re.compile(r"10\.[0-9]{4,9}/[A-Za-z0-9_.:/()+;\-]+\Z", re.IGNORECASE)
_HEADER_RE = re.compile(r"\s*@([A-Za-z]+)\s*([({])")
_FIELD_RE = re.compile(r"([A-Za-z][A-Za-z0-9_-]*)\s*=", re.ASCII)


def _normalize_doi(value: object) -> str:
    """Accept only BibTeX-safe DOI strings; do not guess or sanitize bad input."""
    if not isinstance(value, str):
        raise CitationEnrichmentError("Missing or invalid Semantic Scholar externalIds.DOI")
    doi = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.casefold().startswith(prefix):
            doi = doi[len(prefix):]
            break
    if len(doi) > 255 or not _DOI_RE.fullmatch(doi):
        raise CitationEnrichmentError("Missing or invalid Semantic Scholar externalIds.DOI")
    return doi


def _split_top_level(text: str) -> list[str]:
    """Split outside protected braces and quoted strings; reject malformed input."""
    parts: list[str] = []
    start = 0
    depth = 0
    quoted = False
    escaped = False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"' and depth == 0:
            quoted = not quoted
        elif not quoted:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    raise CitationEnrichmentError("Unbalanced BibTeX braces")
            elif char == "," and depth == 0:
                parts.append(text[start:i].strip())
                start = i + 1
    if depth or quoted or escaped:
        raise CitationEnrichmentError("Unbalanced BibTeX field delimiters")
    parts.append(text[start:].strip())
    return parts


def _bibtex_body(entry: str) -> tuple[str, int]:
    if not isinstance(entry, str):
        raise CitationEnrichmentError("BibTeX must be a string")
    head = _HEADER_RE.match(entry)
    if head is None or head.group(1).casefold() in {"comment", "string", "preamble"}:
        raise CitationEnrichmentError("Expected one standard BibTeX entry")
    opening = head.group(2)
    depth = 0
    quoted = False
    escaped = False
    for i in range(head.end(), len(entry)):
        char = entry[i]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"' and depth == 0:
            quoted = not quoted
        elif not quoted:
            if char == "{":
                depth += 1
            elif char == "}":
                if depth:
                    depth -= 1
                elif opening == "{":
                    if entry[i + 1 :].strip():
                        raise CitationEnrichmentError("Multiple or trailing BibTeX entries")
                    return entry[head.end():i], i
                else:
                    raise CitationEnrichmentError("Unbalanced BibTeX entry braces")
            elif char == ")" and depth == 0 and opening == "(":
                if entry[i + 1 :].strip():
                    raise CitationEnrichmentError("Multiple or trailing BibTeX entries")
                return entry[head.end():i], i
    raise CitationEnrichmentError("Unclosed BibTeX entry")


def enrich_bibtex_with_s2_doi(entry: str, external_doi: object) -> str:
    """Append the S2 structured DOI if absent; fail on disagreement or unsafe syntax.

    This operation does NOT verify the DOI externally, or correct any other
    bibliographic fields. The default export mode must remain verbatim.
    """
    doi = _normalize_doi(external_doi)
    body, end = _bibtex_body(entry)
    items = _split_top_level(body)
    if len(items) < 2 or not items[0]:
        raise CitationEnrichmentError("BibTeX entry has no recognizable citation key")

    found: list[str] = []
    for part in items[1:]:
        if not part:
            continue
        match = _FIELD_RE.match(part)
        if not match:
            raise CitationEnrichmentError("BibTeX field syntax is ambiguous")
        if match.group(1).casefold() != "doi":
            continue
        value = part[match.end():].strip()
        if len(value) >= 2 and (
            (value[0] == "{" and value[-1] == "}")
            or (value[0] == '"' and value[-1] == '"')
        ):
            value = value[1:-1].strip()
        found.append(_normalize_doi(value))

    if found:
        if len(found) != 1 or found[0].casefold() != doi.casefold():
            raise CitationEnrichmentError(
                "BibTeX DOI conflicts with Semantic Scholar externalIds.DOI"
            )
        return entry

    pre = entry[:end].rstrip()
    suffix = entry[end:]
    separator = "" if pre.endswith(",") else ","
    return pre + separator + "\n  doi = {" + doi + "}\n" + suffix
