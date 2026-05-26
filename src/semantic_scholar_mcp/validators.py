"""Paper-ID format validation, executed before any outbound request.

Cheap to run, catches obvious user mistakes (typos, URL injection), and
keeps the Semantic Scholar API surface from echoing back opaque 400s.
"""

from __future__ import annotations

import re

from .errors import ValidationError

# Accepted paper-ID formats per the S2 API docs.
_PAPER_ID_PATTERNS = [
    re.compile(r"^[a-f0-9]{40}$", re.IGNORECASE),  # 40-char hex (S2 ID)
    re.compile(r"^DOI:.+$", re.IGNORECASE),
    re.compile(r"^ARXIV:\d+\.\d+.*$", re.IGNORECASE),
    re.compile(r"^PMID:\d+$", re.IGNORECASE),
    re.compile(r"^CorpusId:\d+$", re.IGNORECASE),
    re.compile(r"^URL:.+$", re.IGNORECASE),
    re.compile(r"^ACL:.+$", re.IGNORECASE),
]


def validate_paper_id(paper_id: str) -> None:
    """Raise :class:`ValidationError` if ``paper_id`` is unparseable.

    Accepts: 40-char hex (S2 ID), ``DOI:xxx``, ``ARXIV:xxx``, ``PMID:xxx``,
    ``CorpusId:xxx``, ``URL:xxx``, ``ACL:xxx``.
    """
    if not paper_id or not paper_id.strip():
        raise ValidationError("Paper ID cannot be empty.", status_code=400)

    paper_id = paper_id.strip()

    # Reject injection attempts: NUL, query/fragment delimiters, path traversal.
    if any(c in paper_id for c in "\x00?#") or "../" in paper_id:
        raise ValidationError("Paper ID contains invalid characters.", status_code=400)

    for pattern in _PAPER_ID_PATTERNS:
        if pattern.match(paper_id):
            return

    raise ValidationError(
        f"Invalid paper ID format: '{paper_id}'. "
        "Accepted formats: 40-char hex (S2 ID), DOI:xxx, ARXIV:xxx, PMID:xxx, "
        "CorpusId:xxx, URL:xxx, ACL:xxx",
        status_code=400,
    )


def is_valid_paper_id(paper_id: str) -> bool:
    """Non-raising twin of :func:`validate_paper_id` for bulk pre-checks."""
    if not paper_id or not paper_id.strip():
        return False
    paper_id = paper_id.strip()
    if any(c in paper_id for c in "\x00?#") or "../" in paper_id:
        return False
    return any(p.match(paper_id) for p in _PAPER_ID_PATTERNS)


__all__ = ["is_valid_paper_id", "validate_paper_id"]
