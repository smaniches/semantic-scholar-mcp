"""
Property-based tests using Hypothesis.

Fuzzes Pydantic input models and paper ID validation with random inputs
to discover edge cases that hand-written tests might miss.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError as PydanticValidationError

from semantic_scholar_mcp.server import (
    AuthorDetailsInput,
    AuthorSearchInput,
    BulkPaperInput,
    PaperDetailsInput,
    PaperRecommendationsInput,
    PaperSearchInput,
    ValidationError,
    _validate_paper_id,
)

# ===============================================================================
# STRATEGIES
# ===============================================================================

# Valid 40-char hex strings
hex_40 = st.text(alphabet=string.hexdigits, min_size=40, max_size=40)

# Valid DOI-prefixed IDs (use printable chars to match regex `.+` which excludes \r\n)
doi_ids = st.from_regex(r"DOI:.+", fullmatch=True).filter(lambda s: len(s) <= 60)

# Valid ArXiv-prefixed IDs (ARXIV:NNNN.NNNNN format)
arxiv_ids = st.tuples(
    st.integers(min_value=1000, max_value=9999),
    st.integers(min_value=10000, max_value=99999),
).map(lambda t: f"ARXIV:{t[0]}.{t[1]}")

# Valid PMID-prefixed IDs
pmid_ids = st.integers(min_value=1, max_value=99999999).map(lambda n: f"PMID:{n}")

# Valid CorpusId-prefixed IDs
corpus_ids = st.integers(min_value=1, max_value=999999999).map(lambda n: f"CorpusId:{n}")

# Valid ACL-prefixed IDs
acl_ids = st.text(min_size=1, max_size=20, alphabet=string.ascii_letters + string.digits + "-").map(
    lambda s: f"ACL:{s}"
)

# Any valid paper ID
valid_paper_ids = st.one_of(hex_40, doi_ids, arxiv_ids, pmid_ids, corpus_ids, acl_ids)

# Strings that should NOT be valid paper IDs
invalid_paper_ids = st.text(min_size=1, max_size=100).filter(
    lambda s: (
        len(s.strip()) > 0
        and not (len(s.strip()) == 40 and all(c in string.hexdigits for c in s.strip()))
        and not s.strip().upper().startswith("DOI:")
        and not s.strip().upper().startswith("ARXIV:")
        and not s.strip().upper().startswith("PMID:")
        and not s.strip().upper().startswith("CORPUSID:")
        and not s.strip().upper().startswith("URL:")
        and not s.strip().upper().startswith("ACL:")
    )
)


# ===============================================================================
# PAPER ID VALIDATION
# ===============================================================================


class TestPaperIdPropertyBased:
    """Property-based tests for _validate_paper_id."""

    @given(paper_id=hex_40)
    @settings(max_examples=50)
    def test_any_40_hex_is_valid(self, paper_id: str):
        """Any 40-char hex string should be accepted."""
        _validate_paper_id(paper_id)

    @given(paper_id=doi_ids)
    @settings(max_examples=50)
    def test_any_doi_prefix_is_valid(self, paper_id: str):
        """Any DOI:xxx string should be accepted."""
        _validate_paper_id(paper_id)

    @given(paper_id=arxiv_ids)
    @settings(max_examples=50)
    def test_any_arxiv_prefix_is_valid(self, paper_id: str):
        """Any ARXIV:NNNN.NNNNN string should be accepted."""
        _validate_paper_id(paper_id)

    @given(paper_id=pmid_ids)
    @settings(max_examples=50)
    def test_any_pmid_prefix_is_valid(self, paper_id: str):
        """Any PMID:NNN string should be accepted."""
        _validate_paper_id(paper_id)

    @given(paper_id=corpus_ids)
    @settings(max_examples=50)
    def test_any_corpusid_prefix_is_valid(self, paper_id: str):
        """Any CorpusId:NNN string should be accepted."""
        _validate_paper_id(paper_id)

    @given(paper_id=acl_ids)
    @settings(max_examples=50)
    def test_any_acl_prefix_is_valid(self, paper_id: str):
        """Any ACL:xxx string should be accepted."""
        _validate_paper_id(paper_id)

    def test_empty_string_rejected(self):
        """Empty string should always raise."""
        with pytest.raises(ValidationError):
            _validate_paper_id("")

    def test_whitespace_only_rejected(self):
        """Whitespace-only should always raise."""
        with pytest.raises(ValidationError):
            _validate_paper_id("   \t\n  ")

    @given(paper_id=invalid_paper_ids)
    @settings(max_examples=100)
    def test_invalid_strings_rejected(self, paper_id: str):
        """Strings not matching any pattern should be rejected."""
        with pytest.raises(ValidationError):
            _validate_paper_id(paper_id)


# ===============================================================================
# PYDANTIC MODEL FUZZING
# ===============================================================================


class TestPaperSearchInputFuzzing:
    """Fuzz PaperSearchInput with random values."""

    @given(
        query=st.text(min_size=1, max_size=500).filter(lambda s: len(s.strip()) > 0),
        limit=st.integers(min_value=1, max_value=100),
        offset=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50)
    def test_valid_params_always_accepted(self, query: str, limit: int, offset: int):
        """Any valid combination should produce a model."""
        m = PaperSearchInput(query=query, limit=limit, offset=offset)
        assert m.query.strip() == query.strip()
        assert m.limit == limit
        assert m.offset == offset

    @given(limit=st.integers(min_value=101, max_value=10000))
    @settings(max_examples=20)
    def test_over_limit_rejected(self, limit: int):
        """Limits above 100 should always be rejected."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="test", limit=limit)

    @given(limit=st.integers(min_value=-1000, max_value=0))
    @settings(max_examples=20)
    def test_zero_or_negative_limit_rejected(self, limit: int):
        """Zero or negative limits should always be rejected."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="test", limit=limit)

    @given(offset=st.integers(min_value=-1000, max_value=-1))
    @settings(max_examples=20)
    def test_negative_offset_rejected(self, offset: int):
        """Negative offsets should always be rejected."""
        with pytest.raises(PydanticValidationError):
            PaperSearchInput(query="test", offset=offset)


class TestPaperDetailsInputFuzzing:
    """Fuzz PaperDetailsInput."""

    @given(
        paper_id=st.text(min_size=1, max_size=100).filter(lambda s: len(s.strip()) > 0),
        citations_limit=st.integers(min_value=1, max_value=100),
        references_limit=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30)
    def test_valid_params_accepted(
        self, paper_id: str, citations_limit: int, references_limit: int
    ):
        """Any valid param combo should produce a model."""
        m = PaperDetailsInput(
            paper_id=paper_id,
            citations_limit=citations_limit,
            references_limit=references_limit,
        )
        assert m.citations_limit == citations_limit


class TestBulkPaperInputFuzzing:
    """Fuzz BulkPaperInput."""

    @given(
        n_ids=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=20)
    def test_valid_id_count_accepted(self, n_ids: int):
        """1 to 500 IDs should always be accepted."""
        ids = ["a" * 40] * n_ids
        m = BulkPaperInput(paper_ids=ids)
        assert len(m.paper_ids) == n_ids

    @given(
        n_ids=st.integers(min_value=501, max_value=1000),
    )
    @settings(max_examples=10)
    def test_over_500_ids_rejected(self, n_ids: int):
        """More than 500 IDs should always be rejected."""
        ids = ["a" * 40] * n_ids
        with pytest.raises(PydanticValidationError):
            BulkPaperInput(paper_ids=ids)


class TestAuthorInputsFuzzing:
    """Fuzz author input models."""

    @given(
        query=st.text(min_size=1, max_size=200).filter(lambda s: len(s.strip()) > 0),
    )
    @settings(max_examples=30)
    def test_author_search_valid_query(self, query: str):
        """Any 1-200 char non-empty query should be accepted."""
        m = AuthorSearchInput(query=query)
        assert m.query.strip() == query.strip()

    @given(
        papers_limit=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20)
    def test_author_details_valid_limit(self, papers_limit: int):
        """Any 1-100 papers_limit should be accepted."""
        m = AuthorDetailsInput(author_id="123", papers_limit=papers_limit)
        assert m.papers_limit == papers_limit

    @given(
        limit=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20)
    def test_recommendations_valid_limit(self, limit: int):
        """Any 1-100 limit should be accepted."""
        m = PaperRecommendationsInput(paper_id="a" * 40, limit=limit)
        assert m.limit == limit
