"""Hermetic DOI-enrichment contract; no third-party API calls or MCP required."""

from __future__ import annotations

import random
import string
import unittest
from contextlib import suppress

from semantic_scholar_mcp.bibtex_doi import (
    CitationEnrichmentError,
    _split_top_level,
    enrich_bibtex_with_s2_doi,
)

DOI = "10.1016/0022-2836(82)90515-0"
BIB = (
    "@Article{alexnet,\n  title = {Title with {nested} braces},"
    "\n  author = {Doe, Jane and Roe, John},\n  year = {2017}\n}"
)


class CitationDoiTests(unittest.TestCase):
    def test_add_missing_doi_preserves_every_other_field(self):
        actual = enrich_bibtex_with_s2_doi(BIB, DOI)
        self.assertEqual(actual, BIB[:-1].rstrip() + ",\n  doi = {" + DOI + "}\n}")
        self.assertIn("year = {2017}", actual)

    def test_no_doi_has_no_automatic_fallback(self):
        for value in (None, "", "  ", 42, "no doi", "10.12/missing"):
            with self.subTest(value=value), self.assertRaises(CitationEnrichmentError):
                enrich_bibtex_with_s2_doi(BIB, value)

    def test_legacy_doi_with_special_characters_is_unsupported(self):
        # Crossref documents these historical SICI-style identifiers as valid DOIs.
        # The insertion path intentionally rejects unsupported suffix punctuation.
        legacy = (
            "10.1002/(SICI)1521-3951(199911)216:1"
            "<135::AID-PSSB135>3.0.CO;2-#"
        )
        with self.assertRaisesRegex(CitationEnrichmentError, "outside the supported"):
            enrich_bibtex_with_s2_doi(BIB, legacy)

    def test_invalid_bibtex(self):
        invalid = (
            None,
            "",
            "random",
            "@string{foo}",
            "@comment{foo}",
            "@preamble{foo}",
            "@article{a, title={x}",
            "@article{a, title={x}} @article{b, title={y}}",
            "@article{, title={x}}",
            "@article{key}",
            "@article{key, title={x}} trailing garbage",
        )
        for entry in invalid:
            with self.subTest(entry=entry), self.assertRaises(CitationEnrichmentError):
                enrich_bibtex_with_s2_doi(entry, DOI)

    def test_paren_delimiters(self):
        raw = '@inproceedings(key, title={Foo, Bar}, author="Smith, Jane")'
        enriched = enrich_bibtex_with_s2_doi(raw, DOI)
        self.assertEqual(enriched, raw[:-1] + ",\n  doi = {" + DOI + "}\n)")

    def test_existing_matching_doi_preserves_exact_original(self):
        values = (
            "doi={" + DOI + "}",
            'DOI = "' + DOI.upper() + '"',
            "doi = {https://doi.org/" + DOI + "}",
            "doi={doi:" + DOI + "}",
        )
        for field in values:
            with self.subTest(field=field):
                raw = "@article{x, title={Foo}, " + field + "\n}"
                self.assertEqual(enrich_bibtex_with_s2_doi(raw, DOI), raw)

    def test_conflicting_existing_doi_fails_closed(self):
        raw = "@article{x, title={A}, doi={10.5555/notthis}}"
        with self.assertRaisesRegex(CitationEnrichmentError, "conflicts"):
            enrich_bibtex_with_s2_doi(raw, DOI)

    def test_duplicate_doi_fields_fail_closed(self):
        raw = "@article{x, title={A}, doi={" + DOI + "}, DOI={" + DOI + "}}"
        with self.assertRaisesRegex(CitationEnrichmentError, "conflicts"):
            enrich_bibtex_with_s2_doi(raw, DOI)

    def test_malformed_existing_doi_fails_closed(self):
        raw = "@article{x, title={A}, doi={not-a-DOI}}"
        with self.assertRaises(CitationEnrichmentError):
            enrich_bibtex_with_s2_doi(raw, DOI)

    def test_doi_word_inside_title_not_a_field(self):
        raw = "@article{x, title={This is a doi = fake value}}"
        self.assertIn("doi = {" + DOI + "}", enrich_bibtex_with_s2_doi(raw, DOI))

    def test_existing_doi_case_insensitivity(self):
        raw = "@article{x, doi={" + DOI.upper() + "}}"
        self.assertEqual(enrich_bibtex_with_s2_doi(raw, DOI), raw)

    def test_trailing_comma(self):
        raw = "@article{x, title={Fine},\n}"
        updated = enrich_bibtex_with_s2_doi(raw, DOI)
        self.assertNotIn(",,", updated)
        self.assertIn("doi = {" + DOI + "}", updated)

    def test_escaped_quote_and_comma(self):
        raw = r'@Article{x, title="A, B and \"quote\"", author={A, B}}'
        updated = enrich_bibtex_with_s2_doi(raw, DOI)
        self.assertIn("doi = {" + DOI + "}", updated)

    def test_braced_escaped_close(self):
        raw = r"@article{x, title={Text \} and more}}"
        self.assertIn("doi = {" + DOI + "}", enrich_bibtex_with_s2_doi(raw, DOI))

    def test_broken_fields_are_rejected(self):
        invalid = (
            "@article{x, title {Foo}}",
            '@article{x, title="unterminated}',
            "@article{x, title={unterminated}",
            "@article{x, title={A}, badfield}",
            "@article{x, title={A}, weird={A}}@article{evil, x={y}}",
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(CitationEnrichmentError):
                enrich_bibtex_with_s2_doi(raw, DOI)

    def test_no_injection_from_doi(self):
        malicious_values = (
            "10.1234/x}\\n@article{evil",
            "10.1234/x\\n",
            "10.1234/abc}x",
            "10.1234/a,b",
            "10.1234/ab#c",
        )
        for malicious in malicious_values:
            with self.subTest(malicious=malicious), self.assertRaises(CitationEnrichmentError):
                enrich_bibtex_with_s2_doi(BIB, malicious)

    def test_supported_doi_prefix(self):
        for value in (DOI, "https://doi.org/" + DOI, "http://doi.org/" + DOI, "doi:" + DOI):
            with self.subTest(value=value):
                self.assertIn("doi = {" + DOI + "}", enrich_bibtex_with_s2_doi(BIB, value))

    def test_bibtex_fields_with_macro_or_nested_braces(self):
        raw = (
            "@inproceedings{foo,\n  title = {{Vision} in {2020}},"
            "\n  publisher = ACM # Press,\n  year = 2020\n}"
        )
        self.assertIn("doi = {" + DOI + "}", enrich_bibtex_with_s2_doi(raw, DOI))

    def test_reject_trailing_content_after_parenthesized_entry(self):
        with self.assertRaises(CitationEnrichmentError):
            enrich_bibtex_with_s2_doi("@article(key, title={A}) trailing", DOI)

    def test_reject_invalid_internal_delimiter_states(self):
        with self.assertRaises(CitationEnrichmentError):
            _split_top_level("a}b")
        with self.assertRaises(CitationEnrichmentError):
            _split_top_level('key, title="unfinished')

    def test_unbraced_existing_doi(self):
        raw = "@article{x, doi = " + DOI + "}"
        self.assertEqual(enrich_bibtex_with_s2_doi(raw, DOI), raw)

    def test_seeded_idempotence_and_field_preservation(self):
        rng = random.Random(1865)
        letters = string.ascii_lowercase
        for i in range(2000):
            title = "".join(rng.choices(letters, k=rng.randrange(1, 40)))
            raw = f"@article{{ref{i}, title={{{title}}}, year={{2017}}}}"
            result = enrich_bibtex_with_s2_doi(raw, DOI)
            self.assertIn(f"title={{{title}}}", result)
            self.assertIn("year={2017}", result)
            self.assertEqual(result.count("doi = {"), 1)
            self.assertEqual(enrich_bibtex_with_s2_doi(result, DOI), result)

    def test_seeded_bad_input_never_escapes_with_unexpected_exception(self):
        rng = random.Random(20261112)
        alphabet = string.ascii_letters + '{}(),=\\\n"'
        for _ in range(2000):
            raw = "@article{" + "".join(rng.choices(alphabet, k=32))
            with suppress(CitationEnrichmentError):
                enrich_bibtex_with_s2_doi(raw, DOI)

    def test_paren_with_close_brace_at_top_level_is_rejected(self):
        with self.assertRaises(CitationEnrichmentError):
            enrich_bibtex_with_s2_doi("@article(x, title={okay} } )", DOI)


if __name__ == "__main__":
    unittest.main()
