# Release Audit — v1.2.2

**Type:** trust-repair patch
**Date:** 2026-05-05
**Scope:** documentation, metadata, and test-contract corrections only.

This release contains **no behavior change to the API surface, no path-encoding
changes, no API-key handling redesign, no SBOM, no Sigstore signing, no
build-provenance attestation, and no new audit/threat-model documents**. Those
substantive items remain tracked for v1.3.0 (see "Deferred to v1.3.0" below).

---

## Summary of v1.2.2 changes

1. **Fixed a self-contradictory live-API test.** `tests/test_api_compatibility.py`
   contained `test_x_api_key_rejected`, whose body was a verbatim duplicate of
   `test_api_key_accepted` (same `x-api-key` header, same endpoint) but
   asserted `403`. With `SEMANTIC_SCHOLAR_API_KEY` set, that test was guaranteed
   to fail in the weekly compat workflow on first run. Replaced with
   `test_bearer_auth_not_accepted`, which sends `Authorization: Bearer <key>`
   without `x-api-key` and asserts the request was not silently authenticated.
2. **Fixed the pre-existing property-test failure** in
   `tests/test_property_based.py::TestPaperIdPropertyBased::test_any_doi_prefix_is_valid`.
   Hypothesis was generating `DOI:?`; `_validate_paper_id` intentionally rejects
   `?` as a URL-injection guard. Narrowed the strategy to `DOI:[^\s?#]+` (also
   excluding `../`) so the generator matches the validator's contract. The
   validator was **not** weakened.
3. **Synchronized version strings to 1.2.2** across `pyproject.toml`,
   `src/semantic_scholar_mcp/server.py` (`__version__`), `server.json`
   (top-level `version` and `packages[0].version`, both previously stuck at
   `1.0.0`), and the README example response for `semantic_scholar_status`.
4. **Refreshed `.github/SECURITY.md` "Supported Versions" table** from
   `1.0.x` to `1.2.x`.
5. **Aligned the README "Tools Reference" with the registered tool surface.**
   Appended the 7 missing tool sections: `bulk_search`, `export_citation`,
   `match_paper`, `paper_authors`, `author_batch`, `multi_recommend`,
   `snippet_search`. The README now documents all 14 registered tools.
6. **Narrowed the unsupported "most comprehensive" tagline** in README.md
   to a measurable claim. `grep` for sibling phrases produced one historical
   hit in the CHANGELOG entry for 1.2.0 ("100% S2 API coverage"), which was
   also rewritten to a measurable phrasing. The rewrite is recorded in the
   v1.2.2 CHANGELOG entry.
7. **Corrected the false API-key privacy claim** in README.md ("Your API
   key never leaves your machine") with an accurate description: the key is
   sent only to `api.semanticscholar.org` over HTTPS as the `x-api-key`
   header, the server stores no key on disk, no telemetry is collected.
8. **Reconciled the README/SECURITY contradiction** about the per-request
   `api_key` parameter. The README previously instructed its use while
   SECURITY.md denied the parameter existed. Both now acknowledge the
   parameter, document the transcript-exposure risk, and recommend the
   `SEMANTIC_SCHOLAR_API_KEY` environment variable. Parameter removal is
   deferred to v1.3.0.
9. **Downgraded the Development Status classifier** in `pyproject.toml`
   from `5 - Production/Stable` to `4 - Beta` as a deliberate honesty
   correction while documentation and security hardening are brought into
   alignment. Re-promotion is contingent on v1.3.0 hardening.
10. **This document (`RELEASE_AUDIT_v1.2.2.md`)**.
11. **CHANGELOG entry** added covering all of the above.

### New tests guarding against regression

- `tests/test_version_consistency.py` — fails fast in CI if `pyproject.toml`,
  `server.json`, and runtime `__version__` diverge, or if README install
  instructions point at the wrong PyPI distribution name.
- `tests/test_docs_tool_surface.py` — fails fast in CI if a registered
  `@mcp.tool` is missing from the README "Tools Reference", or if the
  README advertises a tool the server does not register.

### Files changed

- `pyproject.toml`
- `server.json`
- `src/semantic_scholar_mcp/server.py` (only `__version__`)
- `README.md`
- `.github/SECURITY.md`
- `CHANGELOG.md`
- `tests/test_api_compatibility.py`
- `tests/test_property_based.py`
- `tests/test_version_consistency.py` (new)
- `tests/test_docs_tool_surface.py` (new)
- `RELEASE_AUDIT_v1.2.2.md` (this file, new)

---

## Validation commands and results

Validation was run inside a clean venv with `pip install -e ".[dev]"`:

| Command | Result |
|---|---|
| `ruff check src/ tests/` | **PASS** — All checks passed! |
| `ruff format --check src/ tests/` | **PASS** — 16 files already formatted |
| `mypy src/` | **PASS** — Success: no issues found in 3 source files |
| `pytest --cov=src/semantic_scholar_mcp --cov-report=term-missing --cov-fail-under=80 --tb=short -q` | **PASS** — 253 passed, 14 skipped, **94.94% coverage** (gate: ≥80%) |

`bandit` is **not** declared in `[project.optional-dependencies].dev`, so
the bandit step from the validation gate was not run.

Live-API tests in `test_api_compatibility.py` skip in this environment
because `SEMANTIC_SCHOLAR_API_KEY` is unset. The 14 SKIPPED tests are
all behind that gate; they will execute under the
`.github/workflows/test-api-compat.yml` weekly cron when the secret is
configured. With v1.2.2's fix to `test_bearer_auth_not_accepted`, that
workflow no longer carries the latent guaranteed-fail of v1.2.1.

---

## Deferred to v1.3.0

Out of scope for this trust-repair patch by explicit user instruction.
The audit identified each as a real follow-up:

1. **URL-encode all path-interpolated identifiers** via
   `urllib.parse.quote(x, safe="")` at server.py:895, 911, 921, 1010, 1024,
   1058, 1218, 1292. Currently `paper_id` and `author_id` are interpolated
   raw via f-strings. The validator rejects `?`, `#`, `\x00`, and `../`
   but does not URL-encode reserved characters such as `/`, `:`, or `+`
   that appear in DOIs and `URL:` identifiers.
2. **Resolve the per-request `api_key` parameter.** Either remove it from
   the public input schemas (env-only) or keep it with explicit redaction
   guidance. The current state — advertised in README, denied in SECURITY
   prior to v1.2.2 — was the worst of both. v1.2.2 reconciles the
   documentation; v1.3.0 should make a structural decision.
3. **SHA-pin all GitHub Actions** in `ci.yml`, `publish.yml`,
   `release-please.yml`, and `test-api-compat.yml`. All references are
   currently tag-pinned (`@v4`, `@v5`, `@v3`, `@release/v1`).
4. **Add Sigstore signing + build-provenance attestation** to
   `.github/workflows/publish.yml`
   (`sigstore/gh-action-sigstore-python` + `actions/attest-build-provenance`).
5. **Generate and publish a CycloneDX SBOM** on each release; attach to
   the GitHub Release.
6. **Add `THREAT_MODEL.md`, `PRIVACY.md`, `AUDIT.md`** to align with the
   uniprot-mcp baseline.
7. **Publish ORCID 0009-0005-6480-1987 in PyPI / pyproject metadata.**
8. **Resolve the PyPI namespace collision** with the `semantic-scholar-mcp`
   distribution held by a different author (`hy20191108`, 12 releases on
   PyPI). v1.2.2 adds a regression test asserting the README install
   command remains `s2-mcp-server`, but the discoverability footgun
   itself is unaddressed.
9. **Disclose AI-assisted release engineering** explicitly in
   `CONTRIBUTING.md` or a future `AUDIT.md`. See the attribution note
   below.

---

## Attribution note

Earlier commit `d1dee9b4596f9d7c5251de6e67d6267dc38a68db` ("chore: bump
version to 1.2.1 for PyPI release", 2026-04-06) was authored by
`Claude <noreply@anthropic.com>`. v1.2.2 and later commits carry
`Santiago Maniches <santiago@topologica.ai>` attribution via per-commit
`-c user.name / user.email` overrides (the persistent global git config
was not modified, per the harness's `NEVER update the git config`
guardrail). Historical commits have not been amended; rewriting published
git history is out of scope and would invalidate any tags or releases
already cut from those commits.
