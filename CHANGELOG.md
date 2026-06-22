# Changelog

All notable changes documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Versioning: [Semantic Versioning](https://semver.org/).

## [1.5.4](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.5.3...semantic-scholar-mcp-v1.5.4) (2026-06-22)


### CI/CD

* bump actions/checkout from 6.0.3 to 7.0.0 ([#107](https://github.com/smaniches/semantic-scholar-mcp/issues/107)) ([45cf566](https://github.com/smaniches/semantic-scholar-mcp/commit/45cf56612cca533c66552ff807f898fc2333dcf3))

## [1.5.3](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.5.2...semantic-scholar-mcp-v1.5.3) (2026-06-21)


### Bug Fixes

* retry HTTP 502/503 honoring Retry-After and tighten pydantic/httpx bounds ([#105](https://github.com/smaniches/semantic-scholar-mcp/issues/105)) ([2f1ce8d](https://github.com/smaniches/semantic-scholar-mcp/commit/2f1ce8d1ad1dab71ca0218f4699677e4192c341d))

## [1.5.2](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.5.1...semantic-scholar-mcp-v1.5.2) (2026-06-16)


### Documentation

* correct CITATION.cff date-released to 1.5.1 release date ([#102](https://github.com/smaniches/semantic-scholar-mcp/issues/102)) ([0e12d5f](https://github.com/smaniches/semantic-scholar-mcp/commit/0e12d5fbe24df7ef9b7de8e26a39d2f63cb83d81))

## [1.5.1](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.5.0...semantic-scholar-mcp-v1.5.1) (2026-06-16)


### Documentation

* align SECURITY.md supported versions and guard against drift ([5fb65da](https://github.com/smaniches/semantic-scholar-mcp/commit/5fb65da523951e02aa0f77a7e0d5635474f2425b))
* **citation:** lead with concept DOI and correct release date ([#98](https://github.com/smaniches/semantic-scholar-mcp/issues/98)) ([8d768d3](https://github.com/smaniches/semantic-scholar-mcp/commit/8d768d3f28ea40f5b20dc38ca34368a2121a947d))


### CI/CD

* add bandit SAST gate and pinned dependency CVE audit ([07ce2aa](https://github.com/smaniches/semantic-scholar-mcp/commit/07ce2aa13375f8200339a496e696c39c197c3bc8))
* add SAST + dependency CVE gates, hash-pinned lock, and version-drift guard ([73e3dd4](https://github.com/smaniches/semantic-scholar-mcp/commit/73e3dd4a045ba953232972407c4d9070be27eb7d))
* also audit win32-only lock pins on a Windows leg ([017b9ea](https://github.com/smaniches/semantic-scholar-mcp/commit/017b9ea034e16b25469b76438b4f345d75a24904))

## [1.5.0](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.4.0...semantic-scholar-mcp-v1.5.0) (2026-06-10)


### Features

* add Streamable HTTP transport for remote clients ([#96](https://github.com/smaniches/semantic-scholar-mcp/issues/96)) ([d8a7a0d](https://github.com/smaniches/semantic-scholar-mcp/commit/d8a7a0d822816335525023264b403f3166172b84))


### Bug Fixes

* pin HTTP docs to &gt;=1.5.0 and qualify remote-transport key guidance ([#97](https://github.com/smaniches/semantic-scholar-mcp/issues/97)) ([f5d920c](https://github.com/smaniches/semantic-scholar-mcp/commit/f5d920c61a8ef2562e3c95c9cd21bc42a344d79d))
* remove unlisted-Smithery claim from Zenodo description (honesty) ([#90](https://github.com/smaniches/semantic-scholar-mcp/issues/90)) ([84dc4d2](https://github.com/smaniches/semantic-scholar-mcp/commit/84dc4d2b22ae2a2932480baea69ee19464753d01))

## [1.4.0](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.3.4...semantic-scholar-mcp-v1.4.0) (2026-06-09)


### Features

* **status:** surface retry_after in server_status payload ([#79](https://github.com/smaniches/semantic-scholar-mcp/issues/79)) ([732bfc8](https://github.com/smaniches/semantic-scholar-mcp/commit/732bfc8c93c3df308c6b9132e658a93e45ab8965))


### Documentation

* **landing:** use canonical semantic_scholar_ tool names on the Pages site ([#76](https://github.com/smaniches/semantic-scholar-mcp/issues/76)) ([0509933](https://github.com/smaniches/semantic-scholar-mcp/commit/0509933923d009707ded98de2d55db8f85f56ad2))
* **readme:** surface supply-chain provenance + fix 'How it compares' + unstick version example ([#82](https://github.com/smaniches/semantic-scholar-mcp/issues/82)) ([0ebfe51](https://github.com/smaniches/semantic-scholar-mcp/commit/0ebfe51b3f794dea25032d012806d6922cec3aca))
* remove emoji status legend; align status example with tool output ([#83](https://github.com/smaniches/semantic-scholar-mcp/issues/83)) ([fbf02ea](https://github.com/smaniches/semantic-scholar-mcp/commit/fbf02ea0692e7393752d1940152d8534727df0de))


### CI/CD

* bump codecov/codecov-action from 6.0.1 to 7.0.0 ([#86](https://github.com/smaniches/semantic-scholar-mcp/issues/86)) ([6afe85c](https://github.com/smaniches/semantic-scholar-mcp/commit/6afe85c74d52dc94bdf121f71b15d6508b8ccf54))
* bump the github-actions group with 2 updates ([#85](https://github.com/smaniches/semantic-scholar-mcp/issues/85)) ([f040b80](https://github.com/smaniches/semantic-scholar-mcp/commit/f040b80ee72f663ce692775fd14223832fa47efa))


### Testing

* reach 100% line+branch coverage and enforce it in CI ([#84](https://github.com/smaniches/semantic-scholar-mcp/issues/84)) ([81a27e2](https://github.com/smaniches/semantic-scholar-mcp/commit/81a27e2c0376cb934cb64ae40ce5d82fad4c7f6e))

## [1.3.4](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.3.3...semantic-scholar-mcp-v1.3.4) (2026-06-04)


### CI/CD

* drop --output-reproducible so the SBOM keeps serialNumber ([492851c](https://github.com/smaniches/semantic-scholar-mcp/commit/492851c130f86bd0d80e3d45dbf1d46e2a34c180))
* emit CycloneDX 1.5 SBOM so attest-sbom accepts it ([361c33a](https://github.com/smaniches/semantic-scholar-mcp/commit/361c33a1e81df5b0d39a697c0730929e4edfbdd6))

## [1.3.3](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.3.2...semantic-scholar-mcp-v1.3.3) (2026-06-04)


### Bug Fixes

* report accurate version and rate-limit state to MCP clients ([#68](https://github.com/smaniches/semantic-scholar-mcp/issues/68)) ([9d2b10d](https://github.com/smaniches/semantic-scholar-mcp/commit/9d2b10d8f78398d18014d7da2a7c3765f9381541))


### Documentation

* add a GitHub Pages landing page ([#69](https://github.com/smaniches/semantic-scholar-mcp/issues/69)) ([34f5ec3](https://github.com/smaniches/semantic-scholar-mcp/commit/34f5ec3433c4fff1c9bcf50abee99704fe9272e6))
* add the always-present rate_tier field to the status example ([#65](https://github.com/smaniches/semantic-scholar-mcp/issues/65)) ([db72975](https://github.com/smaniches/semantic-scholar-mcp/commit/db72975b067ed3af60c391beef8b0f41988931de))


### CI/CD

* bump actions/download-artifact from 4.3.0 to 8.0.1 ([#74](https://github.com/smaniches/semantic-scholar-mcp/issues/74)) ([dba7400](https://github.com/smaniches/semantic-scholar-mcp/commit/dba740089a1e6373fec4a5cee31800fd5c68b003))
* bump actions/setup-python from 5.6.0 to 6.2.0 ([#71](https://github.com/smaniches/semantic-scholar-mcp/issues/71)) ([08af123](https://github.com/smaniches/semantic-scholar-mcp/commit/08af123abe738df04df10c0c7b34023fc88230be))
* bump actions/upload-artifact from 4.6.2 to 7.0.1 ([#73](https://github.com/smaniches/semantic-scholar-mcp/issues/73)) ([32c6450](https://github.com/smaniches/semantic-scholar-mcp/commit/32c64500738d7cf2f2aac3459b9e6a4cfd3addbf))
* bump codecov/codecov-action from 4.6.0 to 6.0.1 ([#72](https://github.com/smaniches/semantic-scholar-mcp/issues/72)) ([7bf89ad](https://github.com/smaniches/semantic-scholar-mcp/commit/7bf89adbfad7c8013b3fe9b932d6dc5096b50f8e))
* bump github/codeql-action from 3.36.0 to 4.36.0 ([#70](https://github.com/smaniches/semantic-scholar-mcp/issues/70)) ([33037a3](https://github.com/smaniches/semantic-scholar-mcp/commit/33037a3b20a8bde9905072a9de39b087aa55bbfa))
* harden the release supply chain (SHA-pinned Actions, attestations, SBOM) ([#67](https://github.com/smaniches/semantic-scholar-mcp/issues/67)) ([090100a](https://github.com/smaniches/semantic-scholar-mcp/commit/090100a5d4896955f2847c5a7869d88b4530769e))

## [1.3.2](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.3.1...semantic-scholar-mcp-v1.3.2) (2026-05-24)


### Bug Fixes

* intercept pydantic ValidationError before it reaches MCP clients ([#61](https://github.com/smaniches/semantic-scholar-mcp/issues/61)) ([a795d81](https://github.com/smaniches/semantic-scholar-mcp/commit/a795d8106e7026508a1168a2cdb95c28c7d97368))

## [1.3.1](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.3.0...semantic-scholar-mcp-v1.3.1) (2026-05-18)


### Documentation

* point DOI badge and CITATION.cff at the Zenodo concept DOI ([#60](https://github.com/smaniches/semantic-scholar-mcp/issues/60)) ([f64e756](https://github.com/smaniches/semantic-scholar-mcp/commit/f64e756466362529beca722fd9919d44eda5a497))


### CI/CD

* fix GITHUB_TOKEN suppression of release: published downstream workflows ([#57](https://github.com/smaniches/semantic-scholar-mcp/issues/57)) ([9359952](https://github.com/smaniches/semantic-scholar-mcp/commit/9359952f54451f4731bfef2adac122cb0eec0bf4))
* make workflow_dispatch honor inputs.tag for build ref and Docker semver tags ([#59](https://github.com/smaniches/semantic-scholar-mcp/issues/59)) ([e4f1c1d](https://github.com/smaniches/semantic-scholar-mcp/commit/e4f1c1d3a2001f3824d3f24b8da23e66b8183733))

## [1.3.0](https://github.com/smaniches/semantic-scholar-mcp/compare/semantic-scholar-mcp-v1.2.2...semantic-scholar-mcp-v1.3.0) (2026-05-18)


### Features

* add Docker support and registry badges ([#14](https://github.com/smaniches/semantic-scholar-mcp/issues/14)) ([5f11d8b](https://github.com/smaniches/semantic-scholar-mcp/commit/5f11d8bec0b4196e0f9d3819a35e7eef7cef5389))
* add structured error responses (isError=True) and TTL cache ([9dea864](https://github.com/smaniches/semantic-scholar-mcp/commit/9dea864b5f48819a04d32da05a6169c7c35a8860))
* add structured error responses (isError=True) and TTL cache ([b6f4000](https://github.com/smaniches/semantic-scholar-mcp/commit/b6f4000f38140059e3d2de31b0a2bce843de747a))
* bulk search, citation export, parallel sub-requests (v1.1.0) ([513f622](https://github.com/smaniches/semantic-scholar-mcp/commit/513f6220fa6b3ee7370125a3d1db387fe0ef144e))
* full S2 API coverage, 5 new tools, UX hardening (v1.2.0) ([95a175f](https://github.com/smaniches/semantic-scholar-mcp/commit/95a175f35647ad3f9a8bba90ac09ff68d622cd08))
* user-provided API key support ([981188f](https://github.com/smaniches/semantic-scholar-mcp/commit/981188f8cb6943d21df71614328fcc85062d2a51))


### Bug Fixes

* add 429 retry and increase rate limit delay for CI ([da2e5c5](https://github.com/smaniches/semantic-scholar-mcp/commit/da2e5c51ff5630eb287edb31952afdfaaace6312))
* add missing type annotations to resolve mypy CI failures ([28ccb0a](https://github.com/smaniches/semantic-scholar-mcp/commit/28ccb0a722e1938526e63809d1cb8b0804758c75))
* add s2-mcp-server entry point so uvx s2-mcp-server works ([f06e1a3](https://github.com/smaniches/semantic-scholar-mcp/commit/f06e1a3481b6b79c42d148a191572f4972218364))
* Bearer auth, remove unsupported S2 API fields, add API compat tests ([d37161c](https://github.com/smaniches/semantic-scholar-mcp/commit/d37161c0e8d09537b00b21620f4cba87bd710ed9))
* bulk_search 400 error and snippet_search API key not forwarded ([bcd3a5f](https://github.com/smaniches/semantic-scholar-mcp/commit/bcd3a5f1dce9dd615548f679781c8320f0f98be9))
* filter tldr from citations endpoint (v1.0.3) ([67d98b0](https://github.com/smaniches/semantic-scholar-mcp/commit/67d98b0a37c44955e1da52e2e613ebc2d25f2a2a))
* lint errors and invalid author test ID ([e6b32ba](https://github.com/smaniches/semantic-scholar-mcp/commit/e6b32ba0fabc63261a584e1735af6f7eceab0ea4))
* prevent api-compat CI rate limiting with sequential jobs and retries ([39ee596](https://github.com/smaniches/semantic-scholar-mcp/commit/39ee59674488240a0f3c5f751b1de8749f324afc))
* production hardening — single version source, CI coverage+Windows, examples, entry point tests ([5f60fb8](https://github.com/smaniches/semantic-scholar-mcp/commit/5f60fb895af5b2226f75ee1c2931fe8623c32a38))
* rename PyPI package to s2-mcp-server ([f7797bd](https://github.com/smaniches/semantic-scholar-mcp/commit/f7797bd1baf7afbf836ef95a8eb602080372ff7a))
* resolve mypy no-any-return error in export_citation ([35d4574](https://github.com/smaniches/semantic-scholar-mcp/commit/35d4574e7991e896a5c6f0bf882f69fcd94ee492))
* resolve pre-existing lint errors (E501, F401, F841) ([855e1d1](https://github.com/smaniches/semantic-scholar-mcp/commit/855e1d132c6e92e668a656f9442f0ff9b9e2bacd))
* skip live API tests without SEMANTIC_SCHOLAR_API_KEY ([aae53a8](https://github.com/smaniches/semantic-scholar-mcp/commit/aae53a898534a9f8cc70a586176233287bf47dd4))
* update existing test to expect Bearer auth instead of x-api-key ([50086a0](https://github.com/smaniches/semantic-scholar-mcp/commit/50086a0536dbac9e5b3b44231ffc76e60ee2baa8))
* update repo references from topologica-ai to smaniches ([d47fa45](https://github.com/smaniches/semantic-scholar-mcp/commit/d47fa45cbb08c7337ae7b323c4c7f2530a90f0d3))
* update repo references from topologica-ai to smaniches ([89dbba1](https://github.com/smaniches/semantic-scholar-mcp/commit/89dbba126cf60596a98df29b94e9dc4aae0a8172))
* update tool registration tests to expect all 14 tools ([32f8f55](https://github.com/smaniches/semantic-scholar-mcp/commit/32f8f55aa571a96a674cab2a458165e20dadfd2b))


### Dependencies

* bump python from 3.12-slim to 3.14-slim ([#38](https://github.com/smaniches/semantic-scholar-mcp/issues/38)) ([909175c](https://github.com/smaniches/semantic-scholar-mcp/commit/909175c2708b0bc69112c93ac3b427731168f50c))


### Documentation

* add Zenodo DOI badge ([#30](https://github.com/smaniches/semantic-scholar-mcp/issues/30)) ([65e3938](https://github.com/smaniches/semantic-scholar-mcp/commit/65e3938729169cf5a5caf2f7b4ba2ad7b74352a6))
* audience-neutral framing, shields.io DOI badge, related-MCP cross-promo ([#43](https://github.com/smaniches/semantic-scholar-mcp/issues/43)) ([cf86365](https://github.com/smaniches/semantic-scholar-mcp/commit/cf86365fad450a66d27f3c47c417855bb3f4bb09))
* remove stale "tracked for v1.3.0" promises before 1.3.0 ships ([#46](https://github.com/smaniches/semantic-scholar-mcp/issues/46)) ([19740be](https://github.com/smaniches/semantic-scholar-mcp/commit/19740be088c3151c5bfd266dc110b1e13fe1c090))


### CI/CD

* add release automation, Dependabot, Codecov, and PyPI publishing ([8c2254e](https://github.com/smaniches/semantic-scholar-mcp/commit/8c2254e4d96aea0142c6125cba0aaf38d70711e5))
* add release automation, Dependabot, Codecov, and PyPI publishing ([cc4dcdf](https://github.com/smaniches/semantic-scholar-mcp/commit/cc4dcdf3d8a165df2bf852afb509fa7d6594d365))
* bump docker/build-push-action from 6 to 7 ([#40](https://github.com/smaniches/semantic-scholar-mcp/issues/40)) ([06cfe61](https://github.com/smaniches/semantic-scholar-mcp/commit/06cfe612964514d5cc6b97591accdfcd1d8ce7a5))
* bump docker/login-action from 3 to 4 ([#39](https://github.com/smaniches/semantic-scholar-mcp/issues/39)) ([94d7f0f](https://github.com/smaniches/semantic-scholar-mcp/commit/94d7f0f149a63c1d0754ce5c0a964eabffa63d73))
* bump GitHub Actions versions ([#36](https://github.com/smaniches/semantic-scholar-mcp/issues/36)) ([86ca546](https://github.com/smaniches/semantic-scholar-mcp/commit/86ca546e3414deca964d79fe91403eccbfa54b2f))
* harden dependabot config (group actions, group pip by type, add docker) ([#37](https://github.com/smaniches/semantic-scholar-mcp/issues/37)) ([1adc671](https://github.com/smaniches/semantic-scholar-mcp/commit/1adc671682b774461173b7e7d8e7ad629309691c))
* make releases self-consistent forever (CITATION.cff, .zenodo.json, full release-please coverage) ([#44](https://github.com/smaniches/semantic-scholar-mcp/issues/44)) ([281d1c6](https://github.com/smaniches/semantic-scholar-mcp/commit/281d1c607f6394da024e506455c3f69d99c4f263))
* migrate release-please to v5 manifest mode ([#41](https://github.com/smaniches/semantic-scholar-mcp/issues/41)) ([65c201c](https://github.com/smaniches/semantic-scholar-mcp/commit/65c201cae667f6eab663b4fdf72f798966bb70ab))


### Testing

* close coverage gaps for retry, bulk, detail fields, backoff timing ([46cfec5](https://github.com/smaniches/semantic-scholar-mcp/commit/46cfec5fdcd6cd594de7ab9687e9d459f58b29c3))


### Refactoring

* modularize server.py, single-source __version__, add architecture diagram ([#45](https://github.com/smaniches/semantic-scholar-mcp/issues/45)) ([860a202](https://github.com/smaniches/semantic-scholar-mcp/commit/860a202b426f8dfaa08c7ac7520ec5cd42ddf042))
* production hardening with 10 best practices ([f955bff](https://github.com/smaniches/semantic-scholar-mcp/commit/f955bff2c5728e1ef03a87077558476c655ead99))
* server lifecycle, validation, and comprehensive tests ([a3ebaf3](https://github.com/smaniches/semantic-scholar-mcp/commit/a3ebaf31f8440b89e230a57cf87bf817241ba817))
* zero technical debt, Smithery registry, and installation docs ([787d905](https://github.com/smaniches/semantic-scholar-mcp/commit/787d905ad0f51cf2d675444744a44f9055f49ce4))
* zero technical debt, Smithery registry, and installation docs ([7a26e9e](https://github.com/smaniches/semantic-scholar-mcp/commit/7a26e9e2dcd71a695226977ad1daad2b8871a9ea))

## [1.2.2] - 2026-05-05

Trust-repair patch. Documentation, metadata, and test-contract corrections
only — no behavior change to the API surface, no path-encoding changes, no
key-handling redesign, no supply-chain hardening. Those substantive items
remain tracked for v1.3.0.

### Fixed
- `tests/test_api_compatibility.py::test_x_api_key_rejected` was a verbatim
  duplicate of `test_api_key_accepted` but asserted `403`. Replaced with
  `test_bearer_auth_not_accepted`, which sends `Authorization: Bearer <key>`
  (without `x-api-key`) and asserts the response is not silently authenticated.
- `tests/test_property_based.py::test_any_doi_prefix_is_valid` was failing on
  main: the Hypothesis strategy generated `DOI:?` and `_validate_paper_id`
  intentionally rejects `?` as a URL-injection guard. Narrowed the strategy
  to `DOI:[^\s?#]+` (also excluding `../`) to match the validator's contract.
  The validator was *not* weakened.
- `server.json` `version` and `packages[0].version` were stuck at `1.0.0`
  while the package shipped `1.2.1` to PyPI. Synchronized to `1.2.2`.
- `.github/SECURITY.md` "Supported Versions" table was stuck at `1.0.x`.
  Updated to `1.2.x`.
- README "Tools Reference" documented only 7 of the 14 registered tools.
  Added entries for `bulk_search`, `export_citation`, `match_paper`,
  `paper_authors`, `author_batch`, `multi_recommend`, `snippet_search`.
- README claim "Your API key never leaves your machine" was false: the key
  is sent to `api.semanticscholar.org` as the `x-api-key` header. Replaced
  with an accurate description of where the key actually goes.
- README and SECURITY.md disagreed about the per-request `api_key` parameter
  (README instructed its use; SECURITY denied it existed). Reconciled both
  to acknowledge the parameter, document the transcript-exposure risk, and
  recommend the `SEMANTIC_SCHOLAR_API_KEY` environment variable. Parameter
  removal is deferred to v1.3.0.
- README example response for `semantic_scholar_status` showed
  `"version": "1.0.0"`. Updated to `"1.2.2"`.

### Changed
- README opening tagline narrowed from "The most comprehensive MCP server
  for academic research" (no comparison evidence; three live PyPI rivals
  exist) to a measurable claim: "A comprehensive 14-tool MCP server for
  Semantic Scholar academic research workflows."
- CHANGELOG entry for 1.2.0 narrowed "100% S2 API coverage" to
  "14 tools across the Semantic Scholar Graph, Recommendations, and Snippet
  APIs". Historical accuracy is preserved by this change-log entry recording
  the rewrite.
- `pyproject.toml` `Development Status` classifier downgraded from
  `5 - Production/Stable` to `4 - Beta` as a deliberate honesty correction
  while documentation and security hardening are brought into alignment.
  Re-promotion is contingent on the v1.3.0 hardening landing.

### Added
- `tests/test_version_consistency.py`: fails fast in CI if `pyproject.toml`,
  `server.json`, and runtime `__version__` diverge, or if the README install
  command points at the wrong PyPI distribution name.
- `tests/test_docs_tool_surface.py`: fails fast in CI if a registered
  `@mcp.tool` is missing from the README, or if the README advertises a tool
  the server does not register.
- `RELEASE_AUDIT_v1.2.2.md`: scope, validation results, deferred items, and
  attribution note for this release.
- SECURITY.md: explicit "Known Limitations (v1.2.x)" section enumerating
  the items deferred to v1.3.0 (path encoding, per-request `api_key`
  parameter, GitHub Actions SHA-pinning, Sigstore/SBOM/attestation).

## [1.2.1] - 2026-04-06

### Fixed
- `bulk_search`: HTTP 400 caused by requesting unsupported fields (tldr, influentialCitationCount, openAccessPdf); added PAPER_BULK_SEARCH_FIELDS
- `snippet_search`: API key not forwarded; reverted auth header from Authorization: Bearer back to x-api-key (the Bearer migration in v1.0.2 broke the snippet endpoint)
- mypy no-any-return error in export_citation
- Pre-existing ruff lint (E501, F401, F841) and format issues
- Tool registration tests updated from 7 to 14 tools

### Changed
- Version bump to 1.2.1

## [1.2.0] - 2026-04-05

### Added
- New tool: `semantic_scholar_match_paper` — find paper by exact title match with match score
- New tool: `semantic_scholar_paper_authors` — full author profiles for a paper's authors
- New tool: `semantic_scholar_author_batch` — batch author retrieval (up to 1000)
- New tool: `semantic_scholar_multi_recommend` — multi-paper recommendations with positive and negative examples
- New tool: `semantic_scholar_snippet_search` — search within paper full text (requires API key)
- `from_pool` parameter on recommendations: choose "recent" (default) or "all-cs" paper pool
- Smart status output: rate tier info, API key setup tip for unauthenticated users
- Input sanitization: reject null bytes, path traversal, and query injection in paper IDs
- Friendly 429 error messages: API key signup URL for unauthenticated users

### Changed
- Version bump to 1.2.0 (14 tools across the Semantic Scholar Graph, Recommendations, and Snippet APIs)

## [1.1.0] - 2026-04-05

### Added
- New tool: `semantic_scholar_bulk_search` — sorted search with cursor-based pagination for large result sets
- New tool: `semantic_scholar_export_citation` — BibTeX citation export for any paper
- Parallel fetching of citations and references in `get_paper_details` via `asyncio.gather`

### Changed
- Version bump to 1.1.0

## [1.0.3] - 2026-04-04

### Fixed
- Filter tldr from citations endpoint (missed in v1.0.2); use PAPER_SEARCH_FIELDS_LITE for all sub-endpoints
- Fix incorrect test asserting citations accepts tldr

### Changed
- Version bump to 1.0.3

## [1.0.2] - 2026-04-01

### Fixed
- Auth: migrated from deprecated x-api-key to Authorization: Bearer
- Removed unsupported aliases field from author queries
- Filtered tldr from recommendations, author/papers, and references endpoints

### Added
- PAPER_SEARCH_FIELDS_LITE constant for sub-endpoints with restricted field support
- Automated S2 API compatibility test suite (14 tests)
- Weekly CI workflow to catch S2 API changes early

## [1.0.0] - 2026-02-23

### Changed
- Distribution moved to GitHub releases only (no PyPI)
- Replaced broken PyPI badge with GitHub Release badge

### Added
- SECURITY.md with vulnerability reporting process
- CODEOWNERS file
- py.typed marker for type checker compatibility
- GitHub Actions CI (test matrix: Python 3.10–3.12, lint, type check)
- GitHub issue templates (bug report, feature request)
- Pull request template
- CHANGELOG.md
- Comprehensive pyproject.toml metadata (classifiers, keywords, URLs, ruff config)

### Security
- Environment-variable-only API key management enforced
- Input validation documented
- HTTPS-only API communication

## [0.2.0] - 2026-01-18

### Added
- Full Semantic Scholar API coverage: paper search, author lookup, recommendations, bulk retrieval
- Support for all paper ID formats: DOI, ArXiv, PubMed, ACL, CorpusId
- API key authentication with rate limit handling
- Claude Desktop integration via stdio MCP protocol

## [0.1.0] - 2025-11-25

### Added
- Initial release
