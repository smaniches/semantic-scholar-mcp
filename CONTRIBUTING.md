# Contributing to Semantic Scholar MCP

Thank you for your interest in contributing! This project is maintained by [TOPOLOGICA LLC](https://topologica.ai).

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/semantic-scholar-mcp.git
   cd semantic-scholar-mcp
   ```
3. Install the hash-pinned development environment, then install the local
   package without dependency or build-backend resolution:
   ```bash
   python -m pip install --require-hashes -r requirements-build.lock
   python -m pip install --require-hashes -r requirements-dev.lock
   PIP_NO_INDEX=1 python -m pip install --no-deps --no-build-isolation -e .
   ```
4. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development

### Running Tests
```bash
pytest
```

### Type Checking
```bash
mypy src/
```

### Linting
```bash
ruff check src/
```

### Security scanning
```bash
bandit -c pyproject.toml -r src/      # SAST; `# nosec` needs an inline reason
pip-audit -r requirements-dev.lock --strict   # dependency CVE audit
```

**A red `Dependency Audit` run** means a CVE was disclosed against a pinned
dependency. Fix it by regenerating the locks onto patched versions (see below)
and committing them. If no fix is yet available, add a documented, time-bound
`--ignore-vuln <GHSA-id>` to the pip-audit step with the advisory link and
rationale, and track removal in `CHANGELOG.md`.

### Reproducing CI locally

The repository has three hash-pinned closures:

- `requirements-dev.lock` for tests, linting, typing, Bandit, and pip-audit;
- `requirements-build.lock` for the build frontend and Hatchling backend;
- `requirements-release.lock`, which composes the build closure and adds Twine.

Lock generation is centralized in `scripts/regenerate-locks.sh` and requires
exactly `uv 0.11.29`:

```bash
scripts/regenerate-locks.sh --check    # verify committed locks
scripts/regenerate-locks.sh --write    # preserve current pins and regenerate
scripts/regenerate-locks.sh --upgrade  # deliberately refresh allowed versions
scripts/verify-reproducibility.sh      # run deterministic local quality gates
```

Commit all affected input and lock files together whenever dependency policy
changes.

## Pull Request Process

1. Ensure tests pass
2. Update documentation if needed
3. Add yourself to CONTRIBUTORS.md (optional)
4. Submit PR with clear description

## Releasing

Releases are automated by release-please, which bumps the version in
`pyproject.toml`, `server.json`, `.zenodo.json`, and `CITATION.cff` in the
release PR. Two things are **intentionally not** automated and must be updated
by hand when the **minor or major** version changes:

- `SECURITY.md` "Supported Versions" table — update **both** rows
  (`X.Y.x | Yes` and `< X.Y | No`). It is a prose security policy, not a
  version-bearing build artifact, so it is kept out of release-please.
  `tests/test_version_consistency.py` fails if the table falls behind the
  package version.

**Merge order matters.** Release PRs are created under `GITHUB_TOKEN`, and
GitHub suppresses workflow runs for events created by that token — so a
release PR shows **no CI**, and the SECURITY.md guard above can only fire on
`main` *after* the release PR merges. To keep `main` green through a
minor/major bump, land the SECURITY.md update **before** merging the release
PR, with the table listing both the current and the upcoming minor — the
guard passes in both states (see the v1.7.0 cycle, where #130 landed ahead
of release PR #129; contrast the v1.6.0 cycle, where the missed update went
red on `main` and needed #127). This is an accepted limitation of
token-scoped automation; do not work around it by giving release-please a
personal access token.

A patch release (`X.Y.Z`) needs no SECURITY.md change.

## Code Style

- Use type hints
- Follow PEP 8
- Write docstrings for public functions
- Keep functions focused and small

## Reporting Issues

- Check existing issues first
- Provide reproduction steps
- Include error messages and logs

## Contact

- **Author:** Santiago Maniches
- **Email:** santiago@topologica.ai
- **Website:** https://topologica.ai

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
