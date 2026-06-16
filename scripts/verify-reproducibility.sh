#!/usr/bin/env bash
#
# verify-reproducibility.sh — reproduce every CI quality gate locally, in a
# clean hash-checked virtual environment built from requirements-dev.lock.
#
# Mirrors the CI quality gates (ci.yml: ruff lint + format, mypy --strict,
# 100%-branch pytest, bandit SAST; dependency-audit.yml: pip-audit CVE scan).
# Also advisory-checks that the committed lock still matches a fresh resolve.
#
# Usage:
#   scripts/verify-reproducibility.sh
#
# Requires bash and Python >=3.10. Uses `uv` when available (faster, native
# hash enforcement); otherwise falls back to stdlib venv + pip.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOCK="requirements-dev.lock"
if [ ! -f "$LOCK" ]; then
    echo "ERROR: $LOCK not found. Generate it with:" >&2
    echo "  uv pip compile pyproject.toml --extra dev --universal --generate-hashes -o $LOCK" >&2
    exit 1
fi

WORK="$(mktemp -d)"
VENV="$WORK/venv"
trap 'rm -rf "$WORK"' EXIT

activate() {
    # Windows (Git Bash) puts the activate script under Scripts/; POSIX under bin/.
    if [ -f "$VENV/Scripts/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV/Scripts/activate"
    else
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
    fi
}

echo ">> [1/8] creating clean venv: $VENV"
if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV" >/dev/null
    activate
    echo ">> [2/8] installing dev deps from $LOCK (hash-checked)"
    uv pip install --require-hashes -r "$LOCK" >/dev/null
    echo ">> [3/8] installing package (no deps; already locked)"
    uv pip install --no-deps -e . >/dev/null
else
    python -m venv "$VENV"
    activate
    python -m pip install --upgrade pip >/dev/null
    echo ">> [2/8] installing dev deps from $LOCK (hash-checked)"
    pip install --require-hashes -r "$LOCK" >/dev/null
    echo ">> [3/8] installing package (no deps; already locked)"
    pip install --no-deps -e . >/dev/null
fi

echo ">> [4/8] lint + format (ruff)"
ruff check src/ tests/
ruff format --check src/ tests/

echo ">> [5/8] type check (mypy --strict)"
mypy src/

echo ">> [6/8] tests (100% branch coverage gate)"
# Live-API contract tests (tests/test_api_compatibility.py) require network and
# an API key; they run in their own workflow (.github/workflows/test-api-compat.yml).
# Deselecting them keeps this local check deterministic. src/ coverage stays 100%
# because those tests exercise the remote API over httpx, not this package.
pytest tests/ --deselect tests/test_api_compatibility.py \
    --cov=src/semantic_scholar_mcp --cov-branch --cov-fail-under=100 --tb=short -q

echo ">> [7/8] SAST (bandit) + CVE audit (pip-audit)"
bandit -c pyproject.toml -r src/ -q
pip-audit -r "$LOCK" --strict

echo ">> [8/8] lock freshness (advisory: committed pins == fresh resolve)"
# Advisory only: this re-resolves the '>=' ranges against live PyPI, so it can
# legitimately report drift the moment any upstream dependency publishes a new
# release, with no local change. It therefore WARNS and never fails the run; the
# authoritative gates are steps 1-7. We compare only name==version + --hash
# content: ALL comment lines (including indented '# via' provenance, whose graph
# varies by resolver Python) are stripped, so a comment-only difference is not
# flagged.
if command -v uv >/dev/null 2>&1; then
    FRESH="$WORK/fresh.lock"
    if uv pip compile pyproject.toml --extra dev --universal --generate-hashes \
            --quiet -o "$FRESH" 2>/dev/null; then
        if diff -q <(grep -vE '^[[:space:]]*#' "$LOCK") \
                   <(grep -vE '^[[:space:]]*#' "$FRESH") >/dev/null 2>&1; then
            echo "   lock is current."
        else
            echo "   WARNING: $LOCK differs from a fresh resolve (likely an upstream"
            echo "   release). If intentional, regenerate and commit the lock:"
            echo "     uv pip compile pyproject.toml --extra dev --universal --generate-hashes -o $LOCK"
        fi
    else
        echo "   freshness check skipped (fresh resolve failed; e.g. offline)."
    fi
else
    echo "   skipped (uv not installed)."
fi

echo ""
echo ">> all gates passed."
