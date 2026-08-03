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
BUILD_LOCK="requirements-build.lock"
for required in "$LOCK" "$BUILD_LOCK"; do
    if [ ! -f "$required" ]; then
        echo "ERROR: $required not found. Regenerate the locks with:" >&2
        echo "  scripts/regenerate-locks.sh" >&2
        exit 1
    fi
done

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
    echo ">> [2/8] installing locked build + dev closures (hash-checked)"
    uv pip install --require-hashes -r "$BUILD_LOCK" >/dev/null
    uv pip install --require-hashes -r "$LOCK" >/dev/null
    echo ">> [3/8] installing package offline (editable, no deps, no build isolation)"
    uv pip install --offline --no-deps --no-build-isolation -e . >/dev/null
else
    # Prefer python3: on many distros `python` is absent or points at python2.
    PYTHON="python3"
    command -v python3 >/dev/null 2>&1 || PYTHON="python"
    "$PYTHON" -m venv "$VENV"
    activate
    echo ">> [2/8] installing locked build + dev closures (hash-checked)"
    python -m pip install --require-hashes -r "$BUILD_LOCK" >/dev/null
    python -m pip install --require-hashes -r "$LOCK" >/dev/null
    echo ">> [3/8] installing package offline (editable, no deps, no build isolation)"
    PIP_NO_INDEX=1 python -m pip install --no-deps --no-build-isolation -e . >/dev/null
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

echo ">> [8/8] deterministic lock check (committed locks == fresh resolve)"
# Not advisory. The regeneration is pinned to one resolver version and one index
# cutoff, so a fresh resolve is expected to reproduce the committed locks
# byte-for-byte; any difference is a real defect rather than upstream drift.
# It needs network access and exactly uv 0.11.29, so it is skipped (loudly)
# rather than failed when that resolver is unavailable.
deterministic_lock_check_available() {
    command -v uv >/dev/null 2>&1 || return 1
    case "$(uv --version)" in
        "uv 0.11.29"|"uv 0.11.29 "*) return 0 ;;
        *) return 1 ;;
    esac
}

if deterministic_lock_check_available; then
    "$ROOT/scripts/regenerate-locks.sh" --check
else
    echo "   skipped (requires uv 0.11.29; see CONTRIBUTING.md)."
fi

echo ""
echo ">> all gates passed."
