#!/usr/bin/env bash
# Reproduce the deterministic CI, package, SAST, and dependency-audit gates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV_VERSION="0.11.29"
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv $UV_VERSION is required." >&2
    exit 1
fi
case "$(uv --version)" in
    "uv $UV_VERSION"*) ;;
    *)
        echo "ERROR: expected uv $UV_VERSION, found: $(uv --version)" >&2
        exit 1
        ;;
esac

for lock in requirements-dev.lock requirements-build.lock requirements-release.lock; do
    if [[ ! -f "$lock" ]]; then
        echo "ERROR: missing $lock" >&2
        exit 1
    fi
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
DEV_VENV="$WORK/dev-venv"
BUILD_VENV="$WORK/build-venv"

venv_python() {
    local venv="$1"
    if [[ -x "$venv/Scripts/python.exe" ]]; then
        printf '%s\n' "$venv/Scripts/python.exe"
    else
        printf '%s\n' "$venv/bin/python"
    fi
}

echo ">> [1/9] verifying committed locks"
scripts/regenerate-locks.sh --check

echo ">> [2/9] creating clean development environment"
uv venv "$DEV_VENV" >/dev/null
DEV_PYTHON="$(venv_python "$DEV_VENV")"

echo ">> [3/9] installing hash-pinned build and development closures"
uv pip install --python "$DEV_PYTHON" --require-hashes \
    -r requirements-build.lock >/dev/null
uv pip install --python "$DEV_PYTHON" --require-hashes \
    -r requirements-dev.lock >/dev/null

echo ">> [4/9] installing the local package with index access disabled"
PIP_NO_INDEX=1 uv pip install --python "$DEV_PYTHON" --no-index \
    --no-deps --no-build-isolation -e . >/dev/null

echo ">> [5/9] lint, format, and strict typing"
"$DEV_PYTHON" -m ruff check src/ tests/
"$DEV_PYTHON" -m ruff format --check src/ tests/
"$DEV_PYTHON" -m mypy src/

echo ">> [6/9] tests with 100% branch coverage"
"$DEV_PYTHON" -m pytest tests/ --deselect tests/test_api_compatibility.py \
    --cov=src/semantic_scholar_mcp --cov-branch --cov-fail-under=100 \
    --tb=short -q

echo ">> [7/9] SAST and dependency audit"
"$DEV_PYTHON" -m bandit -c pyproject.toml -r src/ -q
"$DEV_PYTHON" -m pip_audit -r requirements-dev.lock --strict

echo ">> [8/9] creating the locked release-tool environment"
uv venv "$BUILD_VENV" >/dev/null
BUILD_PYTHON="$(venv_python "$BUILD_VENV")"
uv pip install --python "$BUILD_PYTHON" --require-hashes \
    -r requirements-release.lock >/dev/null

echo ">> [9/9] building without index access or build isolation"
rm -rf build dist
PIP_NO_INDEX=1 "$BUILD_PYTHON" -m build --no-isolation
"$BUILD_PYTHON" -m twine check dist/*

echo
echo ">> all reproducibility gates passed."
