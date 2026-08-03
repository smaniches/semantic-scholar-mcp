#!/usr/bin/env bash
#
# regenerate-locks.sh — regenerate the three committed lock files, or verify
# with --check that the committed files still match a fresh resolve.
#
# Usage:
#   scripts/regenerate-locks.sh            # rewrite the committed locks
#   scripts/regenerate-locks.sh --check    # verify only; nonzero on any drift
#
# Requires network access and exactly uv 0.11.29 (override the binary with the
# UV environment variable). Determinism rests on four things:
#
#   * a single pinned resolver version;
#   * --exclude-newer, which freezes the index to one instant, so a later
#     upstream release cannot change the result;
#   * --python-version 3.10 --universal, which makes the resolution independent
#     of the interpreter and platform that happen to run this script;
#   * a scratch directory entered with `cd`, so every path on the uv command
#     line is a bare filename and the command echoed into each lock header is
#     byte-identical on every run.
#
# The development lock is seeded with the authoritative-base lock below, which
# uv reads as a resolution preference source. That seeding is what keeps the
# development resolution minimal; it is deliberately taken from git, never from
# the committed replacement lock, so this script cannot ratify its own output.
# The build and release locks resolve into paths that must not already exist.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE=5ab7a36e52828f726bec764bbbfb2a881b311273
CUTOFF=2026-08-03T15:29:41Z

MODE=write
case "${1:-}" in
    --check) MODE=check ;;
    "")      ;;
    *)       echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

UV="${UV:-uv}"
if ! command -v "$UV" >/dev/null 2>&1; then
    echo "ERROR: uv not found; install uv 0.11.29 or set UV=/path/to/uv" >&2
    exit 1
fi

case "$("$UV" --version)" in
    "uv 0.11.29"|"uv 0.11.29 "*) ;;
    *)
        echo "ERROR: expected uv 0.11.29" >&2
        "$UV" --version >&2
        exit 1
        ;;
esac

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cp pyproject.toml requirements-build.in requirements-release.in "$WORK/"

# Preference seed for the development resolution: authoritative base, from git.
git show "$BASE:requirements-dev.lock" > "$WORK/requirements-dev.lock"

# The build and release locks must be resolved from scratch, never seeded.
test ! -e "$WORK/requirements-build.lock"
test ! -e "$WORK/requirements-release.lock"

cd "$WORK"

# uv echoes each resolved lock to stdout in addition to writing --output-file.
# Only the echo is discarded here; the generated files are never touched.
#
# uv records its own command line verbatim in each lock header, so the argument
# ORDER below is itself part of the locked bytes. Each invocation is spelled out
# in full rather than assembled from a shared helper: reordering a flag would
# silently rewrite the header and register as drift.
uv_compile() {
    env \
        -u UV_INDEX \
        -u UV_EXTRA_INDEX_URL \
        -u UV_DEFAULT_INDEX \
        -u PIP_INDEX_URL \
        -u PIP_EXTRA_INDEX_URL \
        UV_NO_CONFIG=1 \
        UV_CACHE_DIR="$WORK/uv-cache" \
        "$UV" --no-config pip compile "$@" > /dev/null
}

uv_compile pyproject.toml \
    --extra dev \
    --python-version 3.10 \
    --universal \
    --generate-hashes \
    --exclude-newer "$CUTOFF" \
    --default-index https://pypi.org/simple \
    --output-file requirements-dev.lock

uv_compile requirements-build.in \
    --python-version 3.10 \
    --universal \
    --generate-hashes \
    --exclude-newer "$CUTOFF" \
    --default-index https://pypi.org/simple \
    --output-file requirements-build.lock

uv_compile requirements-release.in \
    --python-version 3.10 \
    --universal \
    --generate-hashes \
    --exclude-newer "$CUTOFF" \
    --default-index https://pypi.org/simple \
    --constraint requirements-build.lock \
    --output-file requirements-release.lock

status=0
for lock in requirements-dev.lock requirements-build.lock requirements-release.lock; do
    if [ "$MODE" = check ]; then
        if cmp -s "$ROOT/$lock" "$WORK/$lock"; then
            echo "ok     $lock"
        else
            echo "DRIFT  $lock" >&2
            diff -u "$ROOT/$lock" "$WORK/$lock" || true
            status=1
        fi
    else
        cp "$WORK/$lock" "$ROOT/$lock"
        echo "wrote  $lock"
    fi
done

if [ "$status" -ne 0 ]; then
    echo "" >&2
    echo "The committed locks do not match a fresh deterministic resolve." >&2
    echo "Run scripts/regenerate-locks.sh and commit the result." >&2
fi

exit "$status"
