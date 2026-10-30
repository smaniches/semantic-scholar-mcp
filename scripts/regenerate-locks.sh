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
# Keep the development closure on its previously reviewed index snapshot so a
# build/release-tool bump cannot silently re-resolve unrelated dev dependencies.
DEV_CUTOFF=2026-09-30T23:45:48Z
# Refresh only the security-patched dependencies; unrelated packages remain
# on the original, immutable September 30 index snapshot.
SECURITY_PATCH_CUTOFF=2026-10-23T20:40:00Z
# 1.2.4 (released October 26) closes CVE-2026-114613 in dev-only
# cachecontrol[filecache]. Keep this separate from the older security cutoff
# so unrelated packages do not float.
MSGPACK_PATCH_CUTOFF=2026-10-30T00:00:00Z
# One second after hatchling 1.32.4's final distribution was published
# (2026-09-20T22:48:45Z), the newest exact pin declared in requirements-build.in.
# The build/release base cutoff predates the later explicit cyclonedx-bom 7.5.0
# pin. Give ONLY that release tool a package-specific cutoff, 433 ms after its
# last required distribution appeared on PyPI. This preserves the frozen
# transitive index for every unrelated dependency.
BUILD_RELEASE_CUTOFF=2026-09-20T22:48:46Z
CYCLONEDX_CUTOFF=2026-09-29T15:45:57Z

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

# --upgrade-package exempts exactly the named packages from the preference seed
# above, so each is re-resolved to the newest release allowed by the cutoff while
# every other pin stays pinned by the seed:
#   * mcp         -> newest compatible patched 1.x SDK <= DEV_CUTOFF
#                    (1.30.0 here), resolving CVE-2026-59951;
#   * anyio       -> newest <= DEV_CUTOFF (4.15.1 in this snapshot), preserving
#                     the fixes for CVE-2026-63374/CVE-2026-64847/CVE-2026-63349;
#   * cryptography -> newest <= DEV_CUTOFF (50.0.2 here), preserving the
#                     CVE-2026-69247 fix;
#   * pip          -> 26.2.1, which resolves PYSEC-2026-3721 (seed carried
#                     26.1.2; no newer compatible release exists at this cutoff);
#   * pyjwt        -> newest <= DEV_CUTOFF (2.15.1 here), resolving the
#                     September 2026 JWT advisory set caught by pip-audit;
#   * urllib3      -> newest <= DEV_CUTOFF (2.8.0 here), resolving
#                     CVE-2026-97687/97688/97689;
#   * certifi      -> 2026.10.23, fixes PYSEC-2026-4320.
#   * packaging    -> 26.3, required by the fixed virtualenv.
#   * filelock     -> patched 4.2.4 (requires Python >=3.11); dev-only
#                    pre-commit/pip-audit are likewise scoped to >=3.11.
#   * virtualenv   -> patched 21.14.13 (fixes CVE-2026-113713).
#   * msgpack      -> patched 1.2.4 (fixes CVE-2026-114613), only in
#                     developer tooling on Python >=3.11.
# Security updates to certifi, packaging, filelock and virtualenv use
# SECURITY_PATCH_CUTOFF; msgpack uses MSGPACK_PATCH_CUTOFF. All other packages
# retain DEV_CUTOFF and the frozen base preference.
# Note that uv does NOT echo --upgrade-package into the generated header, so
# these lines are the only record of why the lock carries those versions —
# removing one would silently resolve that package back down to the seed.
uv_compile pyproject.toml \
    --extra dev \
    --python-version 3.10 \
    --universal \
    --generate-hashes \
    --exclude-newer "$DEV_CUTOFF" \
    --exclude-newer-package "certifi=$SECURITY_PATCH_CUTOFF" \
    --exclude-newer-package "filelock=$SECURITY_PATCH_CUTOFF" \
    --exclude-newer-package "packaging=$SECURITY_PATCH_CUTOFF" \
    --exclude-newer-package "virtualenv=$SECURITY_PATCH_CUTOFF" \
    --exclude-newer-package "msgpack=$MSGPACK_PATCH_CUTOFF" \
    --upgrade-package anyio \
    --upgrade-package certifi \
    --upgrade-package cryptography \
    --upgrade-package filelock \
    --upgrade-package mcp \
    --upgrade-package msgpack \
    --upgrade-package packaging \
    --upgrade-package pip \
    --upgrade-package pyjwt \
    --upgrade-package urllib3 \
    --upgrade-package virtualenv \
    --default-index https://pypi.org/simple \
    --output-file requirements-dev.lock

uv_compile requirements-build.in \
    --python-version 3.10 \
    --universal \
    --generate-hashes \
    --exclude-newer "$BUILD_RELEASE_CUTOFF" \
    --default-index https://pypi.org/simple \
    --output-file requirements-build.lock

uv_compile requirements-release.in \
    --python-version 3.10 \
    --universal \
    --generate-hashes \
    --exclude-newer "$BUILD_RELEASE_CUTOFF" \
    --exclude-newer-package "cyclonedx-bom=$CYCLONEDX_CUTOFF" \
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
