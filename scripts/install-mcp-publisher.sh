#!/usr/bin/env bash
#
# install-mcp-publisher.sh — install a pinned, digest-verified mcp-publisher.
#
# Usage:
#   scripts/install-mcp-publisher.sh [destination-directory]   # default: .
#
# Fetches one exact release asset from the official modelcontextprotocol/registry
# releases over HTTPS, verifies its SHA-256 against the digest recorded here,
# verifies the archive contains exactly the expected members, extracts only the
# binary, and asserts the binary reports the pinned version.
#
# This script installs and verifies only. It never logs in and never publishes;
# authentication and publication stay visible in the calling workflow.
#
# Upgrading: bump VERSION, download each asset, and replace every digest below
# with the value that the release itself produces. A digest that is not derived
# from the published asset defeats the check.
#
set -euo pipefail

VERSION=1.8.0
DEST="${1:-.}"

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')"
platform="${os}_${arch}"

case "$platform" in
    linux_amd64)  expected_sha256=1370446bbe74d562608e8005a6ccce02d146a661fbd78674e11cc70b9618d6cf ;;
    linux_arm64)  expected_sha256=c978982c60e1b4903a976de090f04dc4fac4a320daa50704fcad2dbc93433d62 ;;
    darwin_amd64) expected_sha256=5350f756e8408d0e22802b7f384af941448358b503eb1e1772979a61b9b99fde ;;
    darwin_arm64) expected_sha256=e74f8846c3b5d0428cfeae3f9f520bbf9031d18e68224108c3760d60b6aaf2e0 ;;
    *)
        echo "ERROR: no pinned mcp-publisher $VERSION digest for platform $platform" >&2
        exit 1
        ;;
esac

asset="mcp-publisher_${platform}.tar.gz"
url="https://github.com/modelcontextprotocol/registry/releases/download/v${VERSION}/${asset}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo ">> downloading $url"
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
    --output "$WORK/$asset" "$url"

echo ">> verifying SHA-256"
actual_sha256="$(
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$WORK/$asset" | cut -d' ' -f1
    else
        # macOS runners ship shasum rather than GNU coreutils sha256sum.
        shasum -a 256 "$WORK/$asset" | cut -d' ' -f1
    fi
)"

if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "ERROR: digest mismatch for $asset" >&2
    echo "  expected $expected_sha256" >&2
    echo "  actual   $actual_sha256" >&2
    exit 1
fi

echo ">> verifying archive members"
expected_members="LICENSE
README.md
mcp-publisher"
actual_members="$(tar tzf "$WORK/$asset" | LC_ALL=C sort)"

if [ "$actual_members" != "$(printf '%s' "$expected_members" | LC_ALL=C sort)" ]; then
    echo "ERROR: unexpected archive members in $asset" >&2
    echo "--- expected ---" >&2
    printf '%s\n' "$expected_members" >&2
    echo "--- actual ---" >&2
    printf '%s\n' "$actual_members" >&2
    exit 1
fi

echo ">> extracting mcp-publisher into $DEST"
mkdir -p "$DEST"
tar xzf "$WORK/$asset" -C "$DEST" mcp-publisher
chmod +x "$DEST/mcp-publisher"

echo ">> asserting binary version"
# mcp-publisher writes --version to stderr through Go's log package, prefixed
# with a timestamp, so the check reads the merged stream and matches a substring.
version_output="$("$DEST/mcp-publisher" --version 2>&1)"
case "$version_output" in
    *"mcp-publisher $VERSION"*) ;;
    *)
        echo "ERROR: expected mcp-publisher $VERSION" >&2
        printf '%s\n' "$version_output" >&2
        exit 1
        ;;
esac

echo ">> mcp-publisher $VERSION installed and verified ($platform)"
