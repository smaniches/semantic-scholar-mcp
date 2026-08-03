#!/usr/bin/env bash
# Install one verified mcp-publisher release for the Linux release runner.
set -euo pipefail

VERSION="1.8.0"
ARCHIVE_SHA256="1370446bbe74d562608e8005a6ccce02d146a661fbd78674e11cc70b9618d6cf"
DESTINATION="${1:-.}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "ERROR: the pinned publisher artifact is Linux amd64 only." >&2
    exit 1
fi

mkdir -p "$DESTINATION"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ARCHIVE="$WORK/mcp-publisher_linux_amd64.tar.gz"
URL="https://github.com/modelcontextprotocol/registry/releases/download/v${VERSION}/mcp-publisher_linux_amd64.tar.gz"

curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
    "$URL" --output "$ARCHIVE"
printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE" | sha256sum --check --status

members="$(tar -tzf "$ARCHIVE")"
if [[ "$members" != "mcp-publisher" ]]; then
    echo "ERROR: unexpected files in mcp-publisher archive:" >&2
    printf '%s\n' "$members" >&2
    exit 1
fi

tar -xzf "$ARCHIVE" -C "$WORK" mcp-publisher
install -m 0755 "$WORK/mcp-publisher" "$DESTINATION/mcp-publisher"

version_output="$("$DESTINATION/mcp-publisher" --version 2>&1)"
case "$version_output" in
    "mcp-publisher ${VERSION} "*) ;;
    *)
        echo "ERROR: unexpected mcp-publisher version: $version_output" >&2
        exit 1
        ;;
esac
printf '%s\n' "$version_output"
