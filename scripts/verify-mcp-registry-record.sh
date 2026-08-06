#!/usr/bin/env bash
# Validate server.json and verify its immutable Official MCP Registry record.
#
# Required environment:
#   EXPECTED_SERVER_NAME
#   EXPECTED_PACKAGE_ID
#   EXPECTED_REPOSITORY_URL
#
# Usage:
#   scripts/verify-mcp-registry-record.sh [manifest]
#   scripts/verify-mcp-registry-record.sh [manifest] \
#     --base-manifest <base-manifest> --allow-missing-new-version
set -euo pipefail

manifest="${1:-server.json}"
shift || true
base_manifest=""
allow_missing_new_version=false

while (($#)); do
    case "$1" in
        --base-manifest)
            base_manifest="${2:?missing path after --base-manifest}"
            shift 2
            ;;
        --allow-missing-new-version)
            allow_missing_new_version=true
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

: "${EXPECTED_SERVER_NAME:?EXPECTED_SERVER_NAME is required}"
: "${EXPECTED_PACKAGE_ID:?EXPECTED_PACKAGE_ID is required}"
: "${EXPECTED_REPOSITORY_URL:?EXPECTED_REPOSITORY_URL is required}"

state_file="${MCP_REGISTRY_STATE_FILE:-/tmp/mcp-registry-state}"
record_file="${MCP_REGISTRY_RECORD_FILE:-/tmp/mcp-registry-record.json}"
validation_file="${MCP_REGISTRY_VALIDATION_FILE:-/tmp/mcp-registry-validation.json}"
retry_attempts="${MCP_REGISTRY_RETRY_ATTEMPTS:-1}"
retry_delay="${MCP_REGISTRY_RETRY_DELAY_SECONDS:-10}"
semver_re='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$'

[[ -f "$manifest" ]]
[[ "$retry_attempts" =~ ^[1-9][0-9]*$ ]]
[[ "$retry_delay" =~ ^[0-9]+$ ]]

server_name="$(jq -er '.name | select(type == "string" and length > 0)' "$manifest")"
server_version="$(jq -er '.version | select(type == "string" and length > 0)' "$manifest")"
package_version="$(jq -er '.packages[0].version | select(type == "string" and length > 0)' "$manifest")"

[[ "$server_name" == "$EXPECTED_SERVER_NAME" ]]
[[ "$server_version" =~ $semver_re ]]
[[ "$server_version" == "$package_version" ]]

jq -e \
    --arg name "$EXPECTED_SERVER_NAME" \
    --arg package "$EXPECTED_PACKAGE_ID" \
    --arg repository "$EXPECTED_REPOSITORY_URL" '
      .name == $name
      and .repository.url == $repository
      and .repository.source == "github"
      and (.packages | type == "array" and length == 1)
      and .packages[0].registryType == "pypi"
      and .packages[0].identifier == $package
      and .packages[0].runtimeHint == "uvx"
      and .packages[0].transport.type == "stdio"
    ' "$manifest" >/dev/null

validation_code="$(curl --silent --show-error --location --retry 4 --retry-all-errors \
    --request POST \
    --header 'Content-Type: application/json' \
    --data-binary "@$manifest" \
    --output "$validation_file" \
    --write-out '%{http_code}' \
    'https://registry.modelcontextprotocol.io/v0/validate')"

if [[ "$validation_code" != "200" ]]; then
    cat "$validation_file" >&2 || true
    echo "ERROR: Official Registry validation returned HTTP $validation_code" >&2
    exit 1
fi
jq -e '.valid == true' "$validation_file" >/dev/null

encoded_name="$(jq -rn --arg value "$server_name" '$value | @uri')"
encoded_version="$(jq -rn --arg value "$server_version" '$value | @uri')"
registry_url="https://registry.modelcontextprotocol.io/v0.1/servers/${encoded_name}/versions/${encoded_version}"

http_code=""
for ((attempt = 1; attempt <= retry_attempts; attempt++)); do
    http_code="$(curl --silent --show-error --location \
        --output "$record_file" \
        --write-out '%{http_code}' \
        "$registry_url")"
    [[ "$http_code" == "200" ]] && break
    [[ "$http_code" != "404" ]] && break
    if ((attempt < retry_attempts)); then
        echo "Registry record not available, attempt ${attempt}/${retry_attempts}; sleeping ${retry_delay}s"
        sleep "$retry_delay"
    fi
done

version_changed=false
if [[ -n "$base_manifest" && -f "$base_manifest" ]]; then
    base_version="$(jq -er '.version | select(type == "string" and length > 0)' "$base_manifest")"
    [[ "$base_version" != "$server_version" ]] && version_changed=true
fi

if [[ "$http_code" == "404" ]]; then
    printf '%s\n' missing > "$state_file"
    if [[ "$allow_missing_new_version" == true && "$version_changed" == true ]]; then
        echo "NOTICE: $server_name $server_version is valid and pending release publication."
        exit 0
    fi
    echo "ERROR: $server_name $server_version is absent from the Official MCP Registry." >&2
    exit 1
fi

if [[ "$http_code" != "200" ]]; then
    cat "$record_file" >&2 || true
    echo "ERROR: Registry lookup returned HTTP $http_code" >&2
    exit 1
fi

registry_projection='with_entries(select(.key as $key
  | ["description", "icons", "name", "packages", "remotes", "repository", "title", "version", "websiteUrl"]
  | index($key)))'

if jq -e --slurpfile expected "$manifest" \
    "(.server | $registry_projection) == (\$expected[0] | $registry_projection)" \
    "$record_file" >/dev/null; then
    printf '%s\n' exact > "$state_file"
    echo "Verified exact Official MCP Registry record: $server_name $server_version"
    exit 0
fi

manifest_changed=true
if [[ -n "$base_manifest" && -f "$base_manifest" ]]; then
    jq -S "$registry_projection" "$base_manifest" > /tmp/mcp-registry-base-projection.json
    jq -S "$registry_projection" "$manifest" > /tmp/mcp-registry-head-projection.json
    cmp -s /tmp/mcp-registry-base-projection.json /tmp/mcp-registry-head-projection.json \
        && manifest_changed=false
fi

if [[ "$allow_missing_new_version" == true && "$manifest_changed" == false ]]; then
    printf '%s\n' legacy-drift > "$state_file"
    echo "WARNING: published $server_name $server_version predates the current manifest; this change does not alter Registry-persisted metadata." >&2
    exit 0
fi

printf '%s\n' drift > "$state_file"
echo "ERROR: immutable Registry metadata differs from $manifest; publish a new version." >&2
diff -u \
    <(jq -S "$registry_projection" "$manifest") \
    <(jq -S ".server | $registry_projection" "$record_file") || true
exit 1
