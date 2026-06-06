#!/usr/bin/env bash
# Run the YouTube poller from the repo root.
# Required env vars:
#   YOUTUBE_CHANNEL_ID
#   OAUTH_CLIENT_SECRETS  (default: config/client_secrets.json)
#   OAUTH_TOKEN_FILE      (default: config/token.json)
#   APP_DIR               (default: current directory)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$([ -d "$ROOT/.venv" ] && echo "$ROOT/.venv/bin/python" || echo "$ROOT/venv/bin/python")"

export APP_DIR="${APP_DIR:-$ROOT}"
export OAUTH_CLIENT_SECRETS="${OAUTH_CLIENT_SECRETS:-$ROOT/config/client_secrets.json}"
export OAUTH_TOKEN_FILE="${OAUTH_TOKEN_FILE:-$ROOT/config/token.json}"

exec "$PYTHON" "$ROOT/apps/poller/youtube_watcher.py" "$@"
