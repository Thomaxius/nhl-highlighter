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

# Load watcher.env if present and not already set by the environment (e.g. systemd)
ENV_FILE="${ROOT}/config/watcher.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

export APP_DIR="${APP_DIR:-$ROOT}"
export OAUTH_CLIENT_SECRETS="${OAUTH_CLIENT_SECRETS:-$ROOT/config/client_secrets.json}"
export OAUTH_TOKEN_FILE="${OAUTH_TOKEN_FILE:-$ROOT/config/token.json}"

exec "$PYTHON" "$ROOT/apps/poller/youtube_watcher.py" "$@"
