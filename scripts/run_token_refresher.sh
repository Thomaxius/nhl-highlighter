#!/usr/bin/env bash
# Refresh the YouTube uploader OAuth token from the repo root.
# Pass --interval N to run as a long-lived daemon; default is single-shot
# (intended to be driven by cron — see README: "Keeping the OAuth token alive").
#
# Env vars:
#   OAUTH_CLIENT_SECRETS  (default: config/client_secrets.json)
#   APP_DIR               (default: current directory)
set -euo pipefail

# This script lives in scripts/, so the repo root is one level up.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$([ -d "$ROOT/.venv" ] && echo "$ROOT/.venv/bin/python" || echo "$ROOT/venv/bin/python")"

ENV_FILE="${ROOT}/config/watcher.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

export APP_DIR="${APP_DIR:-$ROOT}"
export OAUTH_CLIENT_SECRETS="${OAUTH_CLIENT_SECRETS:-$ROOT/config/client_secrets.json}"

exec "$PYTHON" "$ROOT/apps/auth/refresh_tokens.py" "$@"
