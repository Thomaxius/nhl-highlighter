#!/usr/bin/env bash
# Run the reel builder pipeline from the repo root.
# Usage:
#   ./run_reel_builder.sh --file apps/shared/data/mygame.mp4
#   ./run_reel_builder.sh --folder apps/shared/data/raw
#   ./run_reel_builder.sh --file mygame.mp4 --output apps/shared/data/exports/reel.mp4
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$([ -d "$ROOT/.venv" ] && echo "$ROOT/.venv/bin/python" || echo "$ROOT/venv/bin/python")"

# Load watcher.env if present (gives token-check paths the same env as the poller).
ENV_FILE="${ROOT}/config/watcher.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Preflight: abort if disk is low or OAuth tokens need reauth.
# (Running locally without tokens? Skip with PREFLIGHT_SKIP_TOKENS=1.)
# shellcheck source=scripts/preflight.sh
source "$ROOT/scripts/preflight.sh"
preflight "$ROOT" "$PYTHON"

exec "$PYTHON" "$ROOT/apps/reel_builder/pipeline.py" "$@"
