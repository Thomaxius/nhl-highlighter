#!/usr/bin/env bash
# Upload a finished reel to YouTube from the repo root.
# Usage:
#   ./run_uploader.sh --file apps/shared/data/exports/reel.mp4 --title "NHL 25 Highlights"
#   ./run_uploader.sh --file reel.mp4 --title "My Game" --privacy public
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$([ -d "$ROOT/.venv" ] && echo "$ROOT/.venv/bin/python" || echo "$ROOT/venv/bin/python")"

export APP_DIR="${APP_DIR:-$ROOT}"

exec "$PYTHON" "$ROOT/apps/uploader/upload_youtube.py" "$@"
