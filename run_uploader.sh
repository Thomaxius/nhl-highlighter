#!/usr/bin/env bash
# Upload a finished reel to YouTube from the repo root.
# Usage:
#   ./run_uploader.sh --file apps/shared/data/exports/reel.mp4 --title "NHL 25 Highlights"
#   ./run_uploader.sh --file reel.mp4 --title "My Game" --privacy public
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

export APP_DIR="${APP_DIR:-$ROOT}"

exec "$ROOT/.venv/bin/python" "$ROOT/apps/uploader/upload_youtube.py" "$@"
