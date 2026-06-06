#!/usr/bin/env bash
# Run the reel builder pipeline from the repo root.
# Usage:
#   ./run_reel_builder.sh --file apps/shared/data/mygame.mp4
#   ./run_reel_builder.sh --folder apps/shared/data/raw
#   ./run_reel_builder.sh --file mygame.mp4 --output apps/shared/data/exports/reel.mp4
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$([ -d "$ROOT/.venv" ] && echo "$ROOT/.venv/bin/python" || echo "$ROOT/venv/bin/python")"

exec "$PYTHON" "$ROOT/apps/reel_builder/pipeline.py" "$@"
