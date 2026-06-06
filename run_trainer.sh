#!/usr/bin/env bash
# Run the VideoMAE fine-tuning trainer from the repo root.
# Usage:
#   ./run_trainer.sh
#   ./run_trainer.sh --data_dir apps/shared/data/labeled --epochs 20
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$([ -d "$ROOT/.venv" ] && echo "$ROOT/.venv/bin/python" || echo "$ROOT/venv/bin/python")"

exec "$PYTHON" -m apps.trainer.src.training.trainer "$@"
