#!/usr/bin/env bash
# Run the VideoMAE fine-tuning trainer from the repo root.
# Usage:
#   ./run_trainer.sh
#   ./run_trainer.sh --data_dir apps/shared/data/labeled --epochs 20
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

exec "$ROOT/.venv/bin/python" -m apps.trainer.src.training.trainer "$@"
