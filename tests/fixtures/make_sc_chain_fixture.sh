#!/usr/bin/env bash
#
# Rebuild tests/fixtures/sc_chain/{lead,follow}.mp4 — two arbitrary short clips
# used by tests/test_chance_chaining.py to drive build_reel's scoring_chance
# chaining branch end-to-end. Content doesn't matter (the test supplies its
# own trim_start_s/trim_end_s and confidence — these just need to be real,
# decodable video files a couple of seconds apart in length).
#
# Usage:
#   tests/fixtures/make_sc_chain_fixture.sh /path/to/NHL\ 25_20260903192725_norm.mp4
#
set -euo pipefail

SRC="${1:?path to a normalised recording required (any one will do)}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sc_chain"
mkdir -p "$OUT_DIR"

ffmpeg -y -v error -ss 330 -to 336 -i "$SRC" -c:v libx264 -crf 23 -preset veryfast -an "$OUT_DIR/lead.mp4"
ffmpeg -y -v error -ss 340 -to 344 -i "$SRC" -c:v libx264 -crf 23 -preset veryfast -an "$OUT_DIR/follow.mp4"
echo "wrote $OUT_DIR/lead.mp4 and $OUT_DIR/follow.mp4"
