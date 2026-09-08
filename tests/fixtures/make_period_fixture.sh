#!/usr/bin/env bash
#
# Rebuild tests/fixtures/period_detection_20260903192725.mp4 — the regression
# fixture for game-clock period / overtime detection (see tests/test_period_detection.py).
#
# It splices six short 1080p windows out of the normalised recording
# "NHL 25_20260903192725_norm.mp4", each covering a spot that used to fool the
# detector:
#
#   A  330–356 s  — scoreboard frame that OCRs as "2OT" (false overtime buzzer)
#   B  588–604 s  — real 2nd-period puck drop (20:00 / 2ND)
#   C  916–936 s  — real 3rd-period puck drop (20:00 / 3RD, unambiguous)
#   D 1066–1086 s — mid-3rd; the 2nd- and 3rd-period clock templates both match
#                   high on this frame (period-template tie-break)
#   E 1372–1392 s — a "1:00" clock OCR'd as "20:00 / 3RD" (false period start)
#   F 1480–1500 s — real end-of-regulation buzzer (kept last for the reverse scan)
#
# Re-encodes (does not stream-copy) so every window keeps a real, monotonic
# timeline — scan_video maps frame index → seconds and needs that.
#
# Usage:
#   tests/fixtures/make_period_fixture.sh /path/to/NHL\ 25_20260903192725_norm.mp4
#
set -euo pipefail

SRC="${1:?path to NHL 25_20260903192725_norm.mp4 required}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$OUT_DIR/period_detection_20260903192725.mp4"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

: > "$TMP/list.txt"
i=0
cut() {  # start end
    local part
    part="$(printf '%s/part_%02d.mp4' "$TMP" "$i")"
    ffmpeg -y -v error -ss "$1" -to "$2" -i "$SRC" \
        -c:v libx264 -crf 19 -preset medium -an -pix_fmt yuv420p \
        -vf "fps=30,setsar=1" "$part"
    echo "file '$part'" >> "$TMP/list.txt"
    i=$((i + 1))
}

cut 330 356
cut 588 604
cut 916 936
cut 1066 1086
cut 1372 1392
cut 1480 1500

ffmpeg -y -v error -f concat -safe 0 -i "$TMP/list.txt" \
    -c:v libx264 -crf 19 -preset medium -pix_fmt yuv420p "$OUT"
echo "wrote $OUT"
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT"
