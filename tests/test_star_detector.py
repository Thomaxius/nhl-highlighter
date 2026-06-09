"""
tests/test_star_detector.py

Integration test for the Three Stars screen detector.

Usage (from repo root):
    python tests/test_star_detector.py [--video PATH] [--configs PATH]

Defaults to the pre-cropped last-2-min clip:
    data/stats_frames/eqh-IQga1CM_last2min.mp4

Expected output for that clip:
    ⭐⭐⭐ FIRST  Martin Necas     GOALS: 2 | ASSISTS: 0 | HITS: 8   t=52.0s
    ⭐⭐   SECOND Matthew Tkachuk  GOALS: 1 | ASSISTS: 1 | HITS: 8   t=37.0s
    ⭐     THIRD  Sam Chow         GOALS: 1 | ASSISTS: 1 | HITS: 3   t=27.0s
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.reel_builder.src.detection.star_detector import (
    detect_star_screens,
    StarScreen,
)

_RANK_STARS = {"FIRST": "⭐⭐⭐", "SECOND": "⭐⭐  ", "THIRD": "⭐    "}
_RANK_ORDER = ["FIRST", "SECOND", "THIRD"]


def main():
    ap = argparse.ArgumentParser(description="Three Stars detector integration test")
    ap.add_argument(
        "--video",
        default="data/stats_frames/eqh-IQga1CM_last2min.mp4",
        help="Path to video clip to scan",
    )
    ap.add_argument(
        "--configs",
        default="apps/reel_builder/configs",
        help="Path to configs directory containing star templates",
    )
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: video not found: {video}", file=sys.stderr)
        sys.exit(1)

    configs = Path(args.configs)
    if not configs.exists():
        print(f"ERROR: configs dir not found: {configs}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {video}  ({video.stat().st_size / 1e6:.1f} MB)")
    print(f"Configs:  {configs}")
    print()

    stars = detect_star_screens(video, configs_dir=configs)

    if not stars:
        print("No star screens detected.")
        return

    by_rank = {s.rank: s for s in stars}

    print(f"{'':5}  {'RANK':<7}  {'NAME':<20}  {'STATS':<35}  {'TIME':>7}")
    print("-" * 80)
    for rank in _RANK_ORDER:
        star = by_rank.get(rank)
        if star is None:
            continue
        emoji = _RANK_STARS.get(rank, "⭐")
        name = (star.player_name or "(unknown)").ljust(20)
        stats = (star.player_stats or "(none)").ljust(35)
        print(f"{emoji}  {rank:<7}  {name}  {stats}  t={star.timestamp_s:.1f}s")

    print()
    print(f"Total: {len(stars)} star screen(s) detected.")


if __name__ == "__main__":
    main()
