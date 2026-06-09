"""
tests/test_stats_detector.py

Manual integration test for the stats-screen detector.

Usage (from repo root):
    python tests/test_stats_detector.py [--video PATH]

Defaults to the pre-cropped last-2-min clip:
    data/stats_frames/eqh-IQga1CM_last2min.mp4

Prints a postgres-style table for each detected team/tab combination.
Expected output: up to 4 tables —
    <Team A> / ALL SKATERS
    <Team A> / GOALIES
    <Team B> / ALL SKATERS
    <Team B> / GOALIES
(only the tabs the user actually browsed will appear)
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.reel_builder.src.detection.stats_detector import (
    detect_stats_screens,
    StatsScreen,
)


# ── Pretty-printer ────────────────────────────────────────────────────────────

def _pg_table(screen: StatsScreen) -> str:
    """Format a StatsScreen as a postgres-style ASCII table."""
    rows = screen.rows
    if not rows:
        return f"  (no rows parsed)\n"

    headers = list(rows[0].keys())

    # Column widths = max of header length and widest cell value
    widths = [
        max(len(h), max((len(r.get(h, "")) for r in rows), default=0))
        for h in headers
    ]

    def sep(l="+", m="+", r="+", f="-"):
        return l + m.join(f * (w + 2) for w in widths) + r

    def row_str(cells):
        return "| " + " | ".join(
            (cells[i] if i < len(cells) else "").ljust(widths[i])
            for i in range(len(headers))
        ) + " |"

    lines = [
        sep(),
        row_str(headers),
        sep("|", "+", "|", "="),
    ]
    for r in rows:
        lines.append(row_str([r.get(h, "") for h in headers]))
    lines.append(sep())
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Stats detector integration test")
    ap.add_argument(
        "--video",
        default="data/stats_frames/eqh-IQga1CM_last2min.mp4",
        help="Path to video clip to scan (default: last-2-min crop)",
    )
    ap.add_argument(
        "--scan-window", type=float, default=300.0,
        help="How many seconds from the end to scan (default: 300)",
    )
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: video not found: {video}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {video}  ({video.stat().st_size / 1e6:.1f} MB)")
    print()

    screens = detect_stats_screens(video, scan_from_end_s=args.scan_window)

    if not screens:
        print("No stats screens detected.")
        return

    # Group by team so we print Team A's tables together, then Team B's
    teams: dict[str, list[StatsScreen]] = {}
    for s in screens:
        teams.setdefault(s.team or "UNKNOWN TEAM", []).append(s)

    for team, team_screens in teams.items():
        print("=" * 70)
        print(f"  {team}")
        print("=" * 70)
        for screen in team_screens:
            n_rows = len(screen.rows)
            print(f"\n  {screen.tab}  (t={screen.timestamp:.1f}s, {n_rows} row(s))\n")
            print(_pg_table(screen))
        print()

    print(f"Total: {len(screens)} screen(s) across {len(teams)} team(s).")


if __name__ == "__main__":
    main()
