"""
scripts/collect_scoring_chances.py

Scans a processed segments folder in chronological order, detects when the
SOG (shots on goal) counter increments in the HUD, and copies those clips to
data/labeled/scoring_chance/ for use as training data.

Each output clip is assembled as:
    [last PRE_SHOT_S seconds of the previous segment] + [first POST_SHOT_S seconds of the SOG-change segment]

This gives natural context leading into the shot and a brief moment after.

Usage
-----
    python scripts/collect_scoring_chances.py \
        --segments_dir "data/processed/segments/NHL 25_20260225202819_norm" \
        --output_dir   data/labeled/scoring_chance

    # Dry run to see what would be collected without copying:
    python scripts/collect_scoring_chances.py --dry_run

    # Also collect goal clips (when score increments):
    python scripts/collect_scoring_chances.py --include_goals

    # Adjust window size:
    python scripts/collect_scoring_chances.py --pre_shot 6 --post_shot 1.5

Notes
-----
- Requires: pip install pytesseract && brew install tesseract
- Run --debug_roi first to verify the scoreboard crop looks correct:
    python scripts/collect_scoring_chances.py --debug_roi segment.mp4
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_scoring_chances")


def get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_window(
    prev_seg: Path | None,
    curr_seg: Path,
    output_path: Path,
    pre_shot_s: float = 6.0,
    post_shot_s: float = 2.0,
) -> bool:
    """
    Concatenate [last pre_shot_s of prev_seg] + [first post_shot_s of curr_seg]
    into output_path using ffmpeg. Returns True on success.
    If prev_seg is None, just copies the first post_shot_s of curr_seg.
    """
    with tempfile.TemporaryDirectory() as tmp:
        parts: list[Path] = []

        if prev_seg is not None:
            dur = get_duration(prev_seg)
            start = max(0.0, dur - pre_shot_s)
            part_a = Path(tmp) / "part_a.mp4"
            r = subprocess.run([
                "ffmpeg", "-y", "-ss", str(start), "-i", str(prev_seg),
                "-c", "copy", str(part_a),
            ], capture_output=True)
            if r.returncode == 0:
                parts.append(part_a)

        part_b = Path(tmp) / "part_b.mp4"
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(curr_seg),
            "-t", str(post_shot_s), "-c", "copy", str(part_b),
        ], capture_output=True)
        if r.returncode == 0:
            parts.append(part_b)

        if not parts:
            return False

        if len(parts) == 1:
            import shutil
            shutil.copy2(parts[0], output_path)
            return True

        # Write concat list file
        list_file = Path(tmp) / "list.txt"
        list_file.write_text("\n".join(f"file '{p}'" for p in parts))

        r = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", str(output_path),
        ], capture_output=True)
        return r.returncode == 0


def collect(
    segments_dir: Path,
    output_dir: Path,
    goals_dir: Path | None = None,
    pre_shot_s: float = 6.0,
    post_shot_s: float = 2.0,
    dry_run: bool = False,
) -> None:
    from src.detection.score_detector import ScoreDetector

    segments_dir = Path(segments_dir)
    output_dir = Path(output_dir)

    # Collect all segment subdirs or use the given dir directly
    if any(segments_dir.glob("*/*.mp4")):
        all_segments: list[Path] = []
        for match_dir in sorted(segments_dir.iterdir()):
            if match_dir.is_dir():
                all_segments.extend(sorted(match_dir.glob("*.mp4")))
    else:
        all_segments = sorted(segments_dir.glob("*.mp4"))

    if not all_segments:
        logger.error("No .mp4 files found in %s", segments_dir)
        return

    logger.info("Processing %d segments from %s", len(all_segments), segments_dir)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if goals_dir:
            goals_dir.mkdir(parents=True, exist_ok=True)

    detector = ScoreDetector(sample_frames=6)
    shots_found = 0
    goals_found = 0
    last_match = None
    prev_seg: Path | None = None

    for seg in all_segments:
        match_folder = seg.parent
        if match_folder != last_match:
            if last_match is not None:
                logger.info("Resetting state for new match: %s", match_folder.name)
                detector.reset()
                prev_seg = None
            last_match = match_folder

        result = detector.detect_in_segment(seg)

        if result["score_changed"]:
            goals_found += 1
            dest_dir = goals_dir or output_dir
            dest = dest_dir / f"{seg.stem}_goal{seg.suffix}"
            logger.info("  GOAL    → %s  (score: %s-%s)", seg.name, result["away"], result["home"])
            if not dry_run:
                import shutil
                shutil.copy2(seg, dest)

        elif result["sog_changed"]:
            n = result.get("sog_change_count", 1)
            shots_found += 1
            dest = output_dir / f"{seg.stem}_scoring_chance{seg.suffix}"
            logger.info(
                "  SHOT x%d → %s  (SOG: %s-%s)",
                n, seg.name, result.get("sog_away"), result.get("sog_home"),
            )
            if not dry_run:
                import shutil
                shutil.copy2(seg, dest)

        prev_seg = seg

    logger.info("")
    logger.info("Done%s:", " (dry run)" if dry_run else "")
    logger.info("  Scoring chance clips: %d", shots_found)
    logger.info("  Goal clips:           %d", goals_found)
    if not dry_run:
        logger.info("  Output: %s", output_dir)


def main():
    p = argparse.ArgumentParser(description="Auto-collect scoring chance clips via SOG detection")
    p.add_argument("--segments_dir", default="data/processed/segments",
                   help="Folder containing segment .mp4 files (or match subfolders)")
    p.add_argument("--output_dir", default="data/labeled/scoring_chance",
                   help="Where to copy detected shot-on-goal clips")
    p.add_argument("--include_goals", action="store_true",
                   help="Also copy goal clips (score change) to data/labeled/goal/")
    p.add_argument("--pre_shot", type=float, default=6.0,
                   help="Seconds to include before the shot (from end of previous segment). Default 6.")
    p.add_argument("--post_shot", type=float, default=2.0,
                   help="Seconds to include after the shot (start of SOG-change segment). Default 2.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be collected without copying")
    p.add_argument("--debug_roi", metavar="VIDEO_PATH",
                   help="Save ROI crop debug image for a single video and exit")
    args = p.parse_args()

    if args.debug_roi:
        from src.detection.score_detector import ScoreDetector
        ScoreDetector().debug_frame(args.debug_roi)
        return

    goals_dir = Path("data/labeled/goal") if args.include_goals else None

    collect(
        segments_dir=Path(args.segments_dir),
        output_dir=Path(args.output_dir),
        goals_dir=goals_dir,
        pre_shot_s=args.pre_shot,
        post_shot_s=args.post_shot,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
