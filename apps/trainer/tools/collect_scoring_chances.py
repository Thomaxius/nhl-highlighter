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

    # Point it at a single raw/normalised match file instead of a pre-split
    # segments folder, and it scans the file directly (sampling every ~1s)
    # instead of requiring a PySceneDetect split first:
    python scripts/collect_scoring_chances.py \
        --segments_dir "data/processed/NHL 25_20260903201224_norm.mp4" \
        --output_dir   data/labeled/scoring_chance

    # Dry run to see what would be collected without copying:
    python scripts/collect_scoring_chances.py --dry_run

    # Also collect goal clips (when score increments):
    python scripts/collect_scoring_chances.py --include_goals

    # Adjust window size:
    python scripts/collect_scoring_chances.py --pre_shot 3 --post_shot 2

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

# score_detector.py lives under apps/reel_builder, a sibling app — not under
# apps/trainer (this script's own parent). parents[1] here is apps/trainer.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reel_builder"))

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
    shot_time: float = 0.0,
    pre_shot_s: float = 3.0,
    post_shot_s: float = 2.0,
) -> bool:
    """
    Extract a clip centred on *shot_time* (seconds into curr_seg):

        [shot_time - pre_shot_s  ..  shot_time + post_shot_s]

    If the pre-window extends before the start of curr_seg, the deficit is
    drawn from the tail of prev_seg.  Returns True on success.
    """
    with tempfile.TemporaryDirectory() as tmp:
        parts: list[Path] = []

        # How much of the pre-window lies before curr_seg starts?
        curr_pre_start = max(0.0, shot_time - pre_shot_s)
        pre_shortage   = max(0.0, pre_shot_s - shot_time)  # seconds needed from prev_seg

        if prev_seg is not None and pre_shortage > 0:
            prev_dur = get_duration(prev_seg)
            start = max(0.0, prev_dur - pre_shortage)
            part_a = Path(tmp) / "part_a.mp4"
            r = subprocess.run([
                "ffmpeg", "-y", "-ss", str(start), "-i", str(prev_seg),
                "-t", str(pre_shortage), "-c", "copy", str(part_a),
            ], capture_output=True)
            if r.returncode == 0:
                parts.append(part_a)

        part_b = Path(tmp) / "part_b.mp4"
        clip_duration = (shot_time + post_shot_s) - curr_pre_start
        r = subprocess.run([
            "ffmpeg", "-y", "-ss", str(curr_pre_start), "-i", str(curr_seg),
            "-t", str(clip_duration), "-c", "copy", str(part_b),
        ], capture_output=True)
        if r.returncode == 0:
            parts.append(part_b)

        if not parts:
            return False

        if len(parts) == 1:
            shutil.copy2(str(parts[0]), str(output_path))
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
    pre_shot_s: float = 3.0,
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

    # 12 samples gives ~3s timestamp resolution on a typical ~30-40s segment —
    # important now that the clip window (pre_shot/post_shot) is only a few
    # seconds wide, so a coarse shot_time estimate can run the window off the
    # end of the segment. (Pushing this higher, e.g. 24, surfaces single-frame
    # OCR misreads more often — the score aggregator keeps the last successful
    # read rather than a majority vote, so one bad frame can flip the detected
    # score. 12 stayed clean in testing; revisit alongside a majority-vote
    # aggregator if more resolution is needed later.)
    detector = ScoreDetector(sample_frames=12)
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
                shutil.copy2(seg, dest)

        # Not elif: score_changed uses the noisier team-name+score OCR crop and
        # can misfire on a single bad frame (last-successful-read aggregation,
        # no majority vote). A false goal read should never suppress a real
        # SOG-based scoring-chance detection.
        if result["sog_changed"]:
            shot_timestamps = result.get("shot_timestamps") or [0.0]
            for shot_i, shot_time in enumerate(shot_timestamps):
                shots_found += 1
                clip_label = f"_shot{shot_i + 1:02d}" if len(shot_timestamps) > 1 else "_scoring_chance"
                dest = output_dir / f"{seg.stem}{clip_label}{seg.suffix}"
                logger.info(
                    "  SHOT %d/%d → %s  @t=%.1fs  (SOG: %s-%s)",
                    shot_i + 1, len(shot_timestamps), seg.name, shot_time,
                    result.get("sog_away"), result.get("sog_home"),
                )
                if not dry_run:
                    ok = extract_window(
                        prev_seg, seg, dest,
                        shot_time=shot_time,
                        pre_shot_s=pre_shot_s,
                        post_shot_s=post_shot_s,
                    )
                    if not ok:
                        logger.warning("    ffmpeg failed for %s — falling back to full copy", dest.name)
                        shutil.copy2(seg, dest)

        prev_seg = seg

    logger.info("")
    logger.info("Done%s:", " (dry run)" if dry_run else "")
    logger.info("  Scoring chance clips: %d", shots_found)
    logger.info("  Goal clips:           %d", goals_found)
    if not dry_run:
        logger.info("  Output: %s", output_dir)


def extract_clip(src: Path, dest: Path, start_s: float, duration_s: float) -> bool:
    """
    Cut [start_s, start_s + duration_s] straight out of *src*. Unlike
    extract_window(), there's no stitching here — collect_from_raw() scans
    one continuous file, so a clip window never needs content from a
    "previous segment"; it's all on the same timeline.
    """
    start_s = max(0.0, start_s)
    r = subprocess.run([
        "ffmpeg", "-y", "-ss", str(start_s), "-i", str(src),
        "-t", str(duration_s), "-c", "copy", str(dest),
    ], capture_output=True)
    return r.returncode == 0


def collect_from_raw(
    video_path: Path,
    output_dir: Path,
    goals_dir: Path | None = None,
    pre_shot_s: float = 3.0,
    post_shot_s: float = 2.0,
    sample_interval_s: float = 1.0,
    dry_run: bool = False,
) -> None:
    """
    Scan a single un-split raw/normalised match file directly for scoring
    chances and goals — no PySceneDetect pass needed first. See
    ScoreDetector.scan_video() for the detection side; this just turns its
    events into clips cut straight out of the source file.
    """
    from src.detection.score_detector import ScoreDetector

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if goals_dir:
            goals_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Scanning %s directly (sampling every %.1fs, no scene split needed)",
        video_path.name, sample_interval_s,
    )
    detector = ScoreDetector(sample_frames=12)
    events = detector.scan_video(video_path, sample_interval_s=sample_interval_s)

    shots_found = 0
    goals_found = 0
    for ev in events:
        is_goal = ev["type"] == "goal"
        dest_dir = (goals_dir or output_dir) if is_goal else output_dir
        label = "goal" if is_goal else "scoring_chance"
        # Timestamp baked into the filename since there's no segment stem to
        # key off — multiple events from the same source file need distinct
        # unambiguous names.
        dest = dest_dir / f"{video_path.stem}_t{ev['time']:07.1f}_{label}{video_path.suffix}"

        if is_goal:
            goals_found += 1
            logger.info("  GOAL  → %s  @t=%.1fs  (score: %s-%s)", dest.name, ev["time"], ev["away"], ev["home"])
        else:
            shots_found += 1
            logger.info("  SHOT  → %s  @t=%.1fs  (SOG: %s-%s)", dest.name, ev["time"], ev["sog_away"], ev["sog_home"])

        if not dry_run:
            ok = extract_clip(video_path, dest, ev["time"] - pre_shot_s, pre_shot_s + post_shot_s)
            if not ok:
                logger.warning("    ffmpeg failed for %s", dest.name)

    logger.info("")
    logger.info("Done%s:", " (dry run)" if dry_run else "")
    logger.info("  Scoring chance clips: %d", shots_found)
    logger.info("  Goal clips:           %d", goals_found)
    if not dry_run:
        logger.info("  Output: %s", output_dir)


def main():
    p = argparse.ArgumentParser(description="Auto-collect scoring chance clips via SOG detection")
    p.add_argument("--segments_dir", default="data/processed/segments",
                   help="Folder containing segment .mp4 files (or match subfolders), "
                        "OR a single raw/normalised match .mp4 — if this points to a "
                        "file rather than a folder, it's scanned directly with no "
                        "scene-splitting pass required.")
    p.add_argument("--output_dir", default="data/labeled/scoring_chance",
                   help="Where to copy detected shot-on-goal clips")
    p.add_argument("--include_goals", action="store_true",
                   help="Also copy goal clips (score change) to data/labeled/goal/")
    p.add_argument("--pre_shot", type=float, default=3.0,
                   help="Seconds to include before the shot (from end of previous segment). Default 3.")
    p.add_argument("--post_shot", type=float, default=2.0,
                   help="Seconds to include after the shot (start of SOG-change segment). Default 2.")
    p.add_argument("--sample_interval", type=float, default=1.0,
                   help="Raw-file mode only: seconds between OCR samples while scanning "
                        "the whole match. Default 1.0.")
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
    target = Path(args.segments_dir)

    if target.is_file():
        collect_from_raw(
            video_path=target,
            output_dir=Path(args.output_dir),
            goals_dir=goals_dir,
            pre_shot_s=args.pre_shot,
            post_shot_s=args.post_shot,
            sample_interval_s=args.sample_interval,
            dry_run=args.dry_run,
        )
        return

    collect(
        segments_dir=target,
        output_dir=Path(args.output_dir),
        goals_dir=goals_dir,
        pre_shot_s=args.pre_shot,
        post_shot_s=args.post_shot,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
