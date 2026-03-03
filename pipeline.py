"""
pipeline.py  —  End-to-end highlight reel generator.

Usage
-----
    python pipeline.py --input data/raw --output data/exports/reel.mp4

The pipeline runs:
    1. Ingest & normalise clips  (FFmpeg)
    2. Split into scenes         (PySceneDetect)
    3. Audio spike detection     (librosa)
    4. Video classification      (VideoMAE fine-tune)
    5. Score & rank segments
    6. Assemble highlight reel   (FFmpeg + MoviePy)
"""

import argparse
import logging
from pathlib import Path

from src.ingestion.ingest import ingest_clips
from src.preprocessing.scene_splitter import split_into_scenes
from src.preprocessing.audio_analyzer import extract_audio, detect_energy_spikes, score_segments_by_audio
from src.detection.classifier import HighlightClassifier
from src.detection.banner_detector import BannerDetector
from src.detection.score_detector import ScoreDetector
from src.assembly.reel_builder import build_reel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _chain_goal_sequences(results: list[dict]) -> list[dict]:
    """
    After a banner-detected goal segment, look ahead and tag the immediately
    following celebration and replay segments so they are included in the reel
    in the correct order: goal → celebration → replay.

    Stops chaining when:
    - An `other` segment appears that is longer than 10s (not just a transition flash)
    - A second goal banner is detected
    - MAX_LOOKAHEAD segments have been checked
    """
    FOLLOW_LABELS = {"celebration", "replay"}
    MAX_LOOKAHEAD = 10
    MAX_GAP_S = 10.0   # stop chaining if >10s of non-highlight content

    goal_indices = [i for i, r in enumerate(results) if r.get("banner_detected")]

    for goal_idx in goal_indices:
        base_score = results[goal_idx]["score"]
        order = 1
        accumulated_gap_s = 0.0

        for j in range(1, MAX_LOOKAHEAD + 1):
            next_idx = goal_idx + j
            if next_idx >= len(results):
                break

            seg = results[next_idx]

            # Always stop at another goal
            if seg.get("banner_detected"):
                break

            if seg.get("label") in FOLLOW_LABELS:
                seg["chain_order"] = order
                seg["chain_goal_idx"] = goal_idx
                seg["score"] = base_score - (order * 0.01)
                seg["confidence"] = max(seg.get("confidence", 0.0), 0.9)
                logger.info(
                    "  Chained %s (order %d, goal %d): %s",
                    seg["label"], order, goal_idx, Path(seg["path"]).name,
                )
                order += 1
                accumulated_gap_s = 0.0  # reset gap counter once we find a highlight

            elif seg.get("label") == "other":
                # Measure this segment's duration — short ones are EA transitions, skip over them
                duration = _get_clip_duration(seg["path"])
                if duration <= 3.0:
                    logger.info("  Skipping short transition (%.1fs): %s", duration, Path(seg["path"]).name)
                    continue
                accumulated_gap_s += duration
                if accumulated_gap_s > MAX_GAP_S:
                    logger.info("  Chain stopped — %.1fs gap exceeded 10s limit", accumulated_gap_s)
                    break

    return results


def _get_clip_duration(video_path) -> float:
    """Return the duration of a video clip in seconds."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps



def run_pipeline(
    input_dir: str,
    output_path: str,
    checkpoint: str = "models/checkpoints/videomae_nhl",
    music_path: str | None = None,
    max_clips: int = 10,
    min_confidence: float = 0.55,
) -> None:
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    processed_dir = Path("data/processed")
    segments_dir = Path("data/processed/segments")
    audio_dir = Path("data/processed/audio")

    # ── Step 1: Ingest & normalise ───────────────────────────────────────────
    logger.info("━━━  Step 1: Ingesting clips  ━━━")
    normalised = ingest_clips(input_dir, processed_dir)
    logger.info("Normalised %d clips.", len(normalised))

    # ── Step 2: Scene splitting ──────────────────────────────────────────────
    logger.info("━━━  Step 2: Splitting scenes  ━━━")
    all_segments: list[Path] = []
    for clip in normalised:
        segs = split_into_scenes(clip, segments_dir / clip.stem)
        all_segments.extend(segs)
    logger.info("Total segments: %d", len(all_segments))

    # ── Step 3: Audio analysis ───────────────────────────────────────────────
    logger.info("━━━  Step 3: Audio analysis  ━━━")
    all_spikes = []
    for clip in normalised:
        wav_path = audio_dir / (clip.stem + ".wav")
        try:
            extract_audio(clip, wav_path)
            spikes = detect_energy_spikes(wav_path)
            all_spikes.extend(spikes)
        except Exception as e:
            logger.warning("Audio analysis failed for %s: %s", clip.name, e)

    # ── Step 4: Video classification ─────────────────────────────────────────
    logger.info("━━━  Step 4: Classifying segments  ━━━")
    classifier = HighlightClassifier(checkpoint_path=checkpoint)
    results = classifier.classify_segments(all_segments)

    # ── Step 4b: Banner detection (GOAL! HUD overlay) ───────────────────────
    # Auto-discovers all configs/goal_banner_template*.png — no path needed.
    templates = sorted(Path("configs").glob("goal_banner_template*.png"))
    if templates:
        logger.info("━━━  Step 4b: Banner detection (%d template(s))  ━━━", len(templates))
        detector = BannerDetector()  # auto-loads all matching templates
        results = detector.score_segments(results)
    else:
        logger.info("Skipping banner detection (no templates found in configs/)")

    # ── Step 4c: Banner is the ground truth — hard override ──────────────────
    # If the GOAL! banner was detected, the segment IS a goal, period.
    # No ML confidence threshold can override a pixel-level HUD match.
    # Conversely, demote any 'goal' label that has no banner and low confidence.
    logger.info("━━━  Step 4c: Banner-gating goal labels  ━━━")
    demoted = 0
    promoted = 0
    for r in results:
        if r.get("banner_detected"):
            # Hard override regardless of what the classifier said
            if r.get("label") != "goal":
                logger.info(
                    "  Banner override → goal: %s (was %s, conf=%.0f%%)",
                    Path(r["path"]).name, r.get("label"), r.get("confidence", 0) * 100,
                )
                promoted += 1
            r["label"] = "goal"
            r["confidence"] = 1.0
        elif r.get("label") == "goal" and r.get("confidence", 0) < 0.92:
            logger.info(
                "  Demoted → other: %s (conf=%.0f%%, no banner)",
                Path(r["path"]).name, r.get("confidence", 0) * 100,
            )
            r["label"] = "other"
            demoted += 1
    if promoted:
        logger.info("  Promoted %d segment(s) to goal via banner detection.", promoted)
    if demoted:
        logger.info("  Demoted %d goal segment(s) without banner confirmation.", demoted)

    # ── Step 4e: Score change detection (OCR) ────────────────────────────────
    logger.info("━━━  Step 4e: Score change detection  ━━━")
    score_detector = ScoreDetector(sample_frames=6)
    results = score_detector.score_segments(results)

    # ── Step 4f: Chain goal → celebration → replay ───────────────────────────
    logger.info("━━━  Step 4f: Chaining goal sequences  ━━━")
    results = _chain_goal_sequences(results)

    # ── Step 5: Score with audio boosts ──────────────────────────────────────
    logger.info("━━━  Step 5: Scoring  ━━━")
    for r in results:
        if "score" not in r:
            r["score"] = r["confidence"]
        logger.info("  %-45s  label=%-8s  conf=%.0f%%  banner=%s",
                    Path(r["path"]).name, r["label"], r["confidence"] * 100,
                    r.get("banner_detected", False))

    # Attach rough timing from filename index (real app would parse FFmpeg metadata)
    results = score_segments_by_audio(results, all_spikes)

    # ── Step 6: Assemble reel ────────────────────────────────────────────────
    logger.info("━━━  Step 6: Building reel  ━━━")
    build_reel(
        segments=results,
        output_path=output_path,
        music_path=music_path,
        max_clips=max_clips,
        min_confidence=min_confidence,
    )
    logger.info("Done!  Reel → %s", output_path)


def main():
    p = argparse.ArgumentParser(description="NHL 25 Highlight Reel Generator")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--file",   metavar="FILE",   help="Single raw .mp4 to process")
    src.add_argument("--folder", metavar="FOLDER", help="Folder of raw .mp4 clips (default: data/raw)")

    p.add_argument("--output", default="data/exports/reel.mp4", help="Output reel path")
    p.add_argument("--checkpoint", default="models/checkpoints/videomae_nhl")
    p.add_argument("--music", default=None, help="Optional background music file")
    p.add_argument("--max_clips", type=int, default=10)
    p.add_argument("--min_confidence", type=float, default=0.55)
    args = p.parse_args()

    if args.file:
        # Wrap the single file in a temp dir so the pipeline's ingest step
        # sees it as a one-file folder without touching the original.
        import tempfile, os
        file_path = Path(args.file).resolve()
        tmp = tempfile.mkdtemp(prefix="pipeline_single_")
        os.link(file_path, Path(tmp) / file_path.name)   # hardlink — instant, no extra disk
        input_dir = tmp
    else:
        input_dir = args.folder or "data/raw"

    run_pipeline(
        input_dir=input_dir,
        output_path=args.output,
        checkpoint=args.checkpoint,
        music_path=args.music,
        max_clips=args.max_clips,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
