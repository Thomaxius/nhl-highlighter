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
from src.detection.game_clock_detector import GameClockDetector
from src.detection.vs_screen_detector import VsScreenDetector
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
        g = results[goal_idx]
        base_score = g.get("score", g.get("confidence", 1.0))
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

    # ── Step 4b: Banner detection via raw-video scan ────────────────────────
    # Scan the full normalised video at 1s intervals rather than per-split-
    # segment.  Per-segment detection misses goals whose banner appears right
    # at a scene cut (empirically conf ≈ 0.44 on segments vs 0.97 on the raw
    # source).  After finding banner timestamps we map them back to the
    # correct segment using a cumulative timeline.
    import cv2 as _cv2
    templates = sorted(Path("configs").glob("goal_banner_template*.png"))
    if templates:
        logger.info("━━━  Step 4b: Banner detection (%d template(s))  ━━━", len(templates))
        detector = BannerDetector()  # auto-loads all matching templates
        PRE_GOAL_S  = 7.0
        POST_GOAL_S = 3.0

        for norm_clip in normalised:
            clip_stem = norm_clip.stem
            # Segments for this clip, sorted by filename (== chronological order)
            clip_segs = sorted(
                [r for r in results if clip_stem in str(r["path"])],
                key=lambda r: r["path"],
            )
            if not clip_segs:
                continue

            # Build cumulative timeline: (start_s, end_s, seg_dict, seg_fps)
            seg_ranges: list[tuple[float, float, dict, float]] = []
            cum_t = 0.0
            for seg in clip_segs:
                _cap = _cv2.VideoCapture(str(seg["path"]))
                _fps = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
                _n   = _cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                _cap.release()
                dur = _n / _fps
                seg_ranges.append((cum_t, cum_t + dur, seg, _fps))
                cum_t += dur

            # Scan the uncut normalised source — no scene-boundary artifacts
            banner_times = detector.scan_video(norm_clip, interval_s=1.0)

            for bt in banner_times:
                for (start_s, end_s, seg, seg_fps) in seg_ranges:
                    if start_s <= bt < end_s:
                        local_t  = bt - start_s
                        seg_dur  = end_s - start_s
                        seg["banner_detected"]   = True
                        seg["banner_confidence"] = 0.95
                        seg["best_frame"]        = int(local_t * seg_fps)
                        # Trim window centred on the goal frame, clamped to segment
                        seg["trim_start_s"] = max(0.0, local_t - PRE_GOAL_S)
                        seg["trim_end_s"]   = min(seg_dur, local_t + POST_GOAL_S)
                        logger.info(
                            "Banner detected via raw scan: t=%.1fs → %s  "
                            "trim %.1fs→%.1fs",
                            bt, Path(seg["path"]).name,
                            seg["trim_start_s"], seg["trim_end_s"],
                        )
                        break  # each goal event maps to exactly one segment
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

    # ── Step 4d: Game clock detection (period / game end) ──────────────────
    # Scans the raw video for 0:00 (game end) and 0:01 (period end) clock
    # states. Segments covering the final ~60s before game end are tagged
    # game_end=True so the reel builder always appends them.
    clock_template = Path("configs/game_end_template.png")
    if clock_template.exists():
        logger.info("━━━  Step 4d: Game clock detection  ━━━")
        clock_detector = GameClockDetector(template_path=clock_template)
        FINAL_SECONDS_WINDOW = 60.0  # include up to this many seconds before 0:00

        for norm_clip in normalised:
            clip_stem = norm_clip.stem
            clip_segs = sorted(
                [r for r in results if clip_stem in str(r["path"])],
                key=lambda r: r["path"],
            )
            if not clip_segs:
                continue

            # Build cumulative timeline for this clip
            import cv2 as _cv2
            seg_times: list[tuple[float, float, dict]] = []
            cum_t = 0.0
            for seg in clip_segs:
                _cap = _cv2.VideoCapture(str(seg["path"]))
                _fps = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
                _n   = _cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                _cap.release()
                dur = _n / _fps
                seg_times.append((cum_t, cum_t + dur, seg))
                cum_t += dur

            clock_events = clock_detector.scan_video(norm_clip)

            for ev in clock_events:
                ev_t   = ev["time_s"]
                ev_type = ev["event"]

                if ev_type == "game_end":
                    window_start = ev_t - FINAL_SECONDS_WINDOW
                    tagged = 0
                    for (start_s, end_s, seg) in seg_times:
                        if end_s > window_start and start_s < ev_t:
                            seg["game_end"]   = True
                            seg["clock_event"] = ev_type
                            # Give a strong score so they survive ranking; score
                            # is set per-segment proportional to recency
                            recency = max(0.0, 1.0 - (ev_t - end_s) / FINAL_SECONDS_WINDOW)
                            seg["game_end_score"] = 1.0 + recency
                            tagged += 1
                    logger.info(
                        "Game-end at t=%.1fs: tagged %d segment(s) covering final %.0fs",
                        ev_t, tagged, min(FINAL_SECONDS_WINDOW, ev_t),
                    )

                elif ev_type == "period_end":
                    # Tag the segment containing the period-end moment only
                    for (start_s, end_s, seg) in seg_times:
                        if start_s <= ev_t < end_s:
                            seg["period_end"] = True
                            seg["clock_event"] = ev_type
                            logger.info(
                                "Period-end at t=%.1fs → %s",
                                ev_t, Path(seg["path"]).name,
                            )
                            break

                elif ev_type in ("game_start", "period_start"):
                    # Tag the segment containing the puck-drop moment
                    for (start_s, end_s, seg) in seg_times:
                        if start_s <= ev_t < end_s:
                            seg[ev_type] = True
                            seg["clock_event"] = ev_type
                            logger.info(
                                "%s at t=%.1fs → %s",
                                ev_type, ev_t, Path(seg["path"]).name,
                            )
                            break
    else:
        logger.info("Skipping game clock detection (configs/game_end_template.png not found)")

    # ── Step 4d.5: VS screen → intro segment tagging ─────────────────────────
    # Clip starts 6 s after the VS screen first appears and runs through
    # game_start (20:00 clock) + 7 s of actual play.
    # Tagged segments carry trim offsets so the reel builder can cut precisely:
    #   intro_clip_start_s — seconds from the first tagged segment's start
    #   intro_clip_end_s   — seconds from the first tagged segment's start
    vs_template = Path("configs/vs_screen_template.png")
    PRE_VS_SKIP_S  = 6.0   # skip this many seconds after VS screen first appears
    POST_FACEOFF_S = 7.0   # seconds of actual play to include after 20:00
    if vs_template.exists():
        logger.info("━━━  Step 4d.5: VS screen / intro detection  ━━━")
        vs_detector = VsScreenDetector(template_path=vs_template)

        for norm_clip in normalised:
            clip_stem = norm_clip.stem
            clip_segs = sorted(
                [r for r in results if clip_stem in str(r["path"])],
                key=lambda r: r["path"],
            )
            if not clip_segs:
                continue

            # Build cumulative timeline
            import cv2 as _cv2
            seg_times2: list[tuple[float, float, dict]] = []
            cum_t = 0.0
            for seg in clip_segs:
                _cap = _cv2.VideoCapture(str(seg["path"]))
                _fps = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
                _n   = _cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                _cap.release()
                dur = _n / _fps
                seg_times2.append((cum_t, cum_t + dur, seg))
                cum_t += dur

            vs_t = vs_detector.scan_video(norm_clip, interval_s=0.5, search_window_s=60.0)
            if vs_t is None:
                logger.info("  No VS screen found in %s", norm_clip.name)
                continue

            intro_start = vs_t + PRE_VS_SKIP_S

            # Find game_start time from already-tagged segments
            game_start_t: float | None = None
            for (start_s, end_s, seg) in seg_times2:
                if seg.get("game_start"):
                    game_start_t = start_s
                    break

            intro_end = (game_start_t + POST_FACEOFF_S) if game_start_t is not None else (vs_t + 35.0)

            # Tag segments spanning [intro_start, intro_end) and record exact
            # trim offsets on the first tagged segment for the reel builder.
            tagged = 0
            first_intro_seg_start: float | None = None
            for (start_s, end_s, seg) in seg_times2:
                if end_s > intro_start and start_s < intro_end:
                    seg["intro"] = True
                    tagged += 1
                    if first_intro_seg_start is None:
                        first_intro_seg_start = start_s
                        # How far into the concatenated intro block to start/end
                        seg["intro_clip_start_s"] = max(0.0, intro_start - start_s)
                        seg["intro_clip_end_s"]   = intro_end - start_s

            logger.info(
                "Intro: VS+%.0fs (t=%.1fs) → game_start+%.0fs (t=%.1fs): "
                "tagged %d segment(s)",
                PRE_VS_SKIP_S, intro_start, POST_FACEOFF_S, intro_end, tagged,
            )
    else:
        logger.info("Skipping VS screen detection (configs/vs_screen_template.png not found)")

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
