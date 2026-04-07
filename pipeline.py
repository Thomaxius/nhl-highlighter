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
import datetime
import logging
from pathlib import Path

from src.ingestion.ingest import ingest_clips
from src.preprocessing.scene_splitter import split_into_scenes
from src.preprocessing.audio_analyzer import extract_audio, detect_energy_spikes, score_segments_by_audio
from src.detection.classifier import HighlightClassifier
from src.detection.banner_detector import BannerDetector
from src.detection.game_clock_detector import GameClockDetector
from src.detection.vs_screen_detector import VsScreenDetector
from src.detection.pause_menu_detector import PauseMenuDetector
from src.assembly.reel_builder import build_reel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _setup_file_logging(output_path) -> Path:
    """Attach a timestamped FileHandler to the root logger for this pipeline run."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(output_path).stem
    log_file = logs_dir / f"pipeline_{timestamp}_{stem}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    return log_file


def _chain_goal_sequences(results: list[dict]) -> list[dict]:
    """
    After a banner-detected or faceoff-inferred goal segment, look ahead and
    tag the immediately following celebration and replay segments so they are
    included in the reel in order: goal → celebration → replay.

    Stops chaining when:
    - A faceoff_cutscene or faceoff segment is reached (goal sequence is over)
    - An `other` segment appears that is longer than 10s (not just a transition flash)
    - A second goal is detected
    - MAX_LOOKAHEAD segments have been checked
    """
    FOLLOW_LABELS = {"celebration", "goal_replay"}
    MAX_LOOKAHEAD = 10
    MAX_GAP_S = 10.0   # stop chaining if >10s of non-highlight content

    goal_indices = [i for i, r in enumerate(results) if r.get("banner_detected") or r.get("inferred_goal")]

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
            if seg.get("banner_detected") or seg.get("inferred_goal"):
                break

            # Faceoff sequence means the goal chain is over
            if seg.get("label") in {"faceoff_cutscene", "faceoff"}:
                logger.info("  Chain stopped \u2014 faceoff sequence: %s", Path(seg["path"]).name)
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


def _infer_goals_from_faceoff_pattern(results: list[dict]) -> list[dict]:
    """
    Infer goals from the post-goal sequence when banner detection fails.

    After a real goal, the game always transitions to:
        (celebration) → (replay) → faceoff_cutscene → faceoff

    If a `scoring_chance` segment (or a segment the ML labelled `goal` but
    banner-gating demoted to `other`) is followed by `faceoff_cutscene`
    within the lookahead window, it is promoted to `goal` (inferred_goal=True).
    These inferred goals are then chained by _chain_goal_sequences.
    """
    MAX_LOOKAHEAD = 15

    promoted = 0
    for i, seg in enumerate(results):
        # Skip already-confirmed goals
        if seg.get("banner_detected") or seg.get("inferred_goal"):
            continue
        # Candidate: ML thought this was a goal or scoring chance
        ml = seg.get("ml_label", seg.get("label"))
        if ml not in {"goal", "scoring_chance"}:
            continue
        if seg.get("label") not in {"scoring_chance", "other"}:
            continue

        for j in range(1, MAX_LOOKAHEAD + 1):
            next_idx = i + j
            if next_idx >= len(results):
                break
            nxt = results[next_idx]

            # faceoff_cutscene = normal post-goal sequence
            # faceoff directly = player skipped the cutscene
            if nxt.get("label") in {"faceoff_cutscene", "faceoff"}:
                seg["label"] = "goal"
                seg["inferred_goal"] = True
                seg["confidence"] = max(seg.get("confidence", 0.0), 0.90)
                seg["score"] = max(seg.get("score", seg["confidence"]), 0.90)
                trigger = nxt.get("label")
                logger.info(
                    "  Faceoff-pattern goal inferred: %s  ml=%s conf=%.0f%%  (%s at +%d)",
                    Path(seg["path"]).name, ml, seg["confidence"] * 100, trigger, j,
                )
                promoted += 1
                break

            # Stop scanning if we hit a confirmed goal (new sequence)
            if nxt.get("banner_detected") or nxt.get("label") == "goal":
                break

    if promoted:
        logger.info("  Inferred %d goal(s) from faceoff pattern.", promoted)
    return results


def run_pipeline(
    input_dir: str,
    output_path: str,
    checkpoint: str = "models/checkpoints/videomae_nhl",
    music_path: str | None = None,
    max_clips: int = 20,
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
    # Save each segment's raw ML label before any overrides so that
    # faceoff-pattern inference can still act on originally-goal-labelled segments.
    for r in results:
        r["ml_label"] = r.get("label")
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
        elif r.get("label") == "goal":
            # No banner = no goal, regardless of model confidence
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

    # ── Step 4c.5: Infer goals from faceoff pattern (handles banner failures) ─
    logger.info("\u2501\u2501\u2501  Step 4c.5: Faceoff-pattern goal inference  \u2501\u2501\u2501")
    results = _infer_goals_from_faceoff_pattern(results)

    # ── Step 4d: Game clock detection (period / game end) ──────────────────
    # Scans the raw video for 0:00 (game end) and 0:01 (period end) clock
    # states. Segments covering the final ~60s before game end are tagged
    # game_end=True so the reel builder always appends them.
    clock_template = Path("configs/game_end_template.png")
    if clock_template.exists():
        logger.info("━━━  Step 4d: Game clock detection  ━━━")
        clock_detector = GameClockDetector(template_path=clock_template)
        FINAL_SECONDS_WINDOW = 5.0   # include up to this many seconds before 0:00
        POST_GAME_END_S    = 20.0  # include this many seconds after 0:00 is detected

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

            # If multiple game_end/regulation_end events were detected, only keep
            # the last one so trim offsets are from the true final buzzer.
            for terminal_type in ("game_end", "regulation_end"):
                terminal_events = [e for e in clock_events if e["event"] == terminal_type]
                if len(terminal_events) > 1:
                    logger.info(
                        "Multiple %s events detected (%d); keeping only the last (t=%.1fs)",
                        terminal_type, len(terminal_events), terminal_events[-1]["time_s"],
                    )
                    clock_events = [e for e in clock_events if e["event"] != terminal_type] + [terminal_events[-1]]
                    clock_events.sort(key=lambda e: e["time_s"])

            for ev in clock_events:
                ev_t   = ev["time_s"]
                ev_type = ev["event"]

                if ev_type == "game_end":
                    window_start  = ev_t - FINAL_SECONDS_WINDOW
                    clip_end_abs  = ev_t + POST_GAME_END_S
                    tagged        = 0
                    tagged_times: list[tuple[float, dict]] = []
                    for (start_s, end_s, seg) in seg_times:
                        if end_s > window_start and start_s < clip_end_abs:
                            seg["game_end"]    = True
                            seg["clock_event"] = ev_type
                            # Give a strong score so they survive ranking; score
                            # is set per-segment proportional to recency
                            recency = max(0.0, 1.0 - (ev_t - end_s) / FINAL_SECONDS_WINDOW)
                            seg["game_end_score"] = 1.0 + recency
                            tagged_times.append((start_s, seg))
                            tagged += 1
                    # Store trim offsets on the chronologically first segment so
                    # reel_builder knows exactly where to start and stop the clip.
                    if tagged_times:
                        tagged_times.sort(key=lambda x: x[0])
                        first_start_s, first_seg = tagged_times[0]
                        first_seg["game_end_trim_start_s"] = max(0.0, window_start - first_start_s)
                        first_seg["game_end_trim_end_s"]   = clip_end_abs - first_start_s
                    logger.info(
                        "Game-end at t=%.1fs: tagged %d segment(s) covering final %.0fs + %.0fs post",
                        ev_t, tagged, min(FINAL_SECONDS_WINDOW, ev_t), POST_GAME_END_S,
                    )

                elif ev_type == "regulation_end":
                    # 3rd period final buzzer — tag segments covering the final
                    # FINAL_SECONDS_WINDOW before the buzzer + POST_GAME_END_S after.
                    window_start = ev_t - FINAL_SECONDS_WINDOW
                    clip_end_abs = ev_t + POST_GAME_END_S
                    tagged       = 0
                    tagged_times_re: list[tuple[float, dict]] = []
                    for (start_s, end_s, seg) in seg_times:
                        if end_s > window_start and start_s < clip_end_abs:
                            seg["regulation_end"] = True
                            seg["clock_event"]    = ev_type
                            tagged_times_re.append((start_s, seg))
                            tagged += 1
                    if tagged_times_re:
                        tagged_times_re.sort(key=lambda x: x[0])
                        first_start_s, first_seg = tagged_times_re[0]
                        first_seg["game_end_trim_start_s"] = max(0.0, window_start - first_start_s)
                        first_seg["game_end_trim_end_s"]   = clip_end_abs - first_start_s
                    logger.info(
                        "Regulation-end at t=%.1fs: tagged %d segment(s) covering final %.0fs + %.0fs post",
                        ev_t, tagged, min(FINAL_SECONDS_WINDOW, ev_t), POST_GAME_END_S,
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
    vs_template = Path("configs/vs_screen_template3.png")
    PRE_VS_SKIP_S  = 6.0   # skip this many seconds after VS screen first appears
    POST_FACEOFF_S = 7.0   # seconds of actual play to include after 20:00
    FALLBACK_INTRO_S = 30.0  # fallback intro length when no VS screen is found
    if vs_template.exists():
        logger.info("━━━  Step 4d.5: VS screen / intro detection  ━━━")
        vs_detector = VsScreenDetector(template_path=vs_template)
        fallback_pause_template = Path("configs/pause_menu_template.png")
        fallback_pause_detector = (
            PauseMenuDetector(template_path=fallback_pause_template)
            if fallback_pause_template.exists() else None
        )
        fallback_pause_cache: dict[str, bool] = {}

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

            # Find game_start time from already-tagged segments
            game_start_t: float | None = None
            for (start_s, end_s, seg) in seg_times2:
                if seg.get("game_start"):
                    game_start_t = start_s
                    break

            # Use game_start as the upper bound for the VS screen search, with a
            # generous buffer. Fall back to the full video if clock detection
            # didn't find game_start (e.g. recording starts mid-game).
            if game_start_t is not None:
                vs_search_window = game_start_t + 120.0
                logger.info(
                    "  VS search window: 0 – %.1fs (game_start=%.1fs + 120s buffer)",
                    vs_search_window, game_start_t,
                )
            else:
                import cv2 as _cv2_vs
                _cap_vs = _cv2_vs.VideoCapture(str(norm_clip))
                _fps_vs = _cap_vs.get(_cv2_vs.CAP_PROP_FPS) or 30.0
                _n_vs   = _cap_vs.get(_cv2_vs.CAP_PROP_FRAME_COUNT)
                _cap_vs.release()
                vs_search_window = _n_vs / _fps_vs
                logger.info(
                    "  VS search window: full video (%.1fs) — no game_start detected",
                    vs_search_window,
                )

            vs_hits = vs_detector.find_matches(
                norm_clip,
                interval_s=0.5,
                search_window_s=vs_search_window,
                search_start_s=0.0,
            )
            if not vs_hits:
                fallback_seg = None
                for (start_s, end_s, seg) in seg_times2:
                    if seg.get("label") != "other":
                        continue
                    if fallback_pause_detector is not None:
                        seg_path = str(seg["path"])
                        is_pause = fallback_pause_cache.get(seg_path)
                        if is_pause is None:
                            is_pause = fallback_pause_detector.detect(seg["path"])["detected"]
                            fallback_pause_cache[seg_path] = is_pause
                        if is_pause:
                            logger.info(
                                "  Fallback intro skip (pause menu): %s [t=%.1f–%.1fs]",
                                Path(seg["path"]).name, start_s, end_s,
                            )
                            continue
                    fallback_seg = (start_s, end_s, seg)
                    break
                if fallback_seg is None:
                    logger.info(
                        "  No VS screen found in %s (searched %.1fs) and no 'other' fallback segment is available",
                        norm_clip.name, vs_search_window,
                    )
                    continue
                intro_mode = "fallback"
                intro_start = fallback_seg[0]
                intro_end = min(cum_t, intro_start + FALLBACK_INTRO_S)
                logger.info(
                    "  No VS screen found in %s (searched %.1fs) — fallback to first 'other' at t=%.1fs for %.1fs",
                    norm_clip.name, vs_search_window, intro_start, FALLBACK_INTRO_S,
                )
            else:
                vs_clusters: list[list[tuple[float, float]]] = []
                for hit_t, hit_conf in vs_hits:
                    if not vs_clusters or (hit_t - vs_clusters[-1][-1][0]) > 1.0:
                        vs_clusters.append([(hit_t, hit_conf)])
                    else:
                        vs_clusters[-1].append((hit_t, hit_conf))

                # Use the LAST cluster before game_start — handles the case
                # where the user visited the VS matchup menu and backed out
                # (early false hit) before committing to the game (late true hit).
                # If game_start is unknown, fall back to the last cluster overall.
                if game_start_t is not None:
                    valid_clusters = [c for c in vs_clusters if c[0][0] < game_start_t]
                    chosen_cluster = valid_clusters[-1] if valid_clusters else vs_clusters[-1]
                else:
                    chosen_cluster = vs_clusters[-1]

                vs_t = chosen_cluster[0][0]
                vs_peak_conf = max(conf for _, conf in chosen_cluster)
                logger.info(
                    "  VS screen cluster selected: %.1f–%.1fs (peak conf %.3f, %d sampled hit(s), %d total cluster(s))",
                    chosen_cluster[0][0],
                    chosen_cluster[-1][0],
                    vs_peak_conf,
                    len(chosen_cluster),
                    len(vs_clusters),
                )

                intro_mode = "vs"
                intro_start = vs_t + PRE_VS_SKIP_S
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

            if intro_mode == "vs":
                logger.info(
                    "Intro: VS+%.0fs (t=%.1fs) → game_start+%.0fs (t=%.1fs): "
                    "tagged %d segment(s)",
                    PRE_VS_SKIP_S, intro_start, POST_FACEOFF_S, intro_end, tagged,
                )
            else:
                logger.info(
                    "Fallback intro: first 'other' [t=%.1fs] → +%.0fs (t=%.1fs): "
                    "tagged %d segment(s)",
                    intro_start, FALLBACK_INTRO_S, intro_end, tagged,
                )
    else:
        logger.info("Skipping VS screen detection (configs/vs_screen_template3.png not found)")

    # ── Step 4d.9: Pause-menu detection ────────────────────────────────────
    # Runs AFTER intro and game_end tagging so those guards work correctly.
    # Segments where the ESC/pause menu is visible are demoted to "other".
    # Skips intro/game_end segments (those are assembled directly, not scored).
    pause_template = Path("configs/pause_menu_template.png")
    if pause_template.exists():
        logger.info("━━━  Step 4d.9: Pause-menu detection  ━━━")
        pause_detector = PauseMenuDetector(template_path=pause_template)
        paused_count = 0
        for r in results:
            if r.get("game_end") or r.get("intro"):
                continue
            res = pause_detector.detect(r["path"])
            if res["detected"]:
                logger.info(
                    "  Pause menu → demoted: %s (conf=%.2f)",
                    Path(r["path"]).name, res["max_conf"],
                )
                r["label"]    = "other"
                r["has_menu"] = True
                paused_count += 1
        if paused_count:
            logger.info("  Demoted %d segment(s) containing pause menu.", paused_count)
    else:
        logger.info("Skipping pause-menu detection (configs/pause_menu_template.png not found)")

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
    p.add_argument("--max_clips", type=int, default=20)
    p.add_argument("--min_confidence", type=float, default=0.55)
    args = p.parse_args()

    if args.file:
        # Wrap the single file in a temp dir so the pipeline's ingest step
        # sees it as a one-file folder without touching the original.
        import tempfile, os
        file_path = Path(args.file).resolve()
        tmp = tempfile.mkdtemp(prefix="pipeline_single_", dir='.')
        os.link(file_path, Path(tmp) / file_path.name)   # hardlink — instant, no extra disk
        input_dir = tmp
    else:
        input_dir = args.folder or "data/raw"

    log_file = _setup_file_logging(args.output)
    logger.info("Logging to %s", log_file)

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
