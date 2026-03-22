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
from src.detection.banner_detector import BannerDetector, AssistsBannerDetector
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


def _chain_goal_sequences(results: list[dict]) -> list[dict]:
    """
    After a goal segment (banner-detected or high-confidence classifier),
    look ahead and tag the immediately following celebration and replay
    segments so they are included in the reel in the correct order:
    goal → celebration → replay.

    Stops chaining when:
    - An `other` segment appears that is longer than 10s (not just a transition flash)
    - A second goal is encountered
    - MAX_LOOKAHEAD segments have been checked
    """
    FOLLOW_LABELS = {"celebration", "goal_replay"}
    MAX_LOOKAHEAD = 10
    MAX_GAP_S = 10.0   # stop chaining if >10s of non-highlight content

    goal_indices = [i for i, r in enumerate(results) if r.get("label") == "goal"]

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
            if seg.get("label") == "goal":
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


def _chain_scoring_chance_sequences(results: list[dict]) -> list[dict]:
    """
    After a scoring_chance segment, look ahead up to WINDOW_S seconds for a
    replay (goal_replay or other_replay) and chain it so both are included in
    the reel together: scoring_chance → replay.

    Stops chaining when:
    - A banner-detected goal or another scoring_chance is encountered
    - Cumulative gap exceeds WINDOW_S (20s)
    - MAX_LOOKAHEAD segments have been exhausted
    - A replay already chained to a goal is found (don't steal it)
    """
    FOLLOW_LABELS = {"goal_replay", "other_replay"}
    WINDOW_S = 20.0
    MAX_LOOKAHEAD = 10

    sc_indices = [i for i, r in enumerate(results) if r.get("label") == "scoring_chance"]

    for sc_idx in sc_indices:
        sc = results[sc_idx]
        base_score = sc.get("score", sc.get("confidence", 1.0))
        elapsed_s = 0.0

        for j in range(1, MAX_LOOKAHEAD + 1):
            next_idx = sc_idx + j
            if next_idx >= len(results):
                break

            seg = results[next_idx]

            # Stop at another goal or scoring chance
            if seg.get("label") in {"goal", "scoring_chance"}:
                break

            if seg.get("label") in FOLLOW_LABELS:
                # Don't steal a replay already chained to a goal
                if "chain_goal_idx" in seg:
                    break
                seg["chain_sc_idx"] = sc_idx
                seg["score"] = base_score - 0.01
                seg["confidence"] = max(seg.get("confidence", 0.0), 0.9)
                logger.info(
                    "  Chained %s to scoring_chance %d: %s",
                    seg["label"], sc_idx, Path(seg["path"]).name,
                )
                break  # one replay per scoring chance is enough

            elapsed_s += _get_clip_duration(seg["path"])
            if elapsed_s > WINDOW_S:
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


def _trim_scoring_chances(results: list[dict]) -> list[dict]:
    """
    Set trim_start_s / trim_end_s on every scoring_chance segment.

    - If the preceding segment has label 'other' (e.g. a faceoff ran into
      this clip), trim 3 s from the start to skip past the dead-ball content.
    - Otherwise trim 1 s from the start to remove any transition artifacts.
    - Always trim 1 s from the end.
    """
    TRIM_AFTER_OTHER = 3.0
    TRIM_DEFAULT     = 1.0
    TRIM_END         = 1.0
    MIN_DURATION     = 0.5  # never trim below this

    for i, seg in enumerate(results):
        if seg.get("label") != "scoring_chance":
            continue
        prev_label = results[i - 1].get("label") if i > 0 else None
        trim_start = TRIM_AFTER_OTHER if prev_label == "other" else TRIM_DEFAULT
        duration   = _get_clip_duration(seg["path"])
        trim_end   = max(trim_start + MIN_DURATION, duration - TRIM_END)
        seg["trim_start_s"] = trim_start
        seg["trim_end_s"]   = trim_end
        logger.info(
            "  SC trim [%.1fs → %.1fs] (prev=%s): %s",
            trim_start, trim_end, prev_label, Path(seg["path"]).name,
        )
    return results


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

    # ── Steps 3 + 4 + 4b: run in parallel ─────────────────────────────────
    # These three heavy scans are fully independent — each only needs
    # `normalised` clips and/or `all_segments`.  Run them concurrently to
    # overlap I/O-bound audio extraction, CPU-bound classification, and the
    # OpenCV banner scan.
    #
    # NOTE: HighlightClassifier loads a PyTorch model; it is NOT thread-safe
    # if shared between threads. We create it inside the worker so each thread
    # owns its own instance (fine for single-clip jobs; for large batches use
    # multiprocessing instead).
    import concurrent.futures as _cf

    def _run_audio(_normalised, _audio_dir):
        logger.info("━━━  Step 3: Audio analysis  ━━━")
        _spikes = []
        for clip in _normalised:
            wav_path = _audio_dir / (clip.stem + ".wav")
            try:
                extract_audio(clip, wav_path)
                _spikes.extend(detect_energy_spikes(wav_path))
            except Exception as e:
                logger.warning("Audio analysis failed for %s: %s", clip.name, e)
        return _spikes

    def _run_classify(_all_segments, _checkpoint):
        logger.info("━━━  Step 4: Classifying segments  ━━━")
        _clf = HighlightClassifier(checkpoint_path=_checkpoint)
        return _clf.classify_segments(_all_segments)

    def _run_banner(_normalised):
        _templates = sorted(Path("configs").glob("goal_banner_template*.png"))
        if not _templates:
            logger.info("Skipping banner detection (no templates found in configs/)")
            return {}
        logger.info("━━━  Step 4b: Banner detection (%d template(s))  ━━━", len(_templates))
        import cv2 as _cv2
        _det = BannerDetector()
        PRE_GOAL_S, POST_GOAL_S = 7.0, 3.0
        banner_results: dict = {}  # clip_stem → list of (bt, seg_ranges)
        _timeline: dict = {}
        for norm_clip in _normalised:
            _banner_times = _det.scan_video(norm_clip, interval_s=1.0)
            _timeline[norm_clip.stem] = {"times": _banner_times, "pre": PRE_GOAL_S, "post": POST_GOAL_S}
        return _timeline

    def _run_assists(_normalised):
        assists_tpls = sorted(Path("configs").glob("after_goal_assists_template*.png"))
        if not assists_tpls:
            logger.info("Skipping assists-banner detection (template not found)")
            return {}
        logger.info("━━━  Step 4b.5: Assists-banner detection  ━━━")
        _det = AssistsBannerDetector()  # auto-discovers template
        _timeline: dict = {}
        for norm_clip in _normalised:
            _times = _det.scan_video(norm_clip, interval_s=1.0)
            _timeline[norm_clip.stem] = _times
        return _timeline

    logger.info("Running Steps 3 / 4 / 4b / 4b.5 in parallel…")
    with _cf.ThreadPoolExecutor(max_workers=4) as pool:
        fut_audio   = pool.submit(_run_audio,    normalised, audio_dir)
        fut_clf     = pool.submit(_run_classify,  all_segments, checkpoint)
        fut_banner  = pool.submit(_run_banner,    normalised)
        fut_assists = pool.submit(_run_assists,   normalised)
        all_spikes        = fut_audio.result()
        results           = fut_clf.result()
        banner_timeline   = fut_banner.result()
        assists_timeline  = fut_assists.result()
    logger.info("Parallel steps complete.")

    # ── Step 4b (merge): apply banner timeline back onto results ──────────────
    import cv2 as _cv2
    if banner_timeline:
        for norm_clip in normalised:
            clip_stem = norm_clip.stem
            tl = banner_timeline.get(clip_stem)
            if not tl:
                continue
            PRE_GOAL_S  = tl["pre"]
            POST_GOAL_S = tl["post"]
            clip_segs = sorted(
                [r for r in results if clip_stem in str(r["path"])],
                key=lambda r: r["path"],
            )
            if not clip_segs:
                continue
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
            for bt in tl["times"]:
                for (start_s, end_s, seg, seg_fps) in seg_ranges:
                    if start_s <= bt < end_s:
                        local_t  = bt - start_s
                        seg_dur  = end_s - start_s
                        seg["banner_detected"]   = True
                        seg["banner_confidence"] = 0.95
                        seg["best_frame"]        = int(local_t * seg_fps)
                        seg["trim_start_s"] = max(0.0, local_t - PRE_GOAL_S)
                        seg["trim_end_s"]   = min(seg_dur, local_t + POST_GOAL_S)
                        logger.info(
                            "Banner detected via raw scan: t=%.1fs → %s  trim %.1fs→%.1fs",
                            bt, Path(seg["path"]).name,
                            seg["trim_start_s"], seg["trim_end_s"],
                        )
                        break

    # ── Step 4c: Banner is the ground truth — hard override ──────────────────
    # If the GOAL! banner was detected, the segment IS a goal, period.
    # No ML confidence threshold can override a pixel-level HUD match.
    # High-confidence classifier goals (≥85%) are also trusted even without
    # a banner — the HUD sometimes bugs out and never shows the GOAL text.
    # Only demote low-confidence goal labels that lack banner confirmation.
    GOAL_CONF_THRESHOLD = 0.85
    logger.info("━━━  Step 4c: Banner-gating goal labels  ━━━")
    demoted = 0
    promoted = 0
    kept = 0
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
            if r.get("confidence", 0) >= GOAL_CONF_THRESHOLD:
                # High-confidence classifier goal — trust it without banner
                logger.info(
                    "  Kept high-conf goal (no banner): %s (conf=%.0f%%)",
                    Path(r["path"]).name, r.get("confidence", 0) * 100,
                )
                kept += 1
            else:
                # Low confidence + no banner → demote
                logger.info(
                    "  Demoted → other: %s (conf=%.0f%%, no banner)",
                    Path(r["path"]).name, r.get("confidence", 0) * 100,
                )
                r["label"] = "other"
                demoted += 1
    if promoted:
        logger.info("  Promoted %d segment(s) to goal via banner detection.", promoted)
    if kept:
        logger.info("  Kept %d high-confidence goal(s) without banner (≥%.0f%%).", kept, GOAL_CONF_THRESHOLD * 100)
    if demoted:
        logger.info("  Demoted %d goal segment(s) without banner confirmation.", demoted)

    # ── Step 4c.5: Assists-banner backward recovery ──────────────────────
    # When the GOAL banner glitches out, the assists banner at the next
    # faceoff is our fallback.  For each assists-banner timestamp that has
    # no nearby GOAL-banner-detected goal, walk backward through segments
    # to find the most recent scoring_chance and promote it to a goal.
    if assists_timeline:
        logger.info("━━━  Step 4c.5: Assists-banner backward goal recovery  ━━━")
        recovered = 0
        for norm_clip in normalised:
            clip_stem = norm_clip.stem
            assists_times = assists_timeline.get(clip_stem)
            if not assists_times:
                continue

            # Build cumulative timeline for this clip's segments
            clip_segs = sorted(
                [r for r in results if clip_stem in str(r["path"])],
                key=lambda r: r["path"],
            )
            if not clip_segs:
                continue
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

            # Map each segment dict to its index in clip_segs for backward walk
            seg_to_idx = {id(seg): i for i, seg in enumerate(clip_segs)}

            LOOKBACK_S = 30.0  # goal happened at most 30s before assists banner

            for at in assists_times:
                # Check: is there already a banner-detected goal within LOOKBACK_S?
                goal_nearby = False
                for (start_s, end_s, seg, _) in seg_ranges:
                    if seg.get("banner_detected") and (at - LOOKBACK_S) <= start_s <= at:
                        goal_nearby = True
                        break
                if goal_nearby:
                    logger.info(
                        "  Assists at t=%.1fs — goal already detected nearby, skipping",
                        at,
                    )
                    continue

                # Find the segment that contains the assists-banner timestamp
                assists_seg_idx: int | None = None
                for (start_s, end_s, seg, _) in seg_ranges:
                    if start_s <= at < end_s:
                        assists_seg_idx = seg_to_idx[id(seg)]
                        break
                if assists_seg_idx is None:
                    continue

                # Walk backward from assists banner to find the scoring chance
                # that was the actual goal.  Also accept celebration/goal_replay
                # as waypoints but keep looking for scoring_chance.
                goal_seg: dict | None = None
                goal_seg_start_s: float | None = None
                for k in range(assists_seg_idx - 1, -1, -1):
                    seg_start_s = seg_ranges[k][0]
                    if at - seg_start_s > LOOKBACK_S:
                        break
                    cand = clip_segs[k]
                    if cand.get("label") == "scoring_chance":
                        goal_seg = cand
                        goal_seg_start_s = seg_start_s
                        break
                    # Skip over celebration / replay / short other segments
                    if cand.get("label") in {"celebration", "goal_replay", "other_replay"}:
                        continue
                    if cand.get("label") == "other":
                        dur = _get_clip_duration(cand["path"])
                        if dur <= 5.0:
                            continue  # short transition, keep looking
                        break  # long other = unrelated content

                if goal_seg is None:
                    logger.info(
                        "  Assists at t=%.1fs — no scoring_chance found in lookback window",
                        at,
                    )
                    continue

                # Promote the scoring_chance to a goal
                goal_seg["banner_detected"]   = True
                goal_seg["banner_confidence"] = 0.90
                goal_seg["assists_recovered"] = True
                goal_seg["label"]      = "goal"
                goal_seg["confidence"] = 1.0
                recovered += 1
                logger.info(
                    "  RECOVERED goal via assists banner: assists t=%.1fs → %s (was scoring_chance at t=%.1fs)",
                    at, Path(goal_seg["path"]).name, goal_seg_start_s,
                )

        if recovered:
            logger.info("  Recovered %d missed goal(s) via assists-banner detection.", recovered)
    else:
        logger.info("Skipping assists-banner recovery (no assists timeline)")

    # ── Step 4c.8: Pattern-confirmed goal recovery ───────────────────────────
    # Last chance: if a segment was classifier-labeled 'goal' with high
    # confidence but demoted by Step 4c (no banner), check whether the
    # following segments form a goal-like pattern (celebration / goal_replay).
    # If so, the classifier was right — the HUD just bugged out.
    logger.info("━━━  Step 4c.8: Pattern-confirmed goal recovery  ━━━")
    PATTERN_MIN_CONF = 0.75  # original classifier confidence needed
    PATTERN_FOLLOW = {"celebration", "goal_replay"}
    pattern_recovered = 0
    for i, r in enumerate(results):
        # Only look at segments that were demoted from goal to other in Step 4c
        if r.get("label") != "other" or not r.get("scores"):
            continue
        orig_goal_conf = r["scores"].get("goal", 0.0)
        if orig_goal_conf < PATTERN_MIN_CONF:
            continue
        # Already recovered by another method
        if r.get("banner_detected"):
            continue

        # Look ahead for celebration or goal_replay within the next few segments
        found_pattern = False
        for j in range(1, 6):
            nxt_idx = i + j
            if nxt_idx >= len(results):
                break
            nxt = results[nxt_idx]
            if nxt.get("label") in PATTERN_FOLLOW:
                found_pattern = True
                break
            if nxt.get("label") == "other":
                dur = _get_clip_duration(nxt["path"])
                if dur > 5.0:
                    break  # long gap = unrelated

        if found_pattern:
            r["banner_detected"]     = True
            r["banner_confidence"]   = orig_goal_conf
            r["pattern_recovered"]   = True
            r["label"]      = "goal"
            r["confidence"] = orig_goal_conf
            pattern_recovered += 1
            logger.info(
                "  PATTERN RECOVERED goal: %s (orig_goal_conf=%.0f%%, "
                "followed by %s)",
                Path(r["path"]).name, orig_goal_conf * 100,
                results[nxt_idx]["label"],
            )

    if pattern_recovered:
        logger.info("  Pattern-recovered %d goal(s) via celebration/replay sequence.", pattern_recovered)

    # ── Step 4d: Game clock detection (period / game end) ──────────────────
    # Scans the raw video for 0:00 (regulation_end) and 0:01 (period end) clock
    # states. regulation_end times are recorded and used in Step 4d.8 to anchor
    # the start of the final game_end clip (from buzzer to END OF GAME + 30s).
    clock_template = Path("configs/game_end_template.png")
    # Dict of clip_stem → regulation_end time (seconds) — consumed in Step 4d.8
    _clip_regulation_end_t: dict[str, float] = {}
    if clock_template.exists():
        logger.info("━━━  Step 4d: Game clock detection  ━━━")
        clock_detector = GameClockDetector(template_path=clock_template)

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

                if ev_type == "regulation_end":
                    # 3RD period buzzer — record as window start for game_end clip
                    # AND tag the surrounding segments as a standalone highlight.
                    _clip_regulation_end_t[clip_stem] = ev_t

                    RE_PRE_S  = 5.0   # seconds before buzzer to include
                    RE_POST_S = 60.0  # fallback max — smart trim shrinks this to
                                      # the first intermission/EOG screen found
                    window_start = ev_t - RE_PRE_S
                    clip_end_abs = ev_t + RE_POST_S
                    logger.info(
                        "  regulation_end: buzzer at t=%.1fs, initial window [%.1fs, %.1fs]",
                        ev_t, window_start, clip_end_abs,
                    )

                    # ── Smart trim: stop at the first intermission / EOG screen ──
                    # Scan ALL segments whose start time is after the buzzer and
                    # within the fallback window.  Stop at the first one that
                    # contains INTERMISSION or END OF GAME text.  RE_POST_S is
                    # deliberately generous (60 s) so the intermission screen is
                    # always within range; the trim brings the clip end back.
                    _re_stop_detector = PauseMenuDetector(
                        template_path=Path("configs/pause_menu_template_end_of_game.png"),
                        threshold=0.60,
                        sample_frames=4,
                    ) if Path("configs/pause_menu_template_end_of_game.png").exists() else None
                    if _re_stop_detector is not None:
                        logger.info(
                            "  Scanning for intermission screen in [%.1fs, %.1fs] …",
                            ev_t, clip_end_abs,
                        )
                        for (s_s, e_s, seg) in seg_times:
                            if s_s < ev_t or s_s >= clip_end_abs:
                                continue
                            hit = _re_stop_detector.detect(
                                seg["path"],
                                require_ocr_text=["INTERMISSION", "END OF GAME", "END", "GAME"],
                            )
                            seg_name = Path(seg["path"]).name
                            if hit["detected"]:
                                clip_end_abs = s_s  # stop just before this segment
                                logger.info(
                                    "  regulation_end clip trimmed at %.1fs "
                                    "(found '%s' in %s, conf=%.2f)",
                                    clip_end_abs,
                                    hit.get("ocr_text", "").replace("\n", " ").strip()[:80],
                                    seg_name, hit.get("max_conf", 0.0),
                                )
                                break
                            else:
                                logger.info(
                                    "  Smart trim: %s [%.1f-%.1fs] — no intermission "
                                    "(conf=%.2f, ocr=%r)",
                                    seg_name, s_s, e_s,
                                    hit.get("max_conf", 0.0),
                                    hit.get("ocr_text", "").replace("\n", " ").strip()[:60],
                                )
                    else:
                        logger.info(
                            "  Smart trim skipped (no pause_menu_template_end_of_game.png)"
                        )

                    tagged_re: list[tuple[float, dict]] = []
                    for (start_s, end_s, seg) in seg_times:
                        if end_s > window_start and start_s < clip_end_abs:
                            seg["regulation_end"] = True
                            seg["clock_event"]    = "regulation_end"
                            tagged_re.append((start_s, seg))
                            logger.info(
                                "  Tagged regulation_end: %s  [t=%.1f–%.1fs]",
                                Path(seg["path"]).name, start_s, end_s,
                            )
                    if tagged_re:
                        tagged_re.sort(key=lambda x: x[0])
                        first_start_s, first_seg = tagged_re[0]
                        first_seg["game_end_trim_start_s"] = max(0.0, window_start - first_start_s)
                        first_seg["game_end_trim_end_s"]   = clip_end_abs - first_start_s
                        logger.info(
                            "  regulation_end lead trim: start=%.1fs end=%.1fs (within first segment)",
                            first_seg["game_end_trim_start_s"], first_seg["game_end_trim_end_s"],
                        )
                    logger.info(
                        "Regulation-end (3RD buzzer) at t=%.1fs — tagged %d segment(s) "
                        "final window [%.1fs, %.1fs] in %s",
                        ev_t, len(tagged_re), window_start, clip_end_abs, norm_clip.name,
                    )

                elif ev_type == "ot_period_end":
                    # An OT period ran to 0:00 — another OT follows; game is NOT over.
                    # Log it but don't update regulation_end_t (we always anchor
                    # the game_end window from the 3RD period buzzer so all OT
                    # footage is captured between regulation_end and EOG screen).
                    logger.info(
                        "OT period ran out at t=%.1fs (another OT follows) in %s",
                        ev_t, norm_clip.name,
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

            # Find game_start time from already-tagged segments so we can set
            # a sensible search window for the VS screen scan.  The VS screen
            # always appears before puck-drop, so searching up to game_start+120s
            # covers any amount of pre-game franchise-mode menus.
            game_start_t: float | None = None
            for (start_s, end_s, seg) in seg_times2:
                if seg.get("game_start"):
                    game_start_t = start_s
                    break

            # Use game_start as the upper bound; fall back to full-video scan
            # if clock detection didn't find a game_start for this clip.
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

            VS_SEARCH_START_S = 25.0  # skip first 25s to bypass the brief early VS appearance
            vs_t = vs_detector.scan_video(
                norm_clip, interval_s=0.5,
                search_window_s=vs_search_window,
                search_start_s=VS_SEARCH_START_S,
            )
            if vs_t is None:
                logger.info("  No VS screen found in %s (searched %.1fs)", norm_clip.name, vs_search_window)
                continue

            intro_start = vs_t + PRE_VS_SKIP_S
            intro_end = (game_start_t + POST_FACEOFF_S) if game_start_t is not None else (vs_t + 35.0)

            # Tag segments spanning [intro_start, intro_end) and record exact
            # trim offsets on the first tagged segment for the reel builder.
            # Never absorb a banner-detected goal into the intro — it must stay
            # as its own highlight group.
            tagged = 0
            first_intro_seg_start: float | None = None
            for (start_s, end_s, seg) in seg_times2:
                if end_s > intro_start and start_s < intro_end:
                    if seg.get("banner_detected"):
                        logger.info(
                            "  Intro skip (banner-detected goal): %s [%.1f–%.1fs]",
                            Path(seg["path"]).name, start_s, end_s,
                        )
                        continue
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

    # ── Step 4d.8: END OF GAME screen detection ────────────────────────────────
    # The END OF GAME scoreboard overlay is the most reliable signal for where
    # the game truly ended.  game_end clip = [eog_t−60s, eog_t+30s].
    # Any regulation_end that fired AFTER the EOG screen is a clock OCR false
    # positive (celebration HUD misread as 0:00 3RD) and is discarded here.
    eog_template = Path("configs/pause_menu_template_end_of_game.png")
    if eog_template.exists():
        logger.info("━━━  Step 4d.8: END OF GAME screen detection  ━━━")
        eog_detector = PauseMenuDetector(
            template_path=eog_template,
            threshold=0.60,
            sample_frames=6,
        )
        PRE_EOG_S  = 10.0  # seconds before first EOG screen appearance to include
        POST_EOG_S = 10.0  # seconds after first EOG screen appearance to include

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
            seg_times_eog: list[tuple[float, float, dict]] = []
            cum_t = 0.0
            for seg in clip_segs:
                _cap = _cv2.VideoCapture(str(seg["path"]))
                _fps = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
                _n   = _cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                _cap.release()
                dur = _n / _fps
                seg_times_eog.append((cum_t, cum_t + dur, seg))
                cum_t += dur

            # Find the LAST segment containing END OF GAME screen — always take
            # the latest occurrence because earlier clips can false-positive on
            # intermission or scoreboard overlays that superficially match.
            # OCR verification ensures the frame actually contains "END OF GAME"
            # text, not just a visually similar pause/intermission overlay.
            eog_t_first: float | None = None  # earliest confirmed EOG hit (start anchor)
            eog_t: float | None = None             # latest confirmed EOG hit (end anchor)
            eog_conf: float = 0.0
            eog_name: str = ""
            for (start_s, end_s, seg) in seg_times_eog:
                res = eog_detector.detect(
                    seg["path"],
                    require_ocr_text=["END OF GAME", "END OF", "END", "GAME"],
                )
                if res["detected"]:
                    cand_t = start_s + (end_s - start_s) * 0.5
                    logger.info(
                        "  END OF GAME candidate at ~t=%.1fs (conf=%.2f): %s",
                        cand_t, res["max_conf"], Path(seg["path"]).name,
                    )
                    if eog_t_first is None:
                        eog_t_first = cand_t  # record first occurrence
                    eog_t    = cand_t          # overwrite → last occurrence
                    eog_conf = res["max_conf"]
                    eog_name = Path(seg["path"]).name

            if eog_t is None:
                logger.info("  No END OF GAME screen found in %s", norm_clip.name)
                continue

            logger.info(
                "  END OF GAME screen — using last hit at ~t=%.1fs (conf=%.2f): %s",
                eog_t, eog_conf, eog_name,
            )

            # Discard any regulation_end that fired at/after the EOG screen —
            # that is a clock OCR false positive in the celebration footage.
            regulation_end_t = _clip_regulation_end_t.get(clip_stem)
            if regulation_end_t is not None and regulation_end_t >= eog_t - 5.0:
                logger.warning(
                    "  regulation_end_t (%.1fs) >= eog_t-5s (%.1fs) — "
                    "discarding as false positive, clearing segment tags",
                    regulation_end_t, eog_t,
                )
                for (_, _, seg) in seg_times_eog:
                    seg.pop("regulation_end", None)
                    seg.pop("game_end_trim_start_s", None)
                    seg.pop("game_end_trim_end_s", None)
                _clip_regulation_end_t.pop(clip_stem, None)

            # Clear stale game_end tags before re-tagging.
            # Guard regulation_end-tagged segments — they share the same trim
            # key names and must NOT have their trim stripped here.
            for (_, _, seg) in seg_times_eog:
                if seg.get("regulation_end"):
                    continue
                seg.pop("game_end", None)
                seg.pop("game_end_trim_start_s", None)
                seg.pop("game_end_trim_end_s", None)
                seg.pop("game_end_score", None)

            # Tag game_end window: a tight window around the first EOG screen
            # appearance so OT highlights remain as separate clips and post-game
            # menu navigation is excluded.
            # window_start = eog_t_first − PRE_EOG_S  (30s of pre-screen content)
            # clip_end_abs = eog_t_first + POST_EOG_S (30s after screen first appears)
            # Using eog_t_first (not eog_t_last) for both anchors ensures the
            # clip shows the EOG screen appearing, not the player navigating menus.
            _eog_anchor = eog_t_first if eog_t_first is not None else eog_t
            window_start = _eog_anchor - PRE_EOG_S
            clip_end_abs = _eog_anchor + POST_EOG_S
            logger.info(
                "  game_end window anchored to first EOG hit (t=%.1fs): [%.1fs, %.1fs]",
                _eog_anchor, window_start, clip_end_abs,
            )
            tagged = 0
            tagged_times_eog: list[tuple[float, dict]] = []
            for (start_s, end_s, seg) in seg_times_eog:
                if seg.get("regulation_end"):
                    logger.info(
                        "  game_end skip (regulation_end): %s  [t=%.1f–%.1fs]",
                        Path(seg["path"]).name, start_s, end_s,
                    )
                    continue  # don't absorb the 3RD-buzzer clip into game_end
                if end_s > window_start and start_s < clip_end_abs:
                    seg["game_end"]       = True
                    seg["clock_event"]    = "game_end"
                    seg["game_end_score"] = 1.0
                    tagged_times_eog.append((start_s, seg))
                    tagged += 1
                    logger.info(
                        "  Tagged game_end: %s  [t=%.1f–%.1fs]",
                        Path(seg["path"]).name, start_s, end_s,
                    )
            if tagged_times_eog:
                tagged_times_eog.sort(key=lambda x: x[0])
                first_start_s, first_seg = tagged_times_eog[0]
                first_seg["game_end_trim_start_s"] = max(0.0, window_start - first_start_s)
                first_seg["game_end_trim_end_s"]   = clip_end_abs - first_start_s
                logger.info(
                    "  game_end lead trim: start=%.1fs end=%.1fs (within first segment)",
                    first_seg["game_end_trim_start_s"], first_seg["game_end_trim_end_s"],
                )
            logger.info(
                "Game-end clip: window=[%.1fs, %.1fs], %d segment(s) tagged",
                window_start, clip_end_abs, tagged,
            )
    else:
        logger.info("Skipping END OF GAME screen detection (template not found)")

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

    # ── Step 4f: Chain goal → celebration → replay; scoring_chance → replay ──
    logger.info("━━━  Step 4f: Chaining goal sequences  ━━━")
    results = _chain_goal_sequences(results)
    logger.info("━━━  Step 4f.5: Chaining scoring chance replay sequences  ━━━")
    results = _chain_scoring_chance_sequences(results)
    logger.info("━━━  Step 4f.6: Trimming scoring chance clips  ━━━")
    results = _trim_scoring_chances(results)

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
