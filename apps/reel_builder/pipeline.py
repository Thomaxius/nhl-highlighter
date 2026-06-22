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
import os
import time
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "configs" / "config.yaml"
with _CONFIG_PATH.open() as _f:
    _CFG = yaml.safe_load(_f)

from src.ingestion.ingest import ingest_clips
from src.preprocessing.scene_splitter import split_into_scenes
from src.preprocessing.audio_analyzer import extract_audio, detect_energy_spikes, score_segments_by_audio
from src.detection.classifier import HighlightClassifier
from src.detection.banner_detector import BannerDetector
from src.detection.game_clock_detector import GameClockDetector
from src.detection.vs_screen_detector import VsScreenDetector
from src.detection.pause_menu_detector import PauseMenuDetector
from src.detection.stats_detector import detect_stats_screens, save_stats_csv, format_description
from src.detection.star_detector import detect_star_screens, StarScreen as StarScreenResult
from src.assembly.reel_builder import build_reel
from src.progress import ETAReporter, format_duration

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


def _chain_goal_sequences(
    results: list[dict],
    max_lookahead: int = _CFG["goal_chaining"]["max_lookahead"],
    max_gap_s: float = _CFG["goal_chaining"]["max_gap_s"],
) -> list[dict]:
    """
    After a banner-detected or faceoff-inferred goal segment, look ahead and
    tag the immediately following celebration and replay segments so they are
    included in the reel in order: goal → celebration → replay.

    The in-game replay is one continuous clip that the scene-splitter frequently
    chops into short pieces the classifier labels inconsistently as `goal_replay`
    or `other_replay`. Once the replay run has begun (a `celebration` or
    `goal_replay` has been chained, or a leading `other_replay` directly follows
    the goal with no real gap), subsequent `other_replay` fragments are also
    absorbed so the replay isn't truncated. Inferred goals / scoring chances often
    have no celebration cutscene, so their replay begins immediately as
    `other_replay` — that leading fragment starts the run.

    Stops chaining when:
    - A faceoff_cutscene segment is reached (hard stop — next faceoff is starting)
    - A bare `faceoff` segment accumulates too much non-highlight gap (treated as
      a passable element; goal sequences sometimes contain a stray faceoff label
      before the cutscene arrives)
    - An `other` segment appears that is longer than 10s (not just a transition flash)
    - A second goal is detected
    - MAX_LOOKAHEAD segments have been checked
    """
    FOLLOW_LABELS = {"celebration", "goal_replay"}
    MAX_LOOKAHEAD = max_lookahead
    MAX_GAP_S = max_gap_s

    goal_indices = [i for i, r in enumerate(results) if r.get("banner_detected") or r.get("inferred_goal")]

    for goal_idx in goal_indices:
        g = results[goal_idx]
        base_score = g.get("score", g.get("confidence", 1.0))
        order = 1
        accumulated_gap_s = 0.0
        replay_started = False

        for j in range(1, MAX_LOOKAHEAD + 1):
            next_idx = goal_idx + j
            if next_idx >= len(results):
                break

            seg = results[next_idx]

            # Always stop at another goal
            if seg.get("banner_detected") or seg.get("inferred_goal"):
                break

            # faceoff_cutscene = hard stop: the next faceoff sequence is beginning
            if seg.get("label") == "faceoff_cutscene":
                logger.info("  Chain stopped — faceoff_cutscene: %s", Path(seg["path"]).name)
                break

            label = seg.get("label")
            # Once the replay run has begun, also absorb subsequent `other_replay`
            # fragments. The in-game replay is one continuous clip that the
            # scene-splitter often chops into short pieces the classifier labels
            # inconsistently as goal_replay / other_replay; dropping the
            # other_replay pieces truncates the replay. A leading `other_replay`
            # that directly follows the goal (no real >3s gap yet) also starts the
            # run — inferred goals / scoring chances frequently have no celebration
            # cutscene, so their replay begins straight away as `other_replay`.
            is_follow = label in FOLLOW_LABELS or (
                label == "other_replay" and (replay_started or accumulated_gap_s == 0.0)
            )
            if is_follow:
                seg["chain_order"] = order
                seg["chain_goal_idx"] = goal_idx
                seg["score"] = base_score - (order * 0.01)
                seg["confidence"] = max(seg.get("confidence", 0.0), 0.9)
                logger.info(
                    "  Chained %s (order %d, goal %d): %s",
                    label, order, goal_idx, Path(seg["path"]).name,
                )
                if label in {"celebration", "goal_replay", "other_replay"}:
                    replay_started = True
                order += 1
                accumulated_gap_s = 0.0  # reset gap counter once we find a highlight

            else:
                # other, faceoff, or anything else.
                duration = _get_clip_duration(seg["path"])
                if duration <= 3.0:
                    logger.info("  Skipping short transition (%.1fs): %s", duration, Path(seg["path"]).name)
                    continue
                # Before the celebration/replay run begins, an in-game `other`
                # segment here is the on-ice reaction right after the goal — the
                # goal moment often lands at the very end of the goal scene, so
                # this is the "few seconds after the goal" before the cutscene.
                # The classifier usually tags it `other`, which previously got
                # dropped and caused a hard cut from goal → celebration. Bridge
                # it into the sequence instead, bounded by MAX_GAP_S so a long
                # stoppage can't balloon the clip.
                if not replay_started and accumulated_gap_s + duration <= MAX_GAP_S:
                    seg["chain_order"] = order
                    seg["chain_goal_idx"] = goal_idx
                    seg["score"] = base_score - (order * 0.01)
                    seg["confidence"] = max(seg.get("confidence", 0.0), 0.9)
                    logger.info(
                        "  Bridged in-game footage (order %d, goal %d): %s",
                        order, goal_idx, Path(seg["path"]).name,
                    )
                    order += 1
                    accumulated_gap_s += duration
                    continue
                accumulated_gap_s += duration
                if accumulated_gap_s > MAX_GAP_S:
                    logger.info("  Chain stopped — %.1fs gap exceeded %.0fs limit", accumulated_gap_s, MAX_GAP_S)
                    break

    return results


def _drop_empty_inferred_goals(results: list[dict]) -> list[dict]:
    """
    Demote faceoff-inferred goals that chained no celebration / replay.

    Banner-confirmed goals are trusted even when they appear solo, but an
    *inferred* goal relies entirely on the surrounding post-goal sequence to be
    real: every genuine goal is followed by a celebration and/or replay before
    the next faceoff. When _chain_goal_sequences attaches zero follow-up
    segments to an inferred goal, the inference was almost certainly a false
    positive (e.g. a scoring chance / shot followed straight by a faceoff with
    no goal celebration). Such goals would otherwise emit a bare, contextless
    clip, so we demote them back to `other`.
    """
    chained_targets = {
        r["chain_goal_idx"] for r in results if r.get("chain_goal_idx") is not None
    }
    dropped = 0
    for idx, seg in enumerate(results):
        if seg.get("inferred_goal") and not seg.get("banner_detected"):
            if idx not in chained_targets:
                seg["inferred_goal"] = False
                seg["label"] = "other"
                logger.info(
                    "  Dropped empty inferred goal (no celebration/replay chained): %s",
                    Path(seg["path"]).name,
                )
                dropped += 1
    if dropped:
        logger.info("  Dropped %d empty inferred goal(s).", dropped)
    return results


def _prior_context(results: list[dict], i: int) -> str:
    """
    Classify what precedes results[i]: "normal" play, the "replay_tail" of a
    prior goal, or an inter-period "intermission".

    Walk backward through the results list and stop at the first meaningful
    boundary, then classify the candidate accordingly:

        game_start / period_start first
            → puck has already dropped in this period; normal play
        period_end first (no puck-drop seen between it and the candidate)
            → candidate is in the inter-period intermission
        faceoff / faceoff_cutscene first
            → play resumed after a goal; prior sequence is closed → normal
        banner_detected / inferred_goal first
            → still inside that goal's replay tail
        no boundary found (start of file)
            → very early in the recording; treat as normal play
    """
    for back_idx in range(i - 1, -1, -1):
        prev = results[back_idx]
        if prev.get("game_start") or prev.get("period_start"):
            return "normal"
        if prev.get("period_end"):
            return "intermission"
        if prev.get("label") in {"faceoff_cutscene", "faceoff"}:
            return "normal"
        if prev.get("banner_detected") or prev.get("inferred_goal"):
            return "replay_tail"
    return "normal"


def _rescue_demoted_goals(results: list[dict]) -> list[dict]:
    """
    Re-label high-confidence demoted goals as scoring chances.

    Step 4c demotes every goal-labelled segment without a banner to `other`,
    flagging it sc_rescue_candidate when the ML confidence met the rescue
    threshold. Real goals among them are recovered by faceoff-pattern
    inference; what remains is usually genuine goal-like play that never
    produced a banner — i.e. a scoring chance worth showing. Rescue those as
    `scoring_chance`, unless a later step revealed the segment is structural
    content (period/game boundaries, intermission, pause menu) or the replay
    tail of a goal that is already in the reel.

    Must run after _drop_empty_inferred_goals so candidates that were
    promoted to inferred goals and then dropped fall back to scoring_chance
    rather than disappearing entirely.
    """
    rescued = 0
    for i, seg in enumerate(results):
        if not seg.get("sc_rescue_candidate"):
            continue
        # Recovered as a real goal via inference (or otherwise re-labelled)
        if seg.get("label") != "other" or seg.get("banner_detected") or seg.get("inferred_goal"):
            continue

        skip_reason = None
        if seg.get("has_menu"):
            skip_reason = "pause menu"
        elif seg.get("intro") or seg.get("game_start") or seg.get("period_start"):
            skip_reason = "game/period start"
        elif seg.get("period_end") or seg.get("game_end") or seg.get("regulation_end"):
            skip_reason = "game/period end"
        else:
            context = _prior_context(results, i)
            if context == "replay_tail":
                skip_reason = "replay tail of a prior goal"
            elif context == "intermission":
                skip_reason = "inter-period intermission"
        if skip_reason:
            logger.info(
                "  Rescue skipped (%s): %s",
                skip_reason, Path(seg["path"]).name,
            )
            continue

        seg["label"] = "scoring_chance"
        logger.info(
            "  Rescued demoted goal → scoring_chance: %s (conf=%.0f%%)",
            Path(seg["path"]).name, seg.get("confidence", 0.0) * 100,
        )
        rescued += 1
    if rescued:
        logger.info("  Rescued %d demoted goal(s) as scoring chance(s).", rescued)
    return results


def _get_clip_duration(video_path) -> float:
    """Return the duration of a video clip in seconds."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps


def _audio_peak_time(video_path) -> float | None:
    """Return the timestamp (s) of the loudest moment in a clip's audio.

    Used to anchor scoring-chance trims on the actual action (shot / save /
    crowd reaction). Returns None if the audio can't be read or is silent.
    """
    try:
        import subprocess
        import librosa
        import numpy as np
        # Decode audio with ffmpeg directly into a float32 array. librosa.load()
        # would route .mp4 through soundfile (which can't read it) and fall back
        # to audioread — deprecated in librosa 0.10 and removed in 1.0, and the
        # source of the noisy "PySoundFile failed" warnings. ffmpeg is already a
        # hard dependency, so pipe raw mono PCM and let librosa do only the math.
        sr = 22050
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video_path),
             "-map", "a:0?", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
            capture_output=True,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        y = np.frombuffer(proc.stdout, dtype=np.float32)
        if y.size == 0:
            return None
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        if rms.size == 0:
            return None
        peak_frame = int(np.argmax(rms))
        return float(librosa.frames_to_time(peak_frame, sr=sr, hop_length=hop))
    except Exception as exc:
        logger.debug("Audio-peak detection failed for %s: %s", video_path, exc)
        return None


def _trim_chance_clips(
    results: list[dict],
    pre_s: float = 5.0,
    post_s: float = 3.0,
    min_trim_len_s: float = 12.0,
) -> list[dict]:
    """Trim scoring-chance AND faceoff-inferred-goal leads to a window around
    their loudest moment.

    Banner goals are anchored to the GOAL banner and trimmed tight; scoring
    chances and *inferred* goals have no banner anchor, so their full ~45s scene
    would otherwise play in full (a weak faceoff-pattern inference is usually
    just a chance). Anchor on the audio peak (the shot/save) and keep
    [peak-pre, peak+post]. The reel builder applies this window to the lead clip
    only — chained celebration/replays still follow. Banner goals are never
    touched here (they roll naturally to the scene end). Clips already shorter
    than min_trim_len_s, already carrying a trim window, or with no detectable
    audio peak are left as-is.
    """
    trimmed = 0
    for seg in results:
        is_sc = seg.get("label") == "scoring_chance"
        is_inferred = (
            seg.get("label") == "goal"
            and seg.get("inferred_goal")
            and not seg.get("banner_detected")
        )
        if not (is_sc or is_inferred):
            continue
        if "trim_start_s" in seg and "trim_end_s" in seg:
            continue
        dur = _get_clip_duration(seg["path"])
        if dur <= min_trim_len_s:
            continue
        peak = _audio_peak_time(seg["path"])
        if peak is None:
            continue
        start = max(0.0, peak - pre_s)
        end = min(dur, peak + post_s)
        if end - start < 2.0:
            continue
        seg["trim_start_s"] = start
        seg["trim_end_s"] = end
        logger.info(
            "  Trimmed %s to [%.1f–%.1fs] around audio peak %.1fs: %s",
            "inferred goal" if is_inferred else "scoring_chance",
            start, end, peak, Path(seg["path"]).name,
        )
        trimmed += 1
    if trimmed:
        logger.info("  Trimmed %d chance/inferred clip(s) to the action.", trimmed)
    return results


def _infer_goals_from_faceoff_pattern(
    results: list[dict],
    max_lookahead: int = _CFG["goal_inference"]["max_lookahead"],
    min_sc_confidence: float = _CFG["goal_inference"]["min_sc_confidence"],
    min_faceoff_confidence: float = _CFG["goal_inference"]["min_faceoff_confidence"],
) -> list[dict]:
    """
    Infer goals from the post-goal sequence when banner detection fails.

    After a real goal, the game always transitions to:
        (celebration) → (replay) → faceoff_cutscene → faceoff

    If a `scoring_chance` segment (or a segment the ML labelled `goal` but
    banner-gating demoted to `other`) is followed by `faceoff_cutscene`
    within the lookahead window, it is promoted to `goal` (inferred_goal=True).
    These inferred goals are then chained by _chain_goal_sequences.
    """
    MAX_LOOKAHEAD = max_lookahead
    MIN_SC_CONF = min_sc_confidence

    promoted = 0
    for i, seg in enumerate(results):
        # Skip already-confirmed goals
        if seg.get("banner_detected") or seg.get("inferred_goal"):
            continue
        # A period-end buzzer moment looks structurally identical to a post-goal
        # sequence (replay → efaceoff cutscene) — never promote it as a goal
        if seg.get("period_end") or seg.get("period_start") or seg.get("game_start"):
            continue
        # Candidate: ML thought this was a goal or scoring chance
        ml = seg.get("ml_label", seg.get("label"))
        if ml not in {"goal", "scoring_chance"}:
            continue
        if seg.get("label") not in {"scoring_chance", "other"}:
            continue
        # Weak scoring-chance predictions are unlikely to be real goals
        if ml == "scoring_chance" and seg.get("confidence", 0.0) < MIN_SC_CONF:
            continue

        # Backward scan: determine whether this candidate is in normal play,
        # inside a prior goal's replay tail, or in an inter-period intermission.
        context = _prior_context(results, i)
        if context == "replay_tail":
            logger.info(
                "  Skipping faceoff-pattern inference — replay tail of a prior goal: %s",
                Path(seg["path"]).name,
            )
            continue
        if context == "intermission":
            logger.info(
                "  Skipping faceoff-pattern inference — inter-period intermission: %s",
                Path(seg["path"]).name,
            )
            continue

        # Walk forward using virtual steps: a consecutive run of goal_replay /
        # other_replay segments counts as ONE virtual step regardless of how
        # many individual short clips the scene-splitter produced.
        virtual_step = 0
        actual_idx = i + 1
        in_replay_run = False
        while virtual_step < MAX_LOOKAHEAD and actual_idx < len(results):
            nxt = results[actual_idx]
            actual_idx += 1

            is_replay = nxt.get("label") in {"goal_replay", "other_replay"}
            if is_replay:
                if not in_replay_run:
                    # First segment of a new replay run — costs one virtual step
                    in_replay_run = True
                    virtual_step += 1
                # Subsequent segments in the same run are free
            else:
                in_replay_run = False
                virtual_step += 1

            # faceoff_cutscene = normal post-goal sequence
            # faceoff directly = player skipped the cutscene
            if nxt.get("label") in {"faceoff_cutscene", "faceoff"} and nxt.get("confidence", 0.0) >= min_faceoff_confidence:
                seg["label"] = "goal"
                seg["inferred_goal"] = True
                seg["confidence"] = max(seg.get("confidence", 0.0), 0.90)
                seg["score"] = max(seg.get("score", seg["confidence"]), 0.90)
                trigger = nxt.get("label")
                logger.info(
                    "  Faceoff-pattern goal inferred: %s  ml=%s conf=%.0f%%  (%s at virtual +%d / actual +%d)",
                    Path(seg["path"]).name, ml, seg["confidence"] * 100, trigger,
                    virtual_step, actual_idx - i - 1,
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
    checkpoint: str = "apps/shared/models/checkpoints/videomae_nhl",
    music_path: str | None = None,
    max_clips: int = _CFG["pipeline"]["max_clips"],
    min_confidence: float = _CFG["pipeline"]["min_confidence"],
    sc_min_confidence: float = _CFG["pipeline"]["sc_min_confidence"],
    sc_rescue_confidence: float = _CFG["pipeline"]["sc_rescue_confidence"],
) -> None:
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    processed_dir = Path("apps/shared/data/processed")
    segments_dir = Path("apps/shared/data/processed/segments")
    audio_dir = Path("apps/shared/data/processed/audio")

    # Console-only ETA (learned from past run logs) + overall timer.
    _run_start = time.time()
    _eta = ETAReporter.attach()

    # ── Step 1: Ingest & normalise ───────────────────────────────────────────
    logger.info("━━━  Step 1: Ingesting clips  ━━━")
    normalised = ingest_clips(input_dir, processed_dir)
    logger.info("Normalised %d clips.", len(normalised))

    # Total frame count of the normalised clip(s): drives the frame-rate-based
    # ETA for frame-bound steps (e.g. scene detection) and is parsed back out of
    # the logs to learn the typical processing fps.
    import cv2 as _cv2_fc
    _total_frames = 0
    for _clip in normalised:
        _cap = _cv2_fc.VideoCapture(str(_clip))
        _total_frames += int(_cap.get(_cv2_fc.CAP_PROP_FRAME_COUNT))
        _cap.release()
    if _total_frames > 0:
        logger.info("Clip frames: %d", _total_frames)
        _eta.set_frame_count(_total_frames)

    # ── Step 2: Scene splitting ──────────────────────────────────────────────
    logger.info("━━━  Step 2: Splitting scenes  ━━━")
    all_segments: list[Path] = []
    for clip in normalised:
        segs = split_into_scenes(
            clip,
            segments_dir / clip.stem,
            threshold=18.0,
            max_scene_duration_s=45.0,
        )
        all_segments.extend(segs)
    logger.info("Total segments: %d", len(all_segments))
    # Refine the ETA now that the segment count is known (classify time scales with it).
    _eta.set_segment_count(len(all_segments))

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

    # ── Steps 4b + 4d + 4d.5: Raw-video scans (parallel) ─────────────────────
    # All three steps scan the same normalised video for independent signals.
    # Initialise all detectors once and fire their scan_video / find_matches
    # calls concurrently so total wall time equals the slowest scan rather
    # than the sum of all three.
    import cv2 as _cv2
    from concurrent.futures import ThreadPoolExecutor as _ScanPool

    _bsc_templates  = sorted(Path("apps/reel_builder/configs").glob("goal_banner_template*.png"))
    _bsc_clock_path = Path("apps/reel_builder/configs/game_end_template.png")
    _bsc_vs_path    = Path("apps/reel_builder/configs/vs_screen_template3.png")
    _bsc_gc_cfg     = _CFG.get("game_clock", {})

    _bsc_banner = BannerDetector() if _bsc_templates else None
    _bsc_clock  = GameClockDetector(
        template_path=_bsc_clock_path,
        min_game_end_conf=_bsc_gc_cfg.get("min_game_end_conf", 0.85),
        min_period_end_conf=_bsc_gc_cfg.get("min_period_end_conf", 0.85),
        min_period_start_conf=_bsc_gc_cfg.get("min_period_start_conf", 0.85),
        period_end_confirm_window_s=_bsc_gc_cfg.get("period_end_confirm_window_s", 10.0),
        period_end_confirm_min_hits=int(_bsc_gc_cfg.get("period_end_confirm_min_hits", 2)),
        period_end_confirm_interval_s=_bsc_gc_cfg.get("period_end_confirm_interval_s", 1.0),
    ) if _bsc_clock_path.exists() else None
    _bsc_vs = VsScreenDetector(template_path=_bsc_vs_path) if _bsc_vs_path.exists() else None

    _raw_scan: dict[str, dict] = {}  # clip_stem → {banner_times, clock_events, vs_hits}
    for _bsc_clip in normalised:
        _bsc_stem = _bsc_clip.stem
        if not any(_bsc_stem in str(r["path"]) for r in results):
            continue
        _bsc_cap = _cv2.VideoCapture(str(_bsc_clip))
        _bsc_dur = _bsc_cap.get(_cv2.CAP_PROP_FRAME_COUNT) / (_bsc_cap.get(_cv2.CAP_PROP_FPS) or 30.0)
        _bsc_cap.release()
        logger.info(
            "━━━  Parallel scan: %s  (banner=%s  clock=%s  vs=%s)  ━━━",
            _bsc_clip.name, bool(_bsc_banner), bool(_bsc_clock), bool(_bsc_vs),
        )
        # max_workers=2 (not 3): each scan holds its own FFmpeg decoder whose
        # frame buffers add up — three concurrent 1080p decodes OOM-killed the
        # pipeline on the 8 GB server. The VS scan is capped at 300 s, so it
        # just queues behind whichever full-length scan finishes first.
        # --local (NHL_LOCAL=1) lifts this to 4 so all three scans run at once.
        _scan_workers = 4 if os.environ.get("NHL_LOCAL") == "1" else 2
        with _ScanPool(max_workers=_scan_workers) as _bsc_pool:
            _bsc_bf = _bsc_pool.submit(_bsc_banner.scan_video, _bsc_clip, 1.0) if _bsc_banner else None
            _bsc_cf = _bsc_pool.submit(_bsc_clock.scan_video,  _bsc_clip)      if _bsc_clock  else None
            # VS screen only appears in the first few minutes of a recording;
            # cap the search at 300 s to avoid scanning the full game-length video.
            _bsc_vf = _bsc_pool.submit(
                _bsc_vs.find_matches, _bsc_clip, 0.5, min(300.0, _bsc_dur), 0.0,
            ) if _bsc_vs else None
        _raw_scan[_bsc_stem] = {
            "banner_times": _bsc_bf.result() if _bsc_bf is not None else [],
            "clock_events": _bsc_cf.result() if _bsc_cf is not None else [],
            "vs_hits":      _bsc_vf.result() if _bsc_vf is not None else [],
        }

    # ── Step 4b: Apply banner-detection results ──────────────────────────────
    templates = _bsc_templates  # alias used by the guard below
    if templates:
        logger.info("━━━  Step 4b: Banner detection (%d template(s))  ━━━", len(templates))
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

            # Use pre-computed results from the parallel scan phase above.
            banner_times = _raw_scan.get(norm_clip.stem, {}).get("banner_times", [])

            for bt in banner_times:
                for seg_i, (start_s, end_s, seg, seg_fps) in enumerate(seg_ranges):
                    if start_s <= bt < end_s:
                        local_t  = bt - start_s
                        seg_dur  = end_s - start_s
                        seg["banner_detected"]   = True
                        seg["banner_confidence"] = 0.95
                        seg["best_frame"]        = int(local_t * seg_fps)
                        # Trim window centred on the goal frame, clamped to segment
                        seg["trim_start_s"] = max(0.0, local_t - PRE_GOAL_S)
                        seg["trim_end_s"]   = min(seg_dur, local_t + POST_GOAL_S)
                        # If the goal frame sits near the very start of this segment,
                        # the pre-goal window clamps to 0 and the actual shot lives at
                        # the tail of the preceding segment(s). Borrow that footage so
                        # the build-up isn't cut off at the moment of the shot.
                        deficit = PRE_GOAL_S - local_t
                        if deficit > 0.05 and seg_i > 0:
                            borrow: list[dict] = []
                            remaining = deficit
                            bi = seg_i - 1
                            while remaining > 0.05 and bi >= 0:
                                b_start, b_end, b_seg, _b_fps = seg_ranges[bi]
                                b_dur = b_end - b_start
                                take = min(remaining, b_dur)
                                borrow.insert(0, {
                                    "path": str(b_seg["path"]),
                                    "start_s": round(b_dur - take, 3),
                                    "end_s": round(b_dur, 3),
                                })
                                remaining -= take
                                bi -= 1
                            if borrow:
                                seg["pre_goal_borrow"] = borrow
                                logger.info(
                                    "  Pre-goal borrow: %.1fs from %d prior segment(s) → %s",
                                    deficit - remaining, len(borrow),
                                    ", ".join(Path(b["path"]).name for b in borrow),
                                )
                        logger.info(
                            "Banner detected via raw scan: t=%.1fs → %s  "
                            "trim %.1fs→%.1fs",
                            bt, Path(seg["path"]).name,
                            seg["trim_start_s"], seg["trim_end_s"],
                        )
                        break  # each goal event maps to exactly one segment
    else:
        logger.info("Skipping banner detection (no templates found in apps/reel_builder/configs/)")

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
            # No banner = no goal, regardless of model confidence.
            # High-confidence misses are flagged for the scoring-chance rescue
            # pass (step 4f.5) once structural tags are known.
            if r.get("confidence", 0) >= sc_rescue_confidence:
                r["sc_rescue_candidate"] = True
            logger.info(
                "  Demoted → other: %s (conf=%.0f%%, no banner%s)",
                Path(r["path"]).name, r.get("confidence", 0) * 100,
                ", sc-rescue candidate" if r.get("sc_rescue_candidate") else "",
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
    if _bsc_clock is not None:
        logger.info("━━━  Step 4d: Game clock detection  ━━━")
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

            clock_events = _raw_scan.get(norm_clip.stem, {}).get("clock_events", [])

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
                            # Carry the template's declared period (if any) so the
                            # numbering pass can advance monotonically by template
                            # rather than blindly +1 on every (possibly false) hit.
                            if ev.get("period") is not None:
                                seg["clock_period"] = ev["period"]
                            logger.info(
                                "%s at t=%.1fs → %s",
                                ev_type, ev_t, Path(seg["path"]).name,
                            )
                            break

                elif ev_type == "ot_period_end":
                    # OT buzzer (clock near 0:00 with an "OT" label) — a genuine
                    # overtime clock signal. Tag the covering segment so the OT
                    # boundary can be confirmed from the clock, not just from the
                    # highlight classifier (which menus can fool).
                    for (start_s, end_s, seg) in seg_times:
                        if start_s <= ev_t < end_s:
                            seg["ot_period_end"] = True
                            seg["clock_event"] = ev_type
                            logger.info(
                                "OT period-end at t=%.1fs → %s",
                                ev_t, Path(seg["path"]).name,
                            )
                            break
    else:
        logger.info("Skipping game clock detection (apps/reel_builder/configs/game_end_template.png not found)")

    # ── Period numbering: tag every segment with the period it belongs to ─────
    # period 1 = 1st, 2 = 2nd, 3 = 3rd, 4+ = OT periods.
    #
    # Two independent signals can advance the period counter:
    #   period_end   — the buzzer fired at the END of a period.  The segment
    #                  itself is still in the current period; everything after
    #                  it moves to the next.
    #   period_start — the clock read 20:00 for a NEW period (puck drop).  The
    #                  segment with this flag IS the first of the new period, so
    #                  we advance BEFORE assigning period_number to it.  This
    #                  keeps things correct even when a period_end was missed or
    #                  suppressed as a false positive.
    _period = 1
    for _r in results:
        # Advance BEFORE numbering when we hit a puck-drop start signal.
        # Prefer the template's DECLARED period (clock_period) and take the max
        # so the counter is monotonic: a single false/duplicate period_start
        # (e.g. a "period 2" template matching mid-period, or a stray "period 1"
        # match late in the game) can no longer over-advance or rewind it.
        if _r.get("period_start") or _r.get("game_start"):
            declared = _r.get("clock_period")
            if declared is not None:
                _period = max(_period, min(int(declared), 3))
            elif _r.get("period_start") and _period < 3:
                # No declared period (OCR-only hit) — fall back to a single step.
                _period = min(_period + 1, 3)
        _r["period_number"] = _period
        if _r.get("period_end"):
            _period = min(_period + 1, 3)
        elif _r.get("regulation_end"):
            _period = 4  # first segment after regulation buzzer is OT

    # ── Step 4d.2: Infer goals from faceoff pattern ───────────────────────────
    # Runs AFTER clock detection so that period_end / period_start / game_start
    # flags are already set. The guard inside the function — which skips
    # period-end buzzer segments — is only effective when those flags exist.
    logger.info("\u2501\u2501\u2501  Step 4d.2: Faceoff-pattern goal inference  \u2501\u2501\u2501")
    results = _infer_goals_from_faceoff_pattern(results)

    # ── Step 4d.5: VS screen → intro segment tagging ─────────────────────────
    # Clip starts 6 s after the VS screen first appears and runs through
    # game_start (20:00 clock) + 7 s of actual play.
    # Tagged segments carry trim offsets so the reel builder can cut precisely:
    #   intro_clip_start_s — seconds from the first tagged segment's start
    #   intro_clip_end_s   — seconds from the first tagged segment's start
    PRE_VS_SKIP_S  = 6.0   # skip this many seconds after VS screen first appears
    POST_FACEOFF_S = 7.0   # seconds of actual play to include after 20:00
    FALLBACK_INTRO_S = 30.0  # fallback intro length when no VS screen is found
    FALLBACK_INTRO_MAX_START_S = 20.0  # only treat an early 'other' as a cold-open intro if it begins this soon
    MIN_FALLBACK_INTRO_S = 4.0  # skip a fabricated intro shorter than this (after goal-overlap clamp)
    if _bsc_vs is not None:
        logger.info("━━━  Step 4d.5: VS screen / intro detection  ━━━")
        fallback_pause_template = Path("apps/reel_builder/configs/pause_menu_template.png")
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
            # vs_search_window kept for informational log messages below; the actual
            # scan was pre-computed in the parallel phase above (capped at 300 s).
            if game_start_t is not None:
                vs_search_window = game_start_t + 120.0
                logger.info(
                    "  VS screen scanned up to min(300s, %.1fs) (game_start=%.1fs + 120s buffer)",
                    vs_search_window, game_start_t,
                )
            else:
                vs_search_window = 300.0
                logger.info("  VS screen scanned up to 300s (no game_start detected)")

            vs_hits = _raw_scan.get(norm_clip.stem, {}).get("vs_hits", [])
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
                # A genuine intro / cold-open sits at the very start of the clip.
                # If the first eligible 'other' segment begins well into the clip,
                # the recording simply has no intro (e.g. a mid-game cut) — don't
                # fabricate one from arbitrary gameplay, which could duplicate a
                # later highlight.
                if fallback_seg[0] > FALLBACK_INTRO_MAX_START_S:
                    logger.info(
                        "  No VS screen and first 'other' at t=%.1fs is past the %.0fs cold-open window — no intro for %s",
                        fallback_seg[0], FALLBACK_INTRO_MAX_START_S, norm_clip.name,
                    )
                    continue
                intro_mode = "fallback"
                intro_start = fallback_seg[0]
                intro_end = min(cum_t, intro_start + FALLBACK_INTRO_S)
                # Never let the cold-open extend into a goal sequence — that would
                # show the goal/replay once as a fake intro and again at its real
                # chronological position.
                goal_starts = [
                    s for (s, e, sg) in seg_times2
                    if (sg.get("banner_detected") or sg.get("inferred_goal")) and s > intro_start
                ]
                if goal_starts:
                    next_goal_s = min(goal_starts)
                    if next_goal_s < intro_end:
                        intro_end = next_goal_s
                if intro_end - intro_start < MIN_FALLBACK_INTRO_S:
                    logger.info(
                        "  Fallback intro for %s would be <%.0fs after goal-overlap clamp — skipping intro",
                        norm_clip.name, MIN_FALLBACK_INTRO_S,
                    )
                    continue
                logger.info(
                    "  No VS screen found in %s (searched %.1fs) — fallback cold-open at t=%.1f–%.1fs",
                    norm_clip.name, vs_search_window, intro_start, intro_end,
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
                    "Fallback intro: cold-open [t=%.1f–%.1fs]: tagged %d segment(s)",
                    intro_start, intro_end, tagged,
                )
    else:
        logger.info("Skipping VS screen detection (apps/reel_builder/configs/vs_screen_template3.png not found)")

    # ── Step 4d.9: Pause-menu detection ────────────────────────────────────
    # Runs AFTER intro and game_end tagging so those guards work correctly.
    # Segments where the ESC/pause menu is visible are demoted to "other".
    # Skips intro/game_end segments (those are assembled directly, not scored).
    # Special case: if the END OF GAME menu template (pause_menu_template_end_of_game.png)
    # fires, the segment is tagged game_end=True instead of being demoted.
    pause_template = Path("apps/reel_builder/configs/pause_menu_template.png")
    if pause_template.exists():
        logger.info("━━━  Step 4d.9: Pause-menu detection  ━━━")
        pause_detector = PauseMenuDetector(template_path=pause_template)
        eog_template_path = Path("apps/reel_builder/configs/pause_menu_template_end_of_game.png")
        eog_detector = PauseMenuDetector(template_path=eog_template_path) if eog_template_path.exists() else None
        if eog_detector:
            logger.info("  EOG menu detector loaded (%s)", eog_template_path.name)
        paused_count = 0
        eog_tagged = 0
        for r in results:
            if r.get("game_end") or r.get("intro"):
                continue
            res = pause_detector.detect(r["path"])
            if res["detected"]:
                # Before demoting, check if this is actually the END OF GAME menu.
                is_eog = False
                if eog_detector is not None:
                    eog_res = eog_detector.detect(r["path"])
                    if eog_res["detected"]:
                        is_eog = True
                        logger.info(
                            "  END OF GAME menu → tagged game_end: %s (conf=%.2f)",
                            Path(r["path"]).name, eog_res["max_conf"],
                        )
                        r["game_end"]       = True
                        r["clock_event"]    = "game_end"
                        r["game_end_score"] = 1.0
                        eog_tagged += 1
                if not is_eog:
                    logger.info(
                        "  Pause menu → demoted: %s (conf=%.2f)",
                        Path(r["path"]).name, res["max_conf"],
                    )
                    r["label"]    = "other"
                    r["has_menu"] = True
                    paused_count += 1
        if paused_count:
            logger.info("  Demoted %d segment(s) containing pause menu.", paused_count)
        if eog_tagged:
            logger.info("  Tagged %d segment(s) as END OF GAME via menu template.", eog_tagged)

        # The END-OF-GAME menu template also matches inter-period intermission
        # menus (same scoreboard layout), so the 1st/2nd intermissions get
        # falsely tagged game_end and the reel ends up showing a "1ST
        # Intermission" screen. Only the LAST cluster of game_end segments is the
        # real end of game — demote the earlier (intermission) clusters back to
        # menu so they're excluded.
        ge_idx = [i for i, r in enumerate(results) if r.get("game_end")]
        if len(ge_idx) > 1:
            _CLUSTER_GAP = 5   # >5 segments apart ⇒ a separate cluster
            last_cluster_start = ge_idx[0]
            for a, b in zip(ge_idx, ge_idx[1:]):
                if b - a > _CLUSTER_GAP:
                    last_cluster_start = b
            demoted_ge = 0
            for i in ge_idx:
                if i < last_cluster_start:
                    r = results[i]
                    r["game_end"] = False
                    r.pop("clock_event", None)
                    r.pop("game_end_score", None)
                    r["label"] = "other"
                    r["has_menu"] = True
                    demoted_ge += 1
                    logger.info(
                        "  Demoted earlier game_end (intermission menu, not end of game): %s",
                        Path(r["path"]).name,
                    )
            if demoted_ge:
                logger.info("  Kept only the final game_end cluster; demoted %d intermission menu(s).", demoted_ge)
    else:
        logger.info("Skipping pause-menu detection (apps/reel_builder/configs/pause_menu_template.png not found)")

    # ── Step 4f: Chain goal → celebration → replay ───────────────────────────
    logger.info("━━━  Step 4f: Chaining goal sequences  ━━━")
    results = _chain_goal_sequences(results)

    # Demote inferred goals that ended up with no chained celebration/replay —
    # they are almost always false positives and would emit a bare clip.
    results = _drop_empty_inferred_goals(results)

    # ── Step 4f.5: Rescue high-confidence demoted goals as scoring chances ───
    logger.info("━━━  Step 4f.5: Rescuing demoted goals as scoring chances  ━━━")
    results = _rescue_demoted_goals(results)

    # ── Step 4f.6: Trim scoring chances + inferred goals to the action ───────
    logger.info("━━━  Step 4f.6: Trimming chance/inferred clips to audio peak  ━━━")
    results = _trim_chance_clips(results)

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

    # ── Step 6 (+7/7b): Reel build with concurrent stats + star scan ─────────
    # The stats + Three-Stars OCR scans only read the normalised clips, and the
    # reel build spends nearly all its time waiting on FFmpeg subprocesses, so
    # the scans run in a background thread concurrently with the build. They
    # overlap into a single wall-clock block, so one combined header is logged
    # here; results are collected (and logged/saved) after build_reel returns.
    # max_workers=1: the two scans run sequentially but still fully overlap the
    # reel build, while holding only one extra video decoder in memory.
    # --local (NHL_LOCAL=1) lifts this to 2 so both scans run concurrently.
    from concurrent.futures import ThreadPoolExecutor as _StatsPool
    logger.info("━━━  Step 6: Building reel + stats/star scan (concurrent)  ━━━")
    _stats_pool = _StatsPool(max_workers=2 if os.environ.get("NHL_LOCAL") == "1" else 1)
    _stats_future = _stats_pool.submit(
        lambda: [detect_stats_screens(clip) for clip in normalised]
    )
    _stars_future = _stats_pool.submit(
        lambda: [(clip, detect_star_screens(clip)) for clip in normalised]
    )

    _pt_dir = Path("apps/reel_builder/assets/period_transitions")

    # Build the set of period numbers whose boundaries were confirmed by the
    # clock detector.  Only transitions to confirmed periods are injected.
    confirmed_period_boundaries: set[int] = set()
    _OT_HIGHLIGHT_LABELS = {"goal", "celebration", "goal_replay", "other_replay", "scoring_chance"}
    _OT_MIN_HIGHLIGHTS = 5  # how many real OT highlights it takes to confirm OT
    _ot_highlight_count = 0
    _ot_clock_signal = False
    for _r in results:
        if _r.get("period_end"):
            confirmed_period_boundaries.add(_r.get("period_number", 1) + 1)
        if _r.get("period_start") or _r.get("game_start"):
            confirmed_period_boundaries.add(_r.get("period_number", 1))
        # ── Overtime evidence ────────────────────────────────────────────────
        # regulation_end alone just means the 3rd period ended — the game may be
        # over, with only post-game menu/stats screens after it, so injecting an
        # OT transition off a single (mis)classified menu frame is wrong.
        # Treat OT as real only if EITHER the clock detector actually saw an OT
        # period (ot_period_end), OR there are several genuine OT highlights.
        if _r.get("ot_period_end"):
            _ot_clock_signal = True
        if (_r.get("period_number", 1) >= 4
                and _r.get("label") in _OT_HIGHLIGHT_LABELS
                and not _r.get("game_end")
                and not _r.get("regulation_end")
                and not _r.get("has_menu")):
            _ot_highlight_count += 1

    # Confirm every OT period present only when the evidence clears the bar.
    _ot_periods = {_r.get("period_number", 1) for _r in results if _r.get("period_number", 1) >= 4}
    if _ot_periods:
        if _ot_clock_signal or _ot_highlight_count >= _OT_MIN_HIGHLIGHTS:
            confirmed_period_boundaries.update(_ot_periods)
            logger.info(
                "  Overtime confirmed (clock_signal=%s, ot_highlights=%d) — periods %s",
                _ot_clock_signal, _ot_highlight_count, sorted(_ot_periods),
            )
        else:
            logger.info(
                "  Overtime NOT confirmed (clock_signal=%s, ot_highlights=%d < %d) — "
                "skipping OT transition (likely post-game menu footage).",
                _ot_clock_signal, _ot_highlight_count, _OT_MIN_HIGHLIGHTS,
            )

    if confirmed_period_boundaries:
        logger.info("  Confirmed period boundaries: %s", sorted(confirmed_period_boundaries))
    else:
        logger.info("  No confirmed period boundaries — period transitions will be skipped.")

    build_reel(
        segments=results,
        output_path=output_path,
        music_path=music_path,
        max_clips=max_clips,
        min_confidence=min_confidence,
        sc_min_confidence=sc_min_confidence,
        period_transitions_dir=_pt_dir if _pt_dir.exists() else None,
        confirmed_period_boundaries=confirmed_period_boundaries,
    )
    logger.info("Done!  Reel → %s", output_path)

    # ── Step 7: Stats + Star screen detection ───────────────────────────────
    # Scan each normalised clip for end-of-game stats screens and Three Stars.
    # Results are written as CSVs in a per-reel subfolder, the parsed tables
    # are logged in full, and a YouTube description file is saved alongside
    # the reel so the uploader can attach the stats to the video description.
    logger.info("Collecting stats screen results…")
    # Per-video stats folder: exports/{reel_stem}/stats/
    stats_out_dir = output_path.with_suffix("") / "stats"
    all_stats: list = []
    for screens in _stats_future.result():
        if screens:
            saved = save_stats_csv(screens, stats_out_dir)
            all_stats.extend(screens)
            for s, path in zip(screens, saved):
                logger.info(
                    "  %s / %s  (%d rows) → %s",
                    s.team, s.tab, len(s.rows), path,
                )
                # Log the full table
                if s.rows:
                    cols = list(s.rows[0].keys())
                    widths = {c: max(len(c), max(len(r.get(c, "")) for r in s.rows)) for c in cols}
                    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
                    hdr = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
                    logger.info("  %s", sep)
                    logger.info("  %s", hdr)
                    logger.info("  %s", sep)
                    for row in s.rows:
                        logger.info("  | " + " | ".join(row.get(c, "").ljust(widths[c]) for c in cols) + " |")
                    logger.info("  %s", sep)
    if not all_stats:
        logger.info("  No stats screens found.")

    logger.info("Collecting star screen results…")
    all_stars: list = []
    for clip, stars in _stars_future.result():
        if stars:
            all_stars.extend(stars)
            for star in stars:
                logger.info(
                    "  %s STAR: %s  |  Stats: %s  |  t=%.1fs",
                    star.rank,
                    star.player_name or "(unknown)",
                    star.player_stats or "(none)",
                    star.timestamp_s,
                )
        else:
            logger.info("  No star screens found in %s", clip.name)
    _stats_pool.shutdown()

    # Save the YouTube description (title substituted by the uploader)
    description_path = output_path.with_suffix(".txt")
    try:
        desc = format_description(title="__TITLE__", screens=all_stats, stars=all_stars or None)
        description_path.write_text(desc, encoding="utf-8")
        logger.info("  Description template → %s", description_path)
    except Exception as _exc:
        logger.warning("  Could not write description file: %s", _exc)

    logging.getLogger().removeHandler(_eta)
    logger.info("Pipeline completed in %s", format_duration(time.time() - _run_start))


def run_smoke_test(checkpoint: str) -> int:
    """
    Verify the environment can actually run the pipeline, without processing
    any video. Reaching this function already proves every module import
    (cv2, torch, librosa, pytesseract, …) succeeded — what imports alone
    can't prove is that external binaries and required assets are present:
    pytesseract imports fine without the tesseract binary and only fails at
    call time, hours into a run.

    Returns 0 if everything passed, 1 otherwise.
    """
    import shutil
    import subprocess

    failures = 0

    def _check(name: str, fn) -> None:
        nonlocal failures
        try:
            detail = fn()
            logger.info("  OK   %s%s", name, f"  ({detail})" if detail else "")
        except Exception as exc:
            logger.error("  FAIL %s: %s", name, exc)
            failures += 1

    def _binary(cmd: str) -> str:
        path = shutil.which(cmd)
        if not path:
            raise FileNotFoundError(f"'{cmd}' not found in PATH")
        out = subprocess.run([cmd, "-version"], capture_output=True, text=True)
        return out.stdout.splitlines()[0] if out.stdout else path

    logger.info("━━━  Smoke test  ━━━")
    logger.info("Module imports: OK (this script would not have started otherwise)")

    _check("ffmpeg", lambda: _binary("ffmpeg"))
    _check("ffprobe", lambda: _binary("ffprobe"))

    def _tesseract() -> str:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        langs = pytesseract.get_languages(config="")
        if "eng" not in langs:
            raise RuntimeError(f"tesseract {version} found but 'eng' language pack missing")
        return f"tesseract {version}"
    _check("tesseract + eng language", _tesseract)

    def _checkpoint() -> str:
        ckpt = Path(checkpoint)
        if not ckpt.is_dir() or not any(ckpt.iterdir()):
            raise FileNotFoundError(f"model checkpoint missing or empty: {ckpt}")
        return str(ckpt)
    _check("VideoMAE checkpoint", _checkpoint)

    def _templates() -> str:
        cfg_dir = Path("apps/reel_builder/configs")
        banners = list(cfg_dir.glob("goal_banner_template*.png"))
        if not banners:
            raise FileNotFoundError(f"no goal banner templates in {cfg_dir}")
        missing = [
            n for n in (
                "game_end_template.png",
                "vs_screen_template3.png",
                "pause_menu_template.png",
            )
            if not (cfg_dir / n).exists()
        ]
        if missing:
            raise FileNotFoundError(f"missing templates: {', '.join(missing)}")
        return f"{len(banners)} banner template(s) + clock/vs/pause"
    _check("detection templates", _templates)

    if failures:
        logger.error("Smoke test FAILED — %d check(s) failed.", failures)
    else:
        logger.info("Smoke test passed — environment looks runnable.")
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description="NHL 25 Highlight Reel Generator")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--file",   metavar="FILE",   help="Single raw .mp4 to process")
    src.add_argument("--folder", metavar="FOLDER", help="Folder of raw .mp4 clips (default: apps/shared/data/raw)")

    p.add_argument("--output", default="apps/shared/data/exports/reel.mp4", help="Output reel path")
    p.add_argument("--checkpoint", default="apps/shared/models/checkpoints/videomae_nhl")
    p.add_argument("--music", default=None, help="Optional background music file")
    p.add_argument("--max_clips", type=int, default=_CFG["pipeline"]["max_clips"])
    p.add_argument("--min_confidence", type=float, default=_CFG["pipeline"]["min_confidence"])
    p.add_argument("--sc_min_confidence", type=float, default=_CFG["pipeline"]["sc_min_confidence"],
                   help="Min confidence for scoring_chance clips")
    p.add_argument("--sc_rescue_confidence", type=float, default=_CFG["pipeline"]["sc_rescue_confidence"],
                   help="Demoted no-banner goals at/above this become scoring chances")
    p.add_argument("--smoke-test", action="store_true",
                   help="Verify imports, external binaries and required assets, then exit")
    p.add_argument("--local", action="store_true",
                   help="Local mode: use boosted thread/worker counts suited to a "
                        "beefy local machine (defaults are tuned for the 8 GB server). "
                        "Sets NHL_LOCAL=1 for all detection stages.")
    args = p.parse_args()

    # Propagate boosted-parallelism mode to every stage via the environment so
    # the detectors (classifier, banner) and the scan/stats pools all agree.
    if args.local:
        os.environ["NHL_LOCAL"] = "1"
        logger.info("Local mode enabled — boosted parallelism (NHL_LOCAL=1)")

    if args.smoke_test:
        raise SystemExit(run_smoke_test(checkpoint=args.checkpoint))

    if args.file:
        # Wrap the single file in a temp dir so the pipeline's ingest step
        # sees it as a one-file folder without touching the original.
        import tempfile
        file_path = Path(args.file).resolve()
        tmp = tempfile.mkdtemp(prefix="pipeline_single_", dir='.')
        os.link(file_path, Path(tmp) / file_path.name)   # hardlink — instant, no extra disk
        input_dir = tmp
    else:
        input_dir = args.folder or "apps/shared/data/raw"

    log_file = _setup_file_logging(args.output)
    logger.info("Logging to %s", log_file)

    run_pipeline(
        input_dir=input_dir,
        output_path=args.output,
        checkpoint=args.checkpoint,
        music_path=args.music,
        max_clips=args.max_clips,
        min_confidence=args.min_confidence,
        sc_min_confidence=args.sc_min_confidence,
        sc_rescue_confidence=args.sc_rescue_confidence,
    )


if __name__ == "__main__":
    main()
