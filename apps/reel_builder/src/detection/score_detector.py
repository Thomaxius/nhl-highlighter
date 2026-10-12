"""
detection/score_detector.py

Reads the NHL 25 scoreboard HUD via OCR (pytesseract) and detects when
the score changes between consecutive segments — a strong secondary signal
that a goal was scored.

HUD layout (top-left corner, 1920×1080):
    ┌──────────────────────────────────┐
    │ [NHL] SJS  0 │ SOG 6 │ 9:01 1ST │
    │       EDM  1 │ SOG 5 │          │
    └──────────────────────────────────┘

The detector:
  1. Samples frames from a segment
  2. Crops the scoreboard ROI
  3. Runs OCR to extract digit strings
  4. Parses home/away scores
  5. Compares to last known score — if either increases by 1, score_changed=True
  6. Updates internal state for the next call

Usage
-----
    detector = ScoreDetector()
    result = detector.detect_in_segment("segment.mp4")
    # result = {"home": 1, "away": 0, "score_changed": True, "raw_text": "..."}
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Disable FFmpeg hardware-accelerated video decoding (VideoToolbox on macOS).
# Without this, cv2.VideoCapture deadlocks when called after PyTorch/MPS has
# already claimed the GPU — a common hang in the pipeline's Step 4e.
# Must be set before the first VideoCapture open, so module-level is correct.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")

import cv2

logger = logging.getLogger(__name__)

# ── Scoreboard ROI (fractions of 1920×1080) ───────────────────────────────────
# Covers the full scoreboard panel in the top-left corner.
# Tune these if the crop looks off — use ScoreDetector.debug_frame() to preview.
ROI_X = 0.02
ROI_Y = 0.035
ROI_W = 0.2176  # ~21.76% of width → 418px
ROI_H = 0.077   # ~7.7% of height → 83px


# The five tesseract calls per frame are independent subprocesses, so they run
# concurrently (they release the GIL). Shared pool, created lazily.
_OCR_POOL: Optional[ThreadPoolExecutor] = None


def _ocr_pool() -> ThreadPoolExecutor:
    global _OCR_POOL
    if _OCR_POOL is None:
        _OCR_POOL = ThreadPoolExecutor(max_workers=5, thread_name_prefix="score-ocr")
    return _OCR_POOL


class ScoreDetector:
    """
    Detects score changes between consecutive video segments using OCR.

    The detector is **stateful**: call detect_in_segment() on segments in
    chronological order and it will report whenever the score increments.

    Args:
        sample_frames:  Number of frames to sample per segment for OCR.
        upscale:        Factor to upscale the ROI crop before OCR (improves
                        accuracy on small text). Default 3.
    """

    def __init__(
        self,
        sample_frames: int = 12,
        upscale: int = 3,
    ) -> None:
        try:
            import pytesseract  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pytesseract is required for score detection.\n"
                "  pip install pytesseract\n"
                "  brew install tesseract"
            ) from e

        self.sample_frames = sample_frames
        self.upscale = upscale

        # Stateful last-known score — updated on each successful parse
        self._last_home: Optional[int] = None
        self._last_away: Optional[int] = None

        # Stateful last-known shots on goal
        self._last_sog_home: Optional[int] = None
        self._last_sog_away: Optional[int] = None

        # Was the HUD visible (parseable) in the previous segment?
        # Used to suppress false positives when the HUD reappears after being
        # absent — the accumulated count difference looks like a change.
        self._last_hud_visible: bool = False

        logger.info("ScoreDetector ready (sample_frames=%d, upscale=%dx)", sample_frames, upscale)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_in_segment(self, video_path: str | Path, fast: bool = False) -> dict:
        """
        Sample frames from *video_path*, OCR the scoreboard, detect score/SOG changes.

        Intra-segment SOG tracking: each SOG increment detected across consecutive
        frame reads within the segment is counted separately.  This means a segment
        containing 3 shots returns sog_change_count=3 rather than just sog_changed=True.

        Returns:
            {
                "home":             int | None,  # final parsed home score
                "away":             int | None,  # final parsed away score
                "sog_home":         int | None,  # final parsed home SOG
                "sog_away":         int | None,  # final parsed away SOG
                "score_changed":    bool,        # True if score incremented vs last segment
                "sog_changed":      bool,        # True if any SOG increment detected
                "sog_change_count": int,         # number of individual SOG increments seen
                "hud_visible":      bool,        # True if HUD was readable this segment
                "raw_text":         str,         # raw OCR output for debugging
            }
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(total, 1)

        start_frame = min(2, total - 1)
        n_frames = 6 if fast else self.sample_frames
        frame_indices = np.linspace(start_frame, total - 1, n_frames, dtype=int)

        all_texts: list[str] = []
        best_home: Optional[int] = None
        best_away: Optional[int] = None
        best_sog_home: Optional[int] = None
        best_sog_away: Optional[int] = None

        # Running SOG state used for intra-segment tracking.
        # Seeded from last-segment state only when previous segment had a visible HUD
        # (otherwise stored values are stale and would cause false positives).
        run_sog_home: Optional[int] = self._last_sog_home if self._last_hud_visible else None
        run_sog_away: Optional[int] = self._last_sog_away if self._last_hud_visible else None

        sog_change_count = 0
        shot_timestamps: list[float] = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Frame index where run_sog_home/run_sog_away were last *actually*
        # confirmed by a successful OCR read — not just "the previous coarse
        # sample," which may itself have failed to read and left the running
        # state stale from several samples further back. Refining from the
        # wrong (too-late) lo_idx can start the scan after the real
        # transition already happened.
        last_good_idx: Optional[int] = None

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            crop = self._crop_roi(frame)
            text = self._ocr_crop(crop)
            all_texts.append(text)

            away, home, sog_away, sog_home = self._parse_scores(text)

            # ── Intra-segment SOG tracking ────────────────────────────────
            if sog_home is not None and sog_away is not None:
                if run_sog_home is not None and run_sog_away is not None:
                    home_diff = sog_home - run_sog_home
                    away_diff = sog_away - run_sog_away
                    # Accept increments of 1-2 per frame (rarely 2 if a frame was
                    # skipped), reject decrements (bad OCR) and large jumps (noise).
                    # Each side must be EITHER unchanged OR a small increment —
                    # an "or" here (one side plausible, the other merely >= 0)
                    # let a wild jump on one side ride along with a legitimate
                    # +1 on the other (e.g. away +1 accepted the whole reading,
                    # including an unrelated home misread of +12), and that
                    # bad value then became the trusted baseline for everything
                    # after it.
                    home_plausible = home_diff == 0 or 0 < home_diff <= 2
                    away_plausible = away_diff == 0 or 0 < away_diff <= 2
                    if home_plausible and away_plausible and (home_diff > 0 or away_diff > 0):
                        target_home = sog_home if 0 < home_diff <= 2 else None
                        target_away = sog_away if 0 < away_diff <= 2 else None
                        # A single "plausible +1" read can't distinguish a
                        # real shot from a transient misread blip that
                        # reverts a moment later (a HUD animation frame,
                        # digit bleed) — real SOG never decreases, so check
                        # the reading still holds ~1s on before trusting it.
                        if self._confirm_sog_persists(cap, int(idx), total, fps, target_home, target_away):
                            count = (home_diff if 0 < home_diff <= 2 else 0) + (away_diff if 0 < away_diff <= 2 else 0)
                            sog_change_count += count
                            # The sample grid only tells us the shot happened
                            # *somewhere* between the previous sample (old SOG
                            # total) and this one (new total) — with sparse
                            # sampling that gap can be several seconds, and using
                            # idx/fps directly puts the "shot time" well after the
                            # puck was actually shot. Scan the gap for the actual
                            # transition frame when we have a previous sample to
                            # search from.
                            if last_good_idx is not None:
                                shot_frame = self._refine_shot_frame(
                                    cap, last_good_idx, int(idx),
                                    # Only require the side that actually changed
                                    # to be readable — the away crop in particular
                                    # flickers to None on a lot of frames, and
                                    # requiring *both* sides readable made the
                                    # scan skip right past the real transition
                                    # (home already at its new value) waiting for
                                    # a frame where away also happened to OCR.
                                    target_home=target_home,
                                    target_away=target_away,
                                    fps=fps,
                                )
                            else:
                                shot_frame = int(idx)
                            shot_time = shot_frame / fps
                            shot_timestamps.extend([shot_time] * count)
                            logger.debug(
                                "Intra-segment SOG bump in %s @ frame %d (refined from %s..%d): %s/%s → %s/%s (+%d)",
                                video_path.name, shot_frame, last_good_idx, idx, run_sog_away, run_sog_home, sog_away, sog_home, count,
                            )
                            # Only now — a plausible, confirmed increment —
                            # accept the new reading as the baseline.
                            run_sog_home = sog_home
                            run_sog_away = sog_away
                            last_good_idx = int(idx)
                            best_sog_home = sog_home
                            best_sog_away = sog_away
                        else:
                            logger.debug(
                                "Discarding unconfirmed SOG bump in %s @ frame %d: %s/%s reverted — treating as a misread",
                                video_path.name, idx, sog_away, sog_home,
                            )
                    elif home_diff == 0 and away_diff == 0:
                        # Unchanged — a clean confirmation of the current
                        # baseline, not just noise to ignore.
                        last_good_idx = int(idx)
                        best_sog_home = sog_home
                        best_sog_away = sog_away
                    else:
                        # Implausible jump or decrement — almost certainly a
                        # misread (a goal-scorer overlay, a blurred replay
                        # frame, digit bleed from a neighbouring UI element).
                        # Previously this still got accepted as the new
                        # baseline ("always advance the running pointer"),
                        # which let one bad frame both (a) register a false
                        # shot via a transient blip that reverts a frame
                        # later, and (b) poison every comparison after it so
                        # a real subsequent shot could no longer be seen as
                        # an increment. Leave the baseline untouched instead
                        # and wait for a reading that's actually consistent
                        # with it.
                        logger.debug(
                            "Rejecting implausible SOG read in %s @ frame %d: %s/%s (baseline %s/%s)",
                            video_path.name, idx, sog_away, sog_home, run_sog_away, run_sog_home,
                        )
                else:
                    # First successful read in this segment — nothing to
                    # compare against yet, so it just seeds the baseline.
                    run_sog_home = sog_home
                    run_sog_away = sog_away
                    last_good_idx = int(idx)
                    best_sog_home = sog_home
                    best_sog_away = sog_away

            if home is not None and away is not None:
                if best_home is not None and best_away is not None:
                    if home == best_home and away == best_away:
                        pass  # unchanged — confirms the current reading
                    elif self._check_change(home, away, best_home, best_away) and self._confirm_sog_persists(
                        cap, int(idx), total, fps,
                        target_home=home if home != best_home else None,
                        target_away=away if away != best_away else None,
                        field="score",
                    ):
                        # A plausible, confirmed increment within the segment
                        # itself (rare — a goal partway through one segment).
                        best_home, best_away = home, away
                    else:
                        # Implausible jump — same "last read wins" trap as
                        # the SOG side had: a single misread (team-abbrev
                        # OCR noise, a goal-scorer overlay) could otherwise
                        # silently override a previously good score and
                        # trigger a false goal downstream. Leave it alone.
                        logger.debug(
                            "Rejecting implausible score read in %s @ frame %d: %s-%s (current best %s-%s)",
                            video_path.name, idx, away, home, best_away, best_home,
                        )
                else:
                    best_home, best_away = home, away

        cap.release()

        hud_visible = best_sog_home is not None and best_sog_away is not None
        sog_changed = sog_change_count > 0

        raw_text = " | ".join(t.replace("\n", " ").strip() for t in all_texts if t.strip())

        # Score change still uses segment-to-segment comparison (goals happen once
        # per segment at most, and we still want the HUD-visibility gate).
        score_changed = (
            self._last_hud_visible
            and self._check_change(best_home, best_away, self._last_home, self._last_away)
        )

        if score_changed:
            logger.info(
                "Score change in %s: %s→%s / %s→%s",
                video_path.name, self._last_away, best_away, self._last_home, best_home,
            )
        if sog_changed:
            logger.debug(
                "SOG change(s) in %s: %d shot(s)  final sog=%s/%s",
                video_path.name, sog_change_count, best_sog_away, best_sog_home,
            )

        # Update state
        if best_home is not None and best_away is not None:
            self._last_home = best_home
            self._last_away = best_away
        if best_sog_home is not None and best_sog_away is not None:
            self._last_sog_home = best_sog_home
            self._last_sog_away = best_sog_away
        self._last_hud_visible = hud_visible

        logger.debug(
            "%s: home=%s away=%s sog=%s/%s sog_changes=%d score_changed=%s | %s",
            video_path.name, best_home, best_away,
            best_sog_home, best_sog_away,
            sog_change_count, score_changed, raw_text[:80],
        )

        return {
            "home": best_home,
            "away": best_away,
            "sog_home": best_sog_home,
            "sog_away": best_sog_away,
            "score_changed": score_changed,
            "sog_changed": sog_changed,
            "sog_change_count":  sog_change_count,
            "shot_timestamps":   shot_timestamps,
            "hud_visible":       hud_visible,
            "raw_text":          raw_text,
        }

    def score_segments(
        self,
        segments: list[dict],
        sc_shot_floor: float = 0.20,
        sc_boosted_confidence: float = 0.70,
    ) -> list[dict]:
        """
        Run score + SOG detection on all segments in order.

        Score change is used as an *aid signal* only — it boosts the segment's
        score to help it rank higher, but never overrides the label.  The GOAL
        banner detector (Step 4b/4c in the pipeline) is the definitive source
        of the "goal" label.

        SOG change is used the same way for scoring_chance: a shot detected
        via the scoreboard's own SOG counter is independent corroborating
        evidence, physical not statistical, for a segment the classifier
        already leaned toward calling scoring_chance. sc_shot_floor is
        deliberately low (0.20, not build_reel's 0.45 reel-inclusion gate) —
        a real shot is strong enough evidence on its own that we don't need
        the classifier to have been very confident first, only that
        scoring_chance was its best guess and not pure noise. Once that low
        bar is cleared, confidence jumps to sc_boosted_confidence outright
        rather than nudging just over the reel-inclusion gate — nudging a
        real, corroborated chance to 0.50 when the gate is 0.45 barely
        promotes it at all. Still never sets the label itself if the
        classifier called it something else; the ML prediction stays the
        ground truth for *what* a segment is, this only adjusts confidence
        in *that already-predicted* label.

        Modifies segments in-place. Returns the updated list.
        """
        hud_visible_count = 0
        sog_shots_total = 0
        score_changes_total = 0
        sc_boosted_total = 0

        for seg in segments:
            # Use fast mode (6 frames) for segments already confirmed as
            # non-highlights — we only need enough to keep SOG state current.
            interesting = seg.get("banner_detected") or seg.get("label") not in ("other", None)
            result = self.detect_in_segment(seg["path"], fast=not interesting)
            seg["score_home"] = result["home"]
            seg["score_away"] = result["away"]
            seg["sog_home"] = result["sog_home"]
            seg["sog_away"] = result["sog_away"]
            seg["score_changed"] = result["score_changed"]
            seg["sog_changed"] = result["sog_changed"]

            if result["hud_visible"]:
                hud_visible_count += 1

            if result["score_changed"]:
                score_changes_total += 1
                # Boost score/confidence as a secondary signal — do NOT set
                # label="goal" here. Banner detection is the ground truth.
                logger.info(
                    "  Score change: %s (label=%s)",
                    Path(seg["path"]).name, seg.get("label"),
                )
                seg["confidence"] = max(seg.get("confidence", 0.0), 0.85)
                seg["score"] = seg.get("score", 0.0) + 0.8

            elif result["sog_changed"]:
                n = result.get("sog_change_count", 1)
                sog_shots_total += n
                logger.info(
                    "  SOG shot x%d: %s  (sog=%s/%s)",
                    n, Path(seg["path"]).name,
                    result.get("sog_away"), result.get("sog_home"),
                )
                if seg.get("label") == "scoring_chance" and seg.get("confidence", 0.0) >= sc_shot_floor:
                    old_conf = seg.get("confidence", 0.0)
                    boosted = max(old_conf, sc_boosted_confidence)
                    if boosted > old_conf:
                        seg["confidence"] = boosted
                        seg["score"] = seg.get("score", 0.0) + 0.3
                        sc_boosted_total += 1
                        logger.info(
                            "    Shot corroborates scoring_chance → confidence %.0f%% → %.0f%%: %s",
                            old_conf * 100, boosted * 100, Path(seg["path"]).name,
                        )

        logger.info(
            "Score detector: %d/%d segments with HUD visible, %d shot(s), %d score change(s), "
            "%d scoring_chance boosted by shot corroboration",
            hud_visible_count, len(segments), sog_shots_total, score_changes_total, sc_boosted_total,
        )
        return segments

    def reset(self) -> None:
        """Reset score/SOG state — call between different match files."""
        self._last_home = None
        self._last_away = None
        self._last_sog_home = None
        self._last_sog_away = None
        self._last_hud_visible = False

    def scan_video(
        self,
        video_path: str | Path,
        sample_interval_s: float = 1.0,
    ) -> list[dict]:
        """
        Scan a single continuous video — a whole raw/normalised match file —
        frame-by-frame at *sample_interval_s* spacing for SOG and score
        changes, with no scene-splitting pass required first. Everything
        lives on one timeline here, unlike detect_in_segment() which compares
        across separate segment files, so there's no "previous segment"
        concept and no stitching needed to produce a clip window — just a
        timestamp to cut from the source with ffmpeg.

        Returns events in chronological order:
            {"type": "shot", "time": float, "sog_away": int, "sog_home": int}
            {"type": "goal", "time": float, "away": int, "home": int}

        Resets the detector's state first (a fresh scan, not a continuation
        of a previous detect_in_segment() run).

        A reading is only accepted as the new "known good" baseline when
        it's a plausible step from the current one (unchanged, or a 1-2
        increment) — a wild jump or a decrement is treated as a misread and
        left out, rather than corrupting the baseline for every comparison
        after it. This matters a lot more here than in detect_in_segment():
        scanning a full match hits far more OCR noise (replays, goal-scorer
        overlays replacing the scoreboard, camera cuts) than a handful of
        short segments ever did.
        """
        self.reset()
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(sample_interval_s * fps))

        events: list[dict] = []
        # Every counter is tracked per side, independently. Requiring both
        # sides to OCR on the same frame (the old behaviour) throws away
        # nearly every frame whenever one crop is flaky — on one match the
        # away SOG digit was unreadable on most frames while the home digit
        # read cleanly the whole game, and the scan found 1 shot in 23 min.
        # kind -> side -> accepted baseline / frame it was last confirmed at /
        # a candidate first reading waiting for a second agreeing read.
        base = {"score": {"home": None, "away": None}, "sog": {"home": None, "away": None}}
        last_idx = {"score": {"home": None, "away": None}, "sog": {"home": None, "away": None}}
        pending = {"score": {"home": None, "away": None}, "sog": {"home": None, "away": None}}
        max_inc = {"score": 3, "sog": 2}

        for f in range(0, max(total_frames, 1), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                continue
            crop = self._crop_roi(frame)
            text = self._ocr_crop(crop)
            away, home, sog_away, sog_home = self._parse_scores(text)
            readings = {
                "score": {"home": home, "away": away},
                "sog": {"home": sog_home, "away": sog_away},
            }

            for kind in ("score", "sog"):
                for side, val in readings[kind].items():
                    if val is None:
                        continue
                    cur = base[kind][side]
                    if cur is None:
                        # A first reading only becomes the baseline once a
                        # second reading agrees — one garbage frame (e.g. "2"
                        # misread as "12") must not be able to set it.
                        if pending[kind][side] == val:
                            base[kind][side] = val
                            last_idx[kind][side] = f
                            pending[kind][side] = None
                        else:
                            pending[kind][side] = val
                        continue

                    diff = val - cur
                    if diff == 0:
                        last_idx[kind][side] = f
                        continue
                    if not 0 < diff <= max_inc[kind]:
                        # Decrement or wild jump: a misread. Leave the
                        # baseline alone rather than let it poison every
                        # comparison after it.
                        continue

                    target_home = val if side == "home" else None
                    target_away = val if side == "away" else None
                    # Real counters never decrease, so a plausible increment
                    # that reverts a moment later was a misread blip.
                    if not self._confirm_sog_persists(
                        cap, f, total_frames, fps, target_home, target_away, field=kind,
                    ):
                        continue

                    other = base[kind]["away" if side == "home" else "home"]
                    pair = {side: val, ("away" if side == "home" else "home"): other}
                    if kind == "sog":
                        lo = last_idx[kind][side]
                        shot_frame = (
                            self._refine_shot_frame(
                                cap, lo, f,
                                target_home=target_home, target_away=target_away, fps=fps,
                            )
                            if lo is not None else f
                        )
                        for _ in range(diff):
                            events.append({
                                "type": "shot", "time": shot_frame / fps,
                                "sog_away": pair["away"], "sog_home": pair["home"],
                            })
                    else:
                        events.append({
                            "type": "goal", "time": f / fps,
                            "away": pair["away"], "home": pair["home"],
                        })
                    base[kind][side] = val
                    last_idx[kind][side] = f

        cap.release()
        return events

    def debug_frame(self, video_path: str | Path, frame_idx: int = 0, save_path: str | Path = "/tmp/score_roi_debug.png") -> None:
        """
        Save the scoreboard ROI crop to disk for visual inspection.
        Useful for tuning ROI constants.
        """
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            logger.error("Could not read frame %d from %s", frame_idx, video_path)
            return
        crop = self._crop_roi(frame)
        upscaled = cv2.resize(crop, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(save_path), upscaled)
        logger.info("Debug ROI saved to %s", save_path)
        text = self._ocr_crop(crop)
        print(f"OCR result: {repr(text)}")
        away, home, sog_away, sog_home = self._parse_scores(text)
        print(f"Parsed: home={home} away={away}  SOG: home={sog_home} away={sog_away}")

    def _refine_shot_frame(
        self,
        cap: cv2.VideoCapture,
        lo_idx: int,
        hi_idx: int,
        fps: float,
        target_home: Optional[int] = None,
        target_away: Optional[int] = None,
        step_s: float = 0.5,
    ) -> int:
        """
        Scan the frame range (lo_idx, hi_idx] for the earliest frame where
        the side(s) that changed reach their new SOG value — *target_home*
        and/or *target_away* (whichever is not None; pass only the side that
        actually incremented). The coarse sample grid only brackets the shot
        between two samples — this narrows that down to the actual frame the
        SOG counter ticked over, which can otherwise be several seconds off
        on a sparse grid (shot_time landing after the puck was already shot).

        Only the changed side is required to be readable on a given probe —
        requiring *both* sog_home and sog_away to OCR successfully made the
        scan skip right past the real transition (the changed side already
        at its new value) while waiting for the other, unrelated side to
        also happen to OCR cleanly on the same frame.

        Uses a fixed-step linear scan rather than a binary search: the frames
        right around a shot are exactly the ones most likely to be unreadable
        (motion blur, camera cut to a replay/reaction shot), and a binary
        search that repeatedly lands on those frames can fail to converge at
        all. A linear scan just skips a bad probe and keeps going, trading a
        few extra OCR calls for much better robustness. step_s=0.5 gives
        half-second resolution, which is plenty for a multi-second clip
        window.

        Falls back to hi_idx (the original, conservative estimate) if no
        probe in the range reaches the target(s).
        """
        step = max(1, int(step_s * fps))
        for probe in range(lo_idx + 1, hi_idx, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, probe)
            ret, frame = cap.read()
            if not ret:
                continue
            crop = self._crop_roi(frame)
            text = self._ocr_crop(crop)
            _, _, sog_away, sog_home = self._parse_scores(text)
            # Bounded match, not an open-ended ">=": an unrelated OCR glitch
            # misreading "2" as "12" (digit bleed from the clock, a stray
            # pixel) would satisfy ">= 3" immediately and falsely "confirm"
            # the target on the very next probed frame. target_(home|away)
            # is already the exact value seen at the coarse sample, so only
            # that value (or +1, in case another shot lands in the same gap)
            # should count.
            home_ok = target_home is None or (sog_home is not None and target_home <= sog_home <= target_home + 1)
            away_ok = target_away is None or (sog_away is not None and target_away <= sog_away <= target_away + 1)
            if home_ok and away_ok:
                return probe
        return hi_idx

    def _confirm_sog_persists(
        self,
        cap: cv2.VideoCapture,
        frame_idx: int,
        total_frames: int,
        fps: float,
        target_home: Optional[int],
        target_away: Optional[int],
        confirm_gap_s: float = 1.0,
        field: str = "sog",
    ) -> bool:
        """
        Check that a SOG (or, with field="score", a goal) reading at
        *frame_idx* still holds ~confirm_gap_s later. Real SOG/score counts
        never decrease, so a value that reverts a second later is
        conclusively a misread (a HUD animation blip, a goal-scorer overlay,
        digit bleed) rather than a real event — this caught a case where
        "home SOG 6→7" held for exactly 2 coarse-sample probes (~0.5s) and
        then reverted to 6 for the rest of the segment. A single "plausible
        +1 increment" check can't tell those apart from a real shot; this
        can.

        Returns True (accept) when there's no room left to check, or the
        probe frame is unreadable — we don't want to penalize an event that
        happens right at the end of a segment just because we can't look
        past it.
        """
        probe = frame_idx + int(confirm_gap_s * fps)
        if probe >= total_frames:
            return True
        cap.set(cv2.CAP_PROP_POS_FRAMES, probe)
        ret, frame = cap.read()
        if not ret:
            return True
        crop = self._crop_roi(frame)
        text = self._ocr_crop(crop)
        away, home, sog_away, sog_home = self._parse_scores(text)
        if field == "score":
            sog_home, sog_away = home, away
        # Bounded match, not open-ended ">=" — see _refine_shot_frame for why:
        # an unrelated garbage read (e.g. "12") would otherwise trivially
        # satisfy ">= 3" and "confirm" an event that never actually happened.
        home_ok = target_home is None or sog_home is None or target_home <= sog_home <= target_home + 1
        away_ok = target_away is None or sog_away is None or target_away <= sog_away <= target_away + 1
        return home_ok and away_ok

    # ── Private helpers ───────────────────────────────────────────────────────

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = int(ROI_X * w)
        y1 = int(ROI_Y * h)
        x2 = int((ROI_X + ROI_W) * w)
        y2 = int((ROI_Y + ROI_H) * h)
        return frame[y1:y2, x1:x2]

    def _ocr_crop(self, crop: np.ndarray) -> str:
        """
        Run focused OCR on three separate sub-regions of the HUD ROI.

        HUD layout (% of ROI width):
          0-18%   : logo (ignored)
          15-65%  : team abbrev + score  (top=away, bottom=home)
          65-78%  : SOG count            (top digit=away, bottom digit=home, label in middle)
          80-100% : game clock + period  (top=clock, bottom=period)

        Returns a two-line string:  "<away_row>\n<home_row>"
        where each row is:  "SCORE SOG"  e.g. "0 6"
        """
        import pytesseract

        up = cv2.resize(crop, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        # Otsu threshold: automatically finds the best cut between background and
        # text pixels, so it works for both the normal opaque HUD and the semi-
        # transparent HUD (where background brightens to ~103 from ~60-80).
        # We clamp to a minimum of 90 to avoid noise on fully dark frames.
        otsu_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu_val < 90:
            # Frame is too dark / HUD likely absent — re-threshold at 90 so SOG
            # parse returns None (hud_visible=False) rather than returning garbage.
            _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

        h, w = binary.shape
        mid = h // 2

        OCR_TIMEOUT = 5  # seconds per tesseract call; prevents hangs on bad frames

        def _ocr(crop: np.ndarray, config: str) -> str:
            """Run tesseract with a timeout; return '' on empty crop or timeout."""
            if crop is None or crop.size == 0 or 0 in crop.shape:
                return ""
            try:
                return pytesseract.image_to_string(crop, config=config, timeout=OCR_TIMEOUT).strip()
            except RuntimeError:
                logger.debug("Tesseract timed out (>%ds) on a crop — skipping frame", OCR_TIMEOUT)
                return ""
            except Exception as exc:
                logger.debug("Tesseract error on crop: %s", exc)
                return ""

        # ── Score sub-crop (x: 15-65%) ─────────────────────────────────────
        sx1, sx2 = int(0.15 * w), int(0.65 * w)
        score_top = binary[:mid,  sx1:sx2]
        score_bot = binary[mid:,  sx1:sx2]
        cfg_line  = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        pool = _ocr_pool()
        away_score_f = pool.submit(_ocr, score_top, cfg_line)
        home_score_f = pool.submit(_ocr, score_bot, cfg_line)

        # ── SOG sub-crop (x: 65-78%) — digit rows only, skip middle label ──
        gx1, gx2 = int(0.65 * w), int(0.78 * w)
        # Away SOG digit: tight band avoiding the row above and the "SOG" label.
        # Home SOG digit: tight band avoiding the "SOG" label and the HUD's
        # bottom edge (letting it run to 100% picks up bright background below
        # the panel, which confuses single-char OCR).
        sog_away_crop = binary[int(0.08 * h):int(0.35 * h), gx1:gx2]
        sog_home_crop = binary[int(0.58 * h):int(0.90 * h), gx1:gx2]
        cfg_char = "--psm 10 -c tessedit_char_whitelist=0123456789"
        away_sog_f = pool.submit(_ocr, sog_away_crop, cfg_char)
        home_sog_f = pool.submit(_ocr, sog_home_crop, cfg_char)

        # ── Clock sub-crop (x: 80-100%, top half) ──────────────────────────
        cx1 = int(0.80 * w)
        clock_crop = binary[:mid, cx1:]
        cfg_clock = "--psm 7 -c tessedit_char_whitelist=0123456789:"
        clock_f = pool.submit(_ocr, clock_crop, cfg_clock)

        away_score_text = away_score_f.result()
        home_score_text = home_score_f.result()
        away_sog_text = away_sog_f.result()
        home_sog_text = home_sog_f.result()
        clock_text = clock_f.result()

        # Pack into two lines that _parse_scores can split on
        # Format: "SCORE_TEXT|SOG_TEXT\nSCORE_TEXT|SOG_TEXT"
        return f"{away_score_text}|{away_sog_text}|{clock_text}\n{home_score_text}|{home_sog_text}"

    # Goal-scorer / assist overlay banners replace the scoreboard HUD for a
    # few seconds after a goal — e.g. "7BYFIELD||1:24\nQOAL2|7" and
    # "7NECASPOWER||18\nASSISTS|" are real OCR output from one. Unlike a
    # transient single-frame misread, a banner's text is *consistent* for as
    # long as it's on screen, so it sails straight through the "does this
    # reading persist a second later" confirmation check — that check only
    # catches noise that reverts, not a sustained read of the wrong overlay
    # entirely. This needs to be caught at parse time instead.
    _BANNER_KEYWORDS = ("GOAL", "QOAL", "GOAI", "ASSIST", "SCORED", "SCORER")

    @classmethod
    def _looks_like_banner(cls, text: str) -> bool:
        """True if *text* looks like a goal/assist overlay rather than the scoreboard."""
        upper = text.upper()
        if any(kw in upper for kw in cls._BANNER_KEYWORDS):
            return True
        # Team abbreviations are always 2-3 letters (SJS, CGY, EDM, ...). A
        # run of 5+ consecutive uppercase letters where one is expected is
        # almost certainly a player surname (BYFIELD, NECASPOWER, ...).
        if re.search(r'[A-Z]{5,}', upper):
            return True
        return False

    def _parse_scores(self, text: str) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        Parse home/away scores and SOG counts from the packed OCR string produced by _ocr_crop.

        Format: "<away_score_text>|<away_sog>|<clock>\n<home_score_text>|<home_sog>"

        Returns all-None if the text looks like a goal/assist banner rather
        than the real scoreboard — treating the HUD as "not visible" for that
        frame rather than parsing garbage digits out of overlay text.
        """
        if self._looks_like_banner(text):
            return None, None, None, None

        lines = text.splitlines()
        away_line = lines[0] if len(lines) > 0 else ""
        home_line = lines[1] if len(lines) > 1 else ""

        def parse_score(s: str) -> Optional[int]:
            """Find the rightmost digit after the team abbrev — that's the score.
            Also handles OCR merging O→0 at the end (e.g. 'EDMO' → score=0)."""
            s = re.sub(r'\d{1,2}:\d{2}', '', s)  # strip clock
            # 'EDMO' or 'SJSO' etc. — trailing O after 2-3 uppercase letters is really digit 0
            s = re.sub(r'([A-Z]{2,3})O\b', r'\1 0', s)
            # Standalone 'O' (score digit read as letter by OCR) → 0
            s = re.sub(r'\bO\b', '0', s)
            m = re.findall(r'(?<![:\d])(\d)(?!\d)', s)
            return int(m[-1]) if m else None

        def parse_sog(s: str) -> Optional[int]:
            """The SOG crop should be just 1-2 digits."""
            s = s.strip()
            m = re.fullmatch(r'(\d{1,2})', s)
            return int(m.group(1)) if m else None

        away_parts = away_line.split("|")
        home_parts = home_line.split("|")

        away_score = parse_score(away_parts[0]) if len(away_parts) > 0 else None
        away_sog   = parse_sog(away_parts[1])   if len(away_parts) > 1 else None
        home_score = parse_score(home_parts[0]) if len(home_parts) > 0 else None
        home_sog   = parse_sog(home_parts[1])   if len(home_parts) > 1 else None

        return away_score, home_score, away_sog, home_sog

    def _check_change(
        self,
        new_home: Optional[int],
        new_away: Optional[int],
        last_home: Optional[int],
        last_away: Optional[int],
        strict: bool = True,
    ) -> bool:
        """
        Return True if either counter incremented vs last known values.

        Args:
            strict: If True (score), require exactly one counter to go up by 1
                    and the other to stay the same — guards against double-goals.
                    If False (SOG), either counter going up by ≥1 counts.
        """
        if new_home is None or new_away is None:
            return False
        if last_home is None or last_away is None:
            return False

        home_diff = new_home - last_home
        away_diff = new_away - last_away

        if strict:
            # At least one team scored; allow increments of 1-3 to handle
            # segments where OCR failed and multiple goals were missed.
            # Reject decrements (bad OCR) and no-change. Each side must be
            # EITHER unchanged OR a plausible 1-3 increment — not just ">= 0"
            # on the side that didn't need to match, which let a wild jump on
            # one side (a misread) ride along with a legitimate goal on the
            # other and get accepted as part of the "new" score. See the
            # identical fix in detect_in_segment/scan_video's SOG tracking.
            home_ok = home_diff == 0 or home_diff in (1, 2, 3)
            away_ok = away_diff == 0 or away_diff in (1, 2, 3)
            return home_ok and away_ok and (home_diff != 0 or away_diff != 0)
        else:
            # Either team registered a shot (SOG can go up by 1+, ignore decreases)
            return home_diff > 0 or away_diff > 0
