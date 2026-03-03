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
import re
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ── Scoreboard ROI (fractions of 1920×1080) ───────────────────────────────────
# Covers the full scoreboard panel in the top-left corner.
# Tune these if the crop looks off — use ScoreDetector.debug_frame() to preview.
ROI_X = 0.02
ROI_Y = 0.035
ROI_W = 0.2176  # ~21.76% of width → 418px
ROI_H = 0.077   # ~7.7% of height → 83px


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
        cap = cv2.VideoCapture(str(video_path))
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

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            crop = self._crop_roi(frame)
            text = self._ocr_crop(crop)
            all_texts.append(text)

            home, away, sog_home, sog_away = self._parse_scores(text)

            # ── Intra-segment SOG tracking ────────────────────────────────
            if sog_home is not None and sog_away is not None:
                if run_sog_home is not None and run_sog_away is not None:
                    home_diff = sog_home - run_sog_home
                    away_diff = sog_away - run_sog_away
                    # Accept increments of 1-2 per frame (rarely 2 if a frame was
                    # skipped), reject decrements (bad OCR) and large jumps (noise).
                    if (0 < home_diff <= 2 or 0 < away_diff <= 2) and home_diff >= 0 and away_diff >= 0:
                        count = (home_diff if 0 < home_diff <= 2 else 0) + (away_diff if 0 < away_diff <= 2 else 0)
                        sog_change_count += count
                        logger.debug(
                            "Intra-segment SOG bump in %s @ frame %d: %s/%s → %s/%s (+%d)",
                            video_path.name, idx, run_sog_away, run_sog_home, sog_away, sog_home, count,
                        )
                # Always advance the running pointer when we get a valid read
                run_sog_home = sog_home
                run_sog_away = sog_away
                # Track best SOG independently of whether score parsed —
                # away score text is noisy and often returns None.
                best_sog_home = sog_home
                best_sog_away = sog_away

            if home is not None and away is not None:
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
            "sog_change_count": sog_change_count,
            "hud_visible": hud_visible,
            "raw_text": raw_text,
        }

    def score_segments(self, segments: list[dict]) -> list[dict]:
        """
        Run score + SOG detection on all segments in order.

        Score change is used as an *aid signal* only — it boosts the segment's
        score to help it rank higher, but never overrides the label.  The GOAL
        banner detector (Step 4b/4c in the pipeline) is the definitive source
        of the "goal" label.

        SOG change is tagged for training-data collection use.

        Modifies segments in-place. Returns the updated list.
        """
        for seg in segments:
            # Use fast mode (3 frames) for segments already confirmed as
            # non-highlights — we only need enough to keep SOG state current.
            interesting = seg.get("banner_detected") or seg.get("label") not in ("other", None)
            result = self.detect_in_segment(seg["path"], fast=not interesting)
            seg["score_home"] = result["home"]
            seg["score_away"] = result["away"]
            seg["sog_home"] = result["sog_home"]
            seg["sog_away"] = result["sog_away"]
            seg["score_changed"] = result["score_changed"]
            seg["sog_changed"] = result["sog_changed"]

            if result["score_changed"]:
                # Boost score/confidence as a secondary signal — do NOT set
                # label="goal" here. Banner detection is the ground truth.
                logger.info(
                    "Score change aid signal: %s (current label=%s)",
                    Path(seg["path"]).name, seg.get("label"),
                )
                seg["confidence"] = max(seg.get("confidence", 0.0), 0.85)
                seg["score"] = seg.get("score", 0.0) + 0.8

            elif result["sog_changed"]:
                logger.debug(
                    "SOG change (shot on net): %s", Path(seg["path"]).name
                )

        return segments

    def reset(self) -> None:
        """Reset score/SOG state — call between different match files."""
        self._last_home = None
        self._last_away = None
        self._last_sog_home = None
        self._last_sog_away = None
        self._last_hud_visible = False

    def debug_frame(self, video_path: str | Path, frame_idx: int = 0, save_path: str | Path = "/tmp/score_roi_debug.png") -> None:
        """
        Save the scoreboard ROI crop to disk for visual inspection.
        Useful for tuning ROI constants.
        """
        cap = cv2.VideoCapture(str(video_path))
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
        home, away, sog_home, sog_away = self._parse_scores(text)
        print(f"Parsed: home={home} away={away}  SOG: home={sog_home} away={sog_away}")

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

        # ── Score sub-crop (x: 15-65%) ─────────────────────────────────────
        sx1, sx2 = int(0.15 * w), int(0.65 * w)
        score_top = binary[:mid,  sx1:sx2]
        score_bot = binary[mid:,  sx1:sx2]
        cfg_line  = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        away_score_text = pytesseract.image_to_string(score_top, config=cfg_line).strip()
        home_score_text = pytesseract.image_to_string(score_bot, config=cfg_line).strip()

        # ── SOG sub-crop (x: 65-78%) — digit rows only, skip middle label ──
        gx1, gx2 = int(0.65 * w), int(0.78 * w)
        # Away SOG digit: top 40% of ROI height; Home SOG digit: bottom 40%
        sog_away_crop = binary[:int(0.40 * h), gx1:gx2]
        sog_home_crop = binary[int(0.60 * h):, gx1:gx2]
        cfg_char = "--psm 10 -c tessedit_char_whitelist=0123456789"
        away_sog_text = pytesseract.image_to_string(sog_away_crop, config=cfg_char).strip()
        home_sog_text = pytesseract.image_to_string(sog_home_crop, config=cfg_char).strip()

        # ── Clock sub-crop (x: 80-100%, top half) ──────────────────────────
        cx1 = int(0.80 * w)
        clock_crop = binary[:mid, cx1:]
        cfg_clock = "--psm 7 -c tessedit_char_whitelist=0123456789:"
        clock_text = pytesseract.image_to_string(clock_crop, config=cfg_clock).strip()

        # Pack into two lines that _parse_scores can split on
        # Format: "SCORE_TEXT|SOG_TEXT\nSCORE_TEXT|SOG_TEXT"
        return f"{away_score_text}|{away_sog_text}|{clock_text}\n{home_score_text}|{home_sog_text}"

    def _parse_scores(self, text: str) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        Parse home/away scores and SOG counts from the packed OCR string produced by _ocr_crop.

        Format: "<away_score_text>|<away_sog>|<clock>\n<home_score_text>|<home_sog>"
        """
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
            # Reject decrements (bad OCR) and no-change.
            return (home_diff in (1, 2, 3) or away_diff in (1, 2, 3)) and home_diff >= 0 and away_diff >= 0
        else:
            # Either team registered a shot (SOG can go up by 1+, ignore decreases)
            return home_diff > 0 or away_diff > 0
