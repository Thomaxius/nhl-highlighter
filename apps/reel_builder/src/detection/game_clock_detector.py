"""
detection/game_clock_detector.py

Detects end-of-period and end-of-game moments in NHL 25 footage by reading
the game clock HUD via template matching + OCR.

Clock behaviour:
  - "0:00"  → game ends  (3rd period / OT final buzzer)
  - "0:01"  → period ends (1st or 2nd period final buzzer)

Approach
--------
1. Template match (fast pre-filter): compare each sampled frame against
   ``configs/game_clock.png`` — a reference crop of the clock region at ~0:00.
   Frames below MIN_TEMPLATE_CONF are skipped cheaply.
2. For frames that pass, OCR the tight clock crop with pytesseract and parse
   the displayed time.  "0:00" → game_end; "0:01" → period_end.

Usage
-----
    detector = GameClockDetector()
    events   = detector.scan_video("game_norm.mp4")
    # → [{"time_s": 1762.0, "event": "game_end"},
    #    {"time_s": 601.0,  "event": "period_end"}, ...]
"""

from __future__ import annotations

import re
from pathlib import Path
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ── Clock ROI (fractions of frame size) ──────────────────────────────────────
# Top-LEFT — same HUD strip as the score/goals/shots display in NHL 25.
# Matches the ScoreDetector ROI dimensions.
ROI_X = 0.025  # left edge  (fraction of width)
ROI_Y = 0.035  # top edge   (fraction of height)
ROI_W = 0.215  # width      (fraction of frame width)
ROI_H = 0.074  # height     (fraction of frame height) — tall enough for clock + period label (~80px @ 1080p)

# Template match threshold below which we skip OCR entirely (speed gate).
MIN_TEMPLATE_CONF        = 0.45  # gate to allow OCR (applies to all events)
MIN_GAME_END_CONF        = 0.70  # stricter gate specifically for game_end events


class GameClockDetector:
    """
    Detects end-of-period / end-of-game moments by reading the game clock.

    Args:
        template_path:  Path to a reference crop of the clock at ~0:00.
                        Defaults to ``configs/game_clock.png``.
        template_threshold: Template-match gate; frames below this are skipped
                        without running OCR. Default 0.40 (loose — just to
                        filter out black screens and non-HUD frames).
        scale_factors:  Template scales to try.  Default covers ±20 %.
    """

    def __init__(
        self,
        template_path: str | Path = "apps/reel_builder/configs/game_end_template.png",
        ot_template_path: str | Path = "apps/reel_builder/configs/game_clock_ot.png",
        template_threshold: float = MIN_TEMPLATE_CONF,
        scale_factors: list[float] | None = None,
    ) -> None:
        self.template_threshold = template_threshold
        self.scale_factors = scale_factors or [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

        template_path = Path(template_path)
        if not template_path.exists():
            raise FileNotFoundError(
                f"Game clock template not found: {template_path}\n"
                "Save a screenshot of the clock region at ~0:00 to that path."
            )
        self.template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if self.template_gray is None:
            raise ValueError(f"Could not read template: {template_path}")

        # All templates to check — regular + optional OT variant
        self._all_templates: list[np.ndarray] = [self.template_gray]
        ot_path = Path(ot_template_path)
        if ot_path.exists():
            ot_t = cv2.imread(str(ot_path), cv2.IMREAD_GRAYSCALE)
            if ot_t is not None:
                self._all_templates.append(ot_t)
                logger.info("OT clock template loaded (%dx%d)", ot_t.shape[1], ot_t.shape[0])

        logger.info(
            "GameClockDetector ready — template %dx%d, gate threshold %.2f",
            self.template_gray.shape[1], self.template_gray.shape[0],
            template_threshold,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def scan_video(
        self,
        video_path: str | Path,
        interval_s: float = 2.0,
        merge_gap_s: float = 15.0,
    ) -> list[dict]:
        """
        Scan a full normalised video for end-of-period / end-of-game events.

        Args:
            video_path:   Path to the normalised .mp4 file.
            interval_s:   Seconds between sampled frames. Default 2.0  — the
                          clock changes slowly so this is more than sufficient.
            merge_gap_s:  Consecutive detections within this window (seconds)
                          are collapsed to the earliest timestamp.

        Returns:
            List of event dicts, sorted by time::

                [{"time_s": 1762.0, "event": "game_end"},
                 {"time_s":  601.0, "event": "period_end"}]
        """
        try:
            import pytesseract  # noqa: F401 — checked here so error is clear
        except ImportError:
            raise ImportError(
                "pytesseract is required for GameClockDetector.scan_video(). "
                "Install it: pip install pytesseract"
            )

        # On Windows, Tesseract is often not on PATH even if installed.
        # Auto-detect the default UB-Mannheim install location.
        import sys, os
        if sys.platform == "win32":
            import pytesseract as _pt
            if not _pt.get_tesseract_version:
                pass  # already configured
            default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default_win_path) and not os.path.isabs(_pt.pytesseract.tesseract_cmd):
                _pt.pytesseract.tesseract_cmd = default_win_path

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        step    = max(1, int(fps * interval_s))
        indices = list(range(0, n_total, step))
        logger.info(
            "GameClockDetector: scanning %s — %d samples at %.0fs intervals",
            video_path.name, len(indices), interval_s,
        )

        raw_events: list[dict] = []

        # Terminal events: only regulation_end (3RD buzzer) triggers the
        # reverse-scan stop.  OT period buzzers (ot_period_end) are found in
        # the forward pass and logged but don't anchor the game_end window.
        _TERMINAL = {"regulation_end"}

        # ── Pass 1: scan BACKWARDS for regulation_end, stop at first hit ─────
        # Finds the last 0:00 in the video (could be 3RD or OT period).
        for i in reversed(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            gate_conf = self._template_conf(frame)
            if gate_conf < MIN_GAME_END_CONF:
                continue

            clock_str = self._ocr_clock(frame)
            event     = self._classify_clock(clock_str)
            if event in _TERMINAL:
                t = i / fps
                logger.info(
                    "  Clock event (reverse scan): t=%.1fs  OCR=%r  event=%s  gate_conf=%.2f",
                    t, clock_str, event, gate_conf,
                )
                raw_events.append({"time_s": t, "event": event})
                break  # earliest confirmed hit from the end — done

        # ── Pass 2: scan FORWARD for all other events (game_start, period_end) ──
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            gate_conf = self._template_conf(frame)
            if gate_conf < self.template_threshold:
                continue

            clock_str = self._ocr_clock(frame)
            event     = self._classify_clock(clock_str)
            if event and event not in _TERMINAL:
                t = i / fps
                logger.info(
                    "  Clock event: t=%.1fs  OCR=%r  event=%s  gate_conf=%.2f",
                    t, clock_str, event, gate_conf,
                )
                raw_events.append({"time_s": t, "event": event})

        cap.release()

        # Merge consecutive detections of the same event type
        raw_events.sort(key=lambda e: e["time_s"])
        merged = self._merge_events(raw_events, merge_gap_s)
        logger.info(
            "GameClockDetector: %d raw hit(s) → %d event(s): %s",
            len(raw_events), len(merged),
            [(f"{e['event']}@{e['time_s']:.0f}s") for e in merged],
        )
        return merged

    def detect_in_segment(self, video_path: str | Path, sample_frames: int = 10) -> dict:
        """
        Scan a video segment for near-zero clock state (template match only).
        Kept for backward compatibility — prefer scan_video for full-game use.

        Returns:
            ``{"detected": bool, "max_confidence": float, "best_frame": int}``
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

        indices = np.linspace(0, total - 1, sample_frames, dtype=int)
        max_conf, best_frame = 0.0, 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            c = self._template_conf(frame)
            if c > max_conf:
                max_conf, best_frame = c, int(idx)

        cap.release()
        detected = max_conf >= self.template_threshold
        if detected:
            logger.info(
                "Game-clock match in %s (conf=%.2f, t=%.1fs)",
                video_path.name, max_conf, best_frame / fps,
            )
        return {"detected": detected, "max_confidence": float(max_conf), "best_frame": best_frame}

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _search_region(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        Return a grayscale search region plus the (x0, y0) offset within frame.

        Uses the top-left 30% × 15% of the frame — generously covers the entire
        HUD clock panel regardless of minor layout shifts, and avoids the right
        half of the screen where unrelated elements could confuse the match.
        """
        h, w = frame.shape[:2]
        x0, y0 = 0, 0
        x1 = int(w * 0.30)
        y1 = int(h * 0.15)
        region = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return gray, x0, y0

    def _template_match(self, frame: np.ndarray, template: np.ndarray | None = None) -> tuple[float, int, int, int, int]:
        """
        Return ``(conf, abs_x, abs_y, matched_w, matched_h)`` where
        ``abs_x / abs_y`` are coordinates in the *full frame*.
        """
        tmpl = template if template is not None else self.template_gray
        gray, ox, oy = self._search_region(frame)
        best_conf, best_loc, best_w, best_h = 0.0, (0, 0), 0, 0
        for scale in self.scale_factors:
            th, tw = tmpl.shape[:2]
            new_h = max(1, int(th * scale))
            new_w = max(1, int(tw * scale))
            if new_h > gray.shape[0] or new_w > gray.shape[1]:
                continue
            resized = cv2.resize(tmpl, (new_w, new_h))
            res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(res)
            if v > best_conf:
                best_conf, best_loc, best_w, best_h = v, loc, new_w, new_h
        # Convert match location from search-region coords to full-frame coords
        abs_x = ox + best_loc[0]
        abs_y = oy + best_loc[1]
        return best_conf, abs_x, abs_y, best_w, best_h

    def _best_template_match(self, frame: np.ndarray) -> tuple[float, int, int, int, int]:
        """Try all loaded templates and return the result with highest confidence."""
        best = self._template_match(frame, self._all_templates[0])
        for tmpl in self._all_templates[1:]:
            result = self._template_match(frame, tmpl)
            if result[0] > best[0]:
                best = result
        return best

    def _template_conf(self, frame: np.ndarray) -> float:
        """Return the best template-match confidence across all loaded templates."""
        return self._best_template_match(frame)[0]

    def _ocr_clock(self, frame: np.ndarray) -> str:
        """
        OCR the clock region and return the raw string (e.g. ``"0:00\n3RD"``).

        Uses the template match location to isolate just the clock band
        (time digits + period label strip below), avoiding score/shots
        digits in the surrounding HUD that confuse Tesseract.
        """
        import pytesseract
        h, w = frame.shape[:2]
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Locate the clock in the full frame via template matching
        _, ax, ay, mw, mh = self._best_template_match(frame)

        # Add a strip below the matched region for the period label
        # ("3RD"/"1ST"/"2ND"), roughly 40% of the matched height.
        label_strip = max(8, int(mh * 0.40))
        cy0 = max(0, ay)
        cy1 = min(h, ay + mh + label_strip)
        cx0 = max(0, ax)
        cx1 = min(w, ax + mw)
        clock_band = gray_full[cy0:cy1, cx0:cx1]

        # Upscale 3× and Otsu-threshold for better OCR on the dark HUD
        big = cv2.resize(clock_band, (clock_band.shape[1] * 3, clock_band.shape[0] * 3),
                         interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # PSM 6 = assume uniform block of text (clock digits + period label)
        cfg  = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789:.ORDSTHN"
        text = pytesseract.image_to_string(bw, config=cfg).strip()
        return text

    @staticmethod
    def _classify_clock(text: str) -> str | None:
        """
        Classify a clock OCR string into an event type.

        Returns one of ``"game_start"``, ``"period_start"``, ``"period_end"``,
        ``"game_end"``, or ``None``.

        The OCR text contains two lines:
          - Line 1: clock value, e.g. ``"20:00"``, ``"0:01"``, ``":0.2"``
          - Line 2: period label, e.g. ``"3RD"``, ``"1ST"``, ``"2ND"``, ``"OT"``
        """
        upper = text.upper()
        digits = re.findall(r'\d+', text)

        # ── 1. Start of period: clock reads 20:00 ────────────────────────
        if len(digits) >= 2:
            try:
                if int(digits[0]) == 20 and int(digits[1]) == 0:
                    if '1ST' in upper:
                        return 'game_start'
                    if '2ND' in upper or '3RD' in upper or 'OT' in upper:
                        return 'period_start'
                    return 'game_start'  # conservative fallback (first period)
            except ValueError:
                pass

        # ── 2. Is the clock near zero? ────────────────────────────────────
        # Sub-second display: digit(s) + dot, e.g. "0.2", ":0.2"
        near_zero = False
        if '.' in text:
            pre_dot = re.findall(r'\d+', text.split('.')[0])
            if not pre_dot or int(pre_dot[-1]) == 0:
                near_zero = True
        else:
            if len(digits) >= 2:
                try:
                    near_zero = (int(digits[0]) == 0 and int(digits[1]) <= 5)
                except ValueError:
                    pass
            elif len(digits) == 1:
                try:
                    near_zero = (int(digits[0]) <= 5)
                except ValueError:
                    pass

        if not near_zero:
            return None

        # ── 3. Which period ends? ─────────────────────────────────────────
        if 'OT' in upper:
            return 'ot_period_end'  # OT period ran out — another OT or shootout follows
        if '3RD' in upper:
            return 'regulation_end'  # end of regulation; OT may follow
        if '1ST' in upper or '2ND' in upper:
            return 'period_end'

        # Period label not readable — ignore rather than assume game_end
        return None

    @staticmethod
    def _merge_events(events: list[dict], gap_s: float) -> list[dict]:
        """Collapse consecutive same-event detections within *gap_s* seconds."""
        merged: list[dict] = []
        for ev in events:
            if (merged
                    and merged[-1]["event"] == ev["event"]
                    and ev["time_s"] - merged[-1]["time_s"] <= gap_s):
                continue  # duplicate — keep the first (earliest) occurrence
            merged.append(ev)
        return merged
