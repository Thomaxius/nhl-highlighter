"""
vs_screen_detector.py  —  Detects the pre-match VS screen in NHL 25 recordings.

The VS screen appears a few seconds before the opening puck-drop.  Detecting it
lets the pipeline include the team intro (VS → logos → puck drop → ~8 s of play)
as a natural opening segment for the highlight reel.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Match threshold: single scale, full-frame search in the centre band
MIN_CONF = 0.55
# Centre band to search: horizontal 30-70%, vertical 35-65% of frame
SEARCH_X = (0.30, 0.70)
SEARCH_Y = (0.35, 0.65)


class VsScreenDetector:
    """
    Template-match detector for the pre-match VS screen.

    Parameters
    ----------
    template_path:
        Path to ``configs/vs_screen_template.png`` — a crop of the VS screen
        centre region (produced by extracting t≈11s from a reference game).
    threshold:
        Minimum ``TM_CCOEFF_NORMED`` confidence to accept a match.
    """

    def __init__(
        self,
        template_path: str | Path = "configs/vs_screen_template.png",
        threshold: float = MIN_CONF,
    ) -> None:
        template_path = Path(template_path)
        if not template_path.exists():
            raise FileNotFoundError(
                f"VS screen template not found: {template_path}\n"
                "Run the extraction step first or check configs/."
            )
        self.template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError(f"Could not read template: {template_path}")
        self.threshold = threshold
        logger.debug(
            "VsScreenDetector ready — template %dx%d, threshold %.2f",
            self.template.shape[1], self.template.shape[0], threshold,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def match_frame(self, frame: np.ndarray) -> float:
        """Return template-match confidence for a single frame (0–1)."""
        return self._conf(frame)

    def scan_video(
        self,
        video_path: str | Path,
        interval_s: float = 0.5,
        search_window_s: float = 60.0,
        search_start_s: float = 0.0,
    ) -> float | None:
        """
        Scan *video_path* for the VS screen between *search_start_s* and
        *search_window_s* seconds.

        Returns the timestamp in seconds of the first frame that exceeds the
        threshold, or ``None`` if not found.

        Parameters
        ----------
        interval_s:
            How often (seconds) to sample frames.
        search_window_s:
            Upper bound of the scan window in seconds.
        search_start_s:
            Skip this many seconds before starting the scan.  Useful when a
            brief early VS-screen appearance should be ignored.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        max_t = min(search_window_s, total_frames / fps)

        step = max(1, int(fps * interval_s))
        t = max(0.0, search_start_s)
        result: float | None = None

        while t <= max_t:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            conf = self._conf(frame)
            if conf >= self.threshold:
                logger.info(
                    "VS screen detected at t=%.2fs  conf=%.3f  (%s)",
                    t, conf, Path(video_path).name,
                )
                result = t
                break

            t += interval_s

        cap.release()
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _search_region(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Return (gray_region, x_offset, y_offset) for the centre band."""
        h, w = frame.shape[:2]
        x0 = int(w * SEARCH_X[0]); x1 = int(w * SEARCH_X[1])
        y0 = int(h * SEARCH_Y[0]); y1 = int(h * SEARCH_Y[1])
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        return gray, x0, y0

    def _conf(self, frame: np.ndarray) -> float:
        """Template-match confidence against the search region."""
        gray, _, _ = self._search_region(frame)
        th, tw = self.template.shape[:2]
        if th > gray.shape[0] or tw > gray.shape[1]:
            return 0.0
        res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, v, _, _ = cv2.minMaxLoc(res)
        return float(v)
