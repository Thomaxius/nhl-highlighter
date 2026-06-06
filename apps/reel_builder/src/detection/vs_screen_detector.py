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
# Centre band to search — tight fit around template3 (144x543 at 1920x1080)
# Horizontal: 45.2-54.8% (184px wide, fits 144px template)
# Vertical: 36.0-89.9% (582px tall, fits 543px template)
SEARCH_X = (0.452, 0.548)
SEARCH_Y = (0.360, 0.899)


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
        template_path: str | Path = "configs/vs_screen_template3.png",
        threshold: float = MIN_CONF,
    ) -> None:
        template_path = Path(template_path)
        self.template_paths = self._resolve_template_paths(template_path)
        if not self.template_paths:
            raise FileNotFoundError(
                f"VS screen template not found: {template_path}\n"
                "Run the extraction step first or check configs/."
            )
        self.templates: list[np.ndarray] = []
        for path in self.template_paths:
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise ValueError(f"Could not read template: {path}")
            self.templates.append(template)
        self.threshold = threshold
        sizes = ", ".join(
            f"{path.name}={template.shape[1]}x{template.shape[0]}"
            for path, template in zip(self.template_paths, self.templates, strict=True)
        )
        logger.info(
            "VsScreenDetector ready — %d template(s), threshold %.2f (%s)",
            len(self.templates), threshold, sizes,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def match_frame(self, frame: np.ndarray) -> float:
        """Return template-match confidence for a single frame (0–1)."""
        return self._conf(frame)

    def find_matches(
        self,
        video_path: str | Path,
        interval_s: float = 0.5,
        search_window_s: float = 60.0,
        search_start_s: float = 0.0,
    ) -> list[tuple[float, float]]:
        """
        Return all sampled timestamps that exceed the configured threshold.

        Each result is ``(timestamp_s, confidence)``.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        max_t = min(search_window_s, total_frames / fps)

        t = max(0.0, search_start_s)
        matches: list[tuple[float, float]] = []
        best_conf = float("-inf")
        best_t = t

        while t <= max_t:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            conf = self._conf(frame)
            if conf > best_conf:
                best_conf = conf
                best_t = t
            if conf >= self.threshold:
                matches.append((t, conf))

            t += interval_s

        cap.release()
        if not matches and best_conf != float("-inf"):
            logger.info(
                "VS screen not detected in %s — best conf=%.3f at t=%.2fs (threshold %.2f)",
                Path(video_path).name, best_conf, best_t, self.threshold,
            )
        return matches

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
        matches = self.find_matches(
            video_path,
            interval_s=interval_s,
            search_window_s=search_window_s,
            search_start_s=search_start_s,
        )
        if not matches:
            return None

        t, conf = matches[0]
        logger.info(
            "VS screen detected at t=%.2fs  conf=%.3f  (%s)",
            t, conf, Path(video_path).name,
        )
        return t

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

    def _resolve_template_paths(self, template_path: Path) -> list[Path]:
        if template_path.exists() and template_path.is_file():
            return [template_path]
        return sorted(template_path.parent.glob(template_path.name))

    def _conf(self, frame: np.ndarray) -> float:
        """Template-match confidence against the search region."""
        gray, _, _ = self._search_region(frame)
        best = 0.0
        for template in self.templates:
            th, tw = template.shape[:2]
            if th > gray.shape[0] or tw > gray.shape[1]:
                continue
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, v, _, _ = cv2.minMaxLoc(res)
            best = max(best, float(v))
        return best
