"""
detection/game_clock_detector.py

Detects the NHL 25 end-of-game state (clock at 0:00 in the 3rd period)
using OpenCV template matching against a reference screenshot.

Usage
-----
Provide a cropped screenshot of the 0:00 / 3rd period HUD at:
    configs/game_end_template.png

The detector scans sampled frames from each segment and marks any segment
where the template matches as a "game_end" segment.
"""

from pathlib import Path
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Region of interest — top-center of the frame where the clock lives in NHL 25
# Adjust these fractions if the clock sits elsewhere on your display.
ROI_X = 0.30   # left edge  (fraction of width)
ROI_Y = 0.0    # top edge   (fraction of height)
ROI_W = 0.40   # width      (fraction of frame width)  → centre 40 %
ROI_H = 0.15   # height     (fraction of frame height) → top 15 %


class GameClockDetector:
    """
    Detects the end-of-game moment (0:00, 3rd period) via template matching.

    Args:
        template_path:  Path to a cropped screenshot of the 0:00 HUD state.
        threshold:      Match confidence threshold (0–1). Default 0.72.
        sample_frames:  How many frames to sample per segment.
        scale_factors:  Template scales to try (handles resolution differences).
    """

    def __init__(
        self,
        template_path: str | Path = "configs/game_end_template.png",
        threshold: float = 0.72,
        sample_frames: int = 10,
        scale_factors: list[float] = None,
    ) -> None:
        self.threshold = threshold
        self.sample_frames = sample_frames
        self.scale_factors = scale_factors or [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

        template_path = Path(template_path)
        if not template_path.exists():
            raise FileNotFoundError(
                f"Game-end template not found at {template_path}\n"
                "Take a screenshot of the 0:00 / 3rd period clock and save it to "
                "configs/game_end_template.png"
            )

        self.template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if self.template_gray is None:
            raise ValueError(f"Could not read template image: {template_path}")

        logger.info(
            "GameClockDetector ready — template size: %dx%d, threshold: %.2f",
            self.template_gray.shape[1], self.template_gray.shape[0], threshold,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def detect_in_segment(self, video_path: str | Path) -> dict:
        """
        Scan a video segment for the end-of-game clock state.

        Returns:
            {
                "detected":        bool,
                "max_confidence":  float,
                "best_frame":      int,
            }
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        frame_indices = np.linspace(0, total_frames - 1, self.sample_frames, dtype=int)

        max_conf = 0.0
        best_frame = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            conf = self._match_in_frame(frame)
            if conf > max_conf:
                max_conf = conf
                best_frame = int(idx)

        cap.release()

        detected = max_conf >= self.threshold
        if detected:
            logger.info(
                "Game-end clock detected in %s (conf=%.2f, frame=%d)",
                video_path.name, max_conf, best_frame,
            )

        return {
            "detected": detected,
            "max_confidence": float(max_conf),
            "best_frame": best_frame,
        }

    def score_segments(self, segments: list[dict]) -> list[dict]:
        """
        Scan all segments for the end-of-game clock and tag matching ones.

        Adds to each matching segment:
            label          = "game_end"
            game_end       = True
            confidence     = 1.0
            score          = 0.0  (will always be appended last; score unused)
        """
        logger.info("Scanning %d segments for end-of-game clock …", len(segments))
        found = 0
        for seg in segments:
            result = self.detect_in_segment(seg["path"])
            if result["detected"]:
                seg["label"] = "game_end"
                seg["game_end"] = True
                seg["confidence"] = 1.0
                seg["score"] = 0.0
                found += 1
        logger.info("Game-end clock found in %d segment(s).", found)
        return segments

    # ──────────────────────────────────────────────────────────────────────────

    def _match_in_frame(self, frame: np.ndarray) -> float:
        """Return the best template-match confidence within the clock ROI."""
        h, w = frame.shape[:2]

        # Crop to the clock region
        x0 = int(w * ROI_X)
        y0 = int(h * ROI_Y)
        x1 = min(w, int(w * (ROI_X + ROI_W)))
        y1 = min(h, int(h * (ROI_Y + ROI_H)))
        roi = frame[y0:y1, x0:x1]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        best = 0.0

        for scale in self.scale_factors:
            th, tw = self.template_gray.shape[:2]
            new_h = max(1, int(th * scale))
            new_w = max(1, int(tw * scale))

            if new_h > gray.shape[0] or new_w > gray.shape[1]:
                continue

            resized = cv2.resize(self.template_gray, (new_w, new_h))
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best:
                best = max_val

        return best
