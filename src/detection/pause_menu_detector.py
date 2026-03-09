"""
detection/pause_menu_detector.py

Detects the NHL 25 pause/ESC menu in video segments using template matching
against the top-left HUD region (where "PAUSE" text appears).

The menu is visually distinct: bright text on a near-black overlay in the
top-left corner, absent in normal gameplay.  A single sample frame is
sufficient for short menu segments; a small sample for longer ones.

Template: configs/pause_menu_template.png  (250×80 crop of top-left)
"""

from pathlib import Path
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Top-left region (fraction of frame) — matches where the template was cropped
ROI_Y2 = 0.12   # top 12% of height
ROI_X2 = 0.20   # left 20% of width

DEFAULT_TEMPLATE  = "configs/pause_menu_template.png"
DEFAULT_THRESHOLD = 0.65   # gameplay peaks at ~0.36; menu at ~0.89
DEFAULT_SAMPLES   = 6      # frames to sample per segment


class PauseMenuDetector:
    """
    Detects presence of the pause/ESC menu in a video segment.

    Args:
        template_path:  Path to the 250×80 pause-menu template PNG.
        threshold:      Match confidence above which the menu is considered
                        present.  Default 0.65.
        sample_frames:  Number of frames to sample per segment.  The menu
                        is typically shown for a whole scene so a few frames
                        are more than enough.
    """

    def __init__(
        self,
        template_path: str | Path = DEFAULT_TEMPLATE,
        threshold: float = DEFAULT_THRESHOLD,
        sample_frames: int = DEFAULT_SAMPLES,
    ) -> None:
        self.threshold     = threshold
        self.sample_frames = sample_frames

        tpath = Path(template_path)
        if not tpath.exists():
            raise FileNotFoundError(
                f"Pause-menu template not found: {tpath}. "
                "Expected configs/pause_menu_template.png"
            )
        tmpl = cv2.imread(str(tpath))
        if tmpl is None:
            raise ValueError(f"Could not read template image: {tpath}")
        self._template = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        logger.info(
            "PauseMenuDetector ready — template %dx%d, threshold=%.2f",
            self._template.shape[1], self._template.shape[0], threshold,
        )

    def detect(self, video_path: str | Path) -> dict:
        """
        Check whether the pause menu is visible in a video segment.

        Samples up to *sample_frames* evenly-spaced frames and runs template
        matching against the top-left ROI of each.  Returns as soon as any
        frame clears the threshold (fast-exit for confirmed menu segments).

        Returns:
            dict with keys:
              - ``detected``     bool — True if menu found
              - ``max_conf``     float — highest match confidence across frames
              - ``best_frame``   int  — 0-based frame index of best match
        """
        video_path = Path(video_path)
        cap  = cv2.VideoCapture(str(video_path))
        n    = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # ROI pixel dimensions
        roi_y2 = max(int(h * ROI_Y2), self._template.shape[0] + 1)
        roi_x2 = max(int(w * ROI_X2), self._template.shape[1] + 1)

        indices = np.linspace(0, n - 1, self.sample_frames, dtype=int)
        max_conf, best_frame = 0.0, 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            roi  = cv2.cvtColor(frame[:roi_y2, :roi_x2], cv2.COLOR_BGR2GRAY)
            # Guard: roi must be larger than template in both dimensions
            if roi.shape[0] < self._template.shape[0] or roi.shape[1] < self._template.shape[1]:
                continue
            res  = cv2.matchTemplate(roi, self._template, cv2.TM_CCOEFF_NORMED)
            conf = float(res.max())
            if conf > max_conf:
                max_conf, best_frame = conf, int(idx)
            if conf >= self.threshold:
                break  # fast-exit: confirmed

        cap.release()
        detected = max_conf >= self.threshold
        if detected:
            logger.info(
                "Pause menu detected in %s (conf=%.2f, t=%.1fs)",
                video_path.name, max_conf, best_frame / fps,
            )
        return {"detected": detected, "max_conf": max_conf, "best_frame": best_frame}
