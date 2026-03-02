"""
detection/banner_detector.py

Detects the NHL 25 "GOAL!" HUD banner in video segments using
OpenCV template matching. Fast, reliable, requires no GPU.

Usage
-----
Provide one or more cropped screenshots of the GOAL! banner saved at:
    configs/goal_banner_template.png
    configs/goal_banner_template2.png
    ... (any file matching configs/goal_banner_template*.png is auto-loaded)

Multiple templates are matched per frame; the highest confidence across
all templates is used. This handles team-specific banner colors.

The detector scans sampled frames from each segment and returns
True if the banner is found with sufficient confidence.
"""

from pathlib import Path
from typing import Optional
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Region of interest — top-left corner as a fraction of frame size
# Adjust if the banner sits in a different part of the screen
ROI_X = 0.0    # left edge (fraction of width)
ROI_Y = 0.0    # top edge  (fraction of height)
ROI_W = 0.35   # width     (fraction of frame width)
ROI_H = 0.15   # height    (fraction of frame height)


class BannerDetector:
    """
    Detects the GOAL! banner in video segments via template matching.

    Supports multiple templates (e.g. different team banner colors).
    The highest confidence match across all templates is used per frame.

    Args:
        template_path:  Path to a .png template, a list of paths, or a glob
                        pattern string (e.g. "configs/goal_banner_template*.png").
                        Defaults to auto-discovering all
                        configs/goal_banner_template*.png files.
        threshold:      Match confidence threshold (0–1). Default 0.75.
        sample_frames:  How many frames to sample per segment.
        scale_factors:  Template scales to try (handles slight resolution differences).
    """

    def __init__(
        self,
        template_path: str | Path | list[str | Path] | None = None,
        threshold: float = 0.60,
        sample_frames: int = 16,
        scale_factors: list[float] = None,
    ) -> None:
        self.threshold = threshold
        self.sample_frames = sample_frames
        # Wide scale range: templates 2-4 are ~610px wide (nearly full ROI at 1.0x),
        # so we need to search smaller scales in case the banner in-video is smaller
        # than the crop. 0.5–1.2x covers all realistic capture size differences.
        self.scale_factors = scale_factors or [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

        # ── Resolve template paths ────────────────────────────────────────
        if template_path is None:
            # Auto-discover all goal_banner_template*.png in configs/
            paths = sorted(Path("configs").glob("goal_banner_template*.png"))
        elif isinstance(template_path, list):
            paths = [Path(p) for p in template_path]
        else:
            p = Path(template_path)
            # If it looks like a glob pattern, expand it
            if "*" in str(p) or "?" in str(p):
                paths = sorted(p.parent.glob(p.name))
            else:
                paths = [p]

        if not paths:
            raise FileNotFoundError(
                "No goal banner templates found. Extract a frame with the GOAL! "
                "banner visible and save to configs/goal_banner_template.png\n"
                "  ffmpeg -ss 00:00:05 -i your_goal_clip.webm -frames:v 1 /tmp/frame.png"
            )

        self.templates_gray: list[np.ndarray] = []
        for tp in paths:
            if not tp.exists():
                raise FileNotFoundError(f"Goal banner template not found: {tp}")
            img = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read template image: {tp}")
            self.templates_gray.append(img)
            logger.info(
                "BannerDetector loaded template: %s (%dx%d)",
                tp.name, img.shape[1], img.shape[0],
            )

        logger.info(
            "BannerDetector ready — %d template(s), threshold: %.2f",
            len(self.templates_gray), threshold,
        )

    def detect_in_segment(self, video_path: str | Path) -> dict:
        """
        Scan a video segment for the GOAL! banner.

        Returns:
            {
                "detected": bool,
                "max_confidence": float,
                "best_frame":    int,   # frame index with highest match
            }
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = max(total_frames, 1)

        frame_indices = np.linspace(0, total_frames - 1, self.sample_frames, dtype=int)

        max_conf = 0.0
        best_frame = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            confidence = self._match_template_in_frame(frame)
            if confidence > max_conf:
                max_conf = confidence
                best_frame = int(idx)

        cap.release()

        detected = max_conf >= self.threshold
        if detected:
            logger.info(
                "GOAL banner DETECTED in %s (conf=%.3f, frame=%d)",
                video_path.name, max_conf, best_frame,
            )
        else:
            logger.debug(
                "No banner in %s (best_conf=%.3f, threshold=%.2f)",
                video_path.name, max_conf, self.threshold,
            )

        return {
            "detected": detected,
            "max_confidence": float(max_conf),
            "best_frame": best_frame,
        }

    def score_segments(self, segments: list[dict], pre_goal_s: float = 7.0, post_goal_s: float = 3.0) -> list[dict]:
        """
        Run banner detection on all segments and boost scores where detected.
        When a banner is found, also record trim points around the goal frame
        so the reel builder can extract just the relevant window.

        Args:
            segments:    List of segment dicts with "path" key.
            pre_goal_s:  Seconds to keep before the goal frame.
            post_goal_s: Seconds to keep after the goal frame.

        Modifies segments in-place. Returns the updated list.
        """
        for seg in segments:
            result = self.detect_in_segment(seg["path"])
            seg["banner_detected"] = result["detected"]
            seg["banner_confidence"] = result["max_confidence"]

            if result["detected"]:
                # Work out trim points from the detected frame + video fps
                cap = cv2.VideoCapture(str(seg["path"]))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()

                goal_s = result["best_frame"] / fps
                trim_start = max(0.0, goal_s - pre_goal_s)
                trim_end = min(total_frames / fps, goal_s + post_goal_s)

                seg["trim_start_s"] = trim_start
                seg["trim_end_s"] = trim_end

                # Override label and give a very high score
                seg["label"] = "goal"
                seg["confidence"] = 1.0
                seg["score"] = seg.get("score", 0.0) + 2.0  # fixed large boost
                logger.info(
                    "Banner boost applied to: %s — trim %.1fs → %.1fs (goal at %.1fs)",
                    Path(seg["path"]).name, trim_start, trim_end, goal_s,
                )

        return segments

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _match_template_in_frame(self, frame: np.ndarray) -> float:
        """Return the best template match confidence for a single frame."""
        h, w = frame.shape[:2]

        # Crop to ROI
        x1 = int(ROI_X * w)
        y1 = int(ROI_Y * h)
        x2 = int((ROI_X + ROI_W) * w)
        y2 = int((ROI_Y + ROI_H) * h)
        roi = frame[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        best = 0.0
        for t_idx, template_gray in enumerate(self.templates_gray):
            t_best = 0.0
            for scale in self.scale_factors:
                th, tw = template_gray.shape
                new_w = int(tw * scale)
                new_h = int(th * scale)

                if new_w > roi_gray.shape[1] or new_h > roi_gray.shape[0]:
                    continue

                scaled = cv2.resize(template_gray, (new_w, new_h))
                result = cv2.matchTemplate(roi_gray, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > t_best:
                    t_best = max_val

            logger.debug("  template[%d] best_conf=%.3f", t_idx, t_best)
            if t_best > best:
                best = t_best

        return best


def scan_segment_for_banner(
    video_path: str | Path,
    template_path: str | Path | list[str | Path] | None = None,
    threshold: float = 0.75,
) -> bool:
    """Convenience one-shot function — returns True if GOAL banner found."""
    detector = BannerDetector(template_path=template_path, threshold=threshold)
    result = detector.detect_in_segment(video_path)
    return result["detected"]
