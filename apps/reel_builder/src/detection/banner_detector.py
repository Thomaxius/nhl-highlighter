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
                        pattern string (e.g. "apps/reel_builder/configs/goal_banner_template*.png").
                        Defaults to auto-discovering all
                        apps/reel_builder/configs/goal_banner_template*.png files.
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
            # Auto-discover all goal_banner_template*.png in apps/reel_builder/configs/
            paths = sorted(Path("apps/reel_builder/configs").glob("goal_banner_template*.png"))
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

    def scan_video(
        self,
        video_path: str | Path,
        interval_s: float = 0.5,
        merge_gap_s: float = 3.0,
    ) -> list[float]:
        """
        Scan a full (non-split) video for GOAL banner frames.

        Samples every *interval_s* seconds by seeking, which means it runs
        on the raw high-quality video and is not affected by scene-cut
        boundary artifacts that reduce per-segment confidence.

        Args:
            video_path:   Path to the normalised .mp4 file.
            interval_s:   Seconds between sampled frames (default 0.5 → 2 fps).
            merge_gap_s:  Consecutive banner timestamps within this window are
                          collapsed into a single event (keeps only the first).

        Returns:
            Sorted list of timestamps (seconds) where the banner was detected.
            Consecutive detections within *merge_gap_s* are merged to one event.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        step = max(1, int(fps * interval_s))
        indices = range(0, total_frames, step)
        n_samples = len(indices)
        # For a full-video scan the source is always the normalised 1920×1080 file
        # so the banner appears at a predictable scale. Use only 3 scales to cut
        # the per-frame cost from 40 matchTemplate calls down to 15, ~2.5× faster.
        _fast_scales = [0.8, 0.9, 1.0]
        logger.info(
            "BannerDetector: scanning %s — %d samples at %.1fs intervals (fast scales)",
            video_path.name, n_samples, interval_s,
        )

        # Read sampled frames sequentially (VideoCapture is not thread-safe)
        # in small chunks, matching each chunk in parallel (OpenCV/numpy
        # releases the GIL) before reading the next. Holding all samples at
        # once would need ~6 MB per 1080p frame — tens of GB for a full game —
        # which OOM-killed the pipeline; a chunk caps peak memory at ~200 MB.
        import os
        from concurrent.futures import ThreadPoolExecutor

        # Default cap 4 (server); --local (NHL_LOCAL=1) lifts it to 8.
        _cap = 8 if os.environ.get("NHL_LOCAL") == "1" else 4
        workers = min(_cap, max(1, os.cpu_count() or 2))
        CHUNK_FRAMES = 32

        def _match(item: tuple[int, np.ndarray]) -> tuple[int, float]:
            idx, frm = item
            return idx, self._match_template_in_frame(frm, scale_factors=_fast_scales)

        raw_times: list[float] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:

            def _flush(chunk: list[tuple[int, np.ndarray]]) -> None:
                for idx, conf in pool.map(_match, chunk):
                    if conf >= self.threshold:
                        t = idx / fps
                        raw_times.append(t)
                        logger.info(
                            "  Raw-scan banner hit: t=%.2fs  frame=%d  conf=%.3f",
                            t, idx, conf,
                        )

            chunk: list[tuple[int, np.ndarray]] = []
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    chunk.append((i, frame))
                if len(chunk) >= CHUNK_FRAMES:
                    _flush(chunk)
                    chunk = []
            if chunk:
                _flush(chunk)
        cap.release()
        raw_times.sort()

        # Merge consecutive hits within merge_gap_s into a single event
        merged: list[float] = []
        for t in raw_times:
            if not merged or t - merged[-1] > merge_gap_s:
                merged.append(t)

        logger.info(
            "BannerDetector: raw scan found %d hit(s) → %d merged goal event(s)",
            len(raw_times), len(merged),
        )
        return merged

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

    def _match_template_in_frame(
        self,
        frame: np.ndarray,
        scale_factors: list[float] | None = None,
        early_exit_threshold: float | None = None,
    ) -> float:
        """
        Return the best template match confidence for a single frame.

        Args:
            scale_factors:          Override scale list (uses self.scale_factors by default).
            early_exit_threshold:   Stop early if any template hits this confidence.
                                    Defaults to self.threshold.
        """
        scales = scale_factors if scale_factors is not None else self.scale_factors
        early_exit = early_exit_threshold if early_exit_threshold is not None else self.threshold

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
            for scale in scales:
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
            # Early exit — already above threshold, no need to try remaining templates
            if best >= early_exit:
                break

        return best


class AssistsBannerDetector:
    """
    Detects the "assists" banner that appears at the faceoff after a goal.

    Same ROI as the GOAL banner (top-left HUD area).  Uses a single template
    (configs/after_goal_assists_template.png) and a full-video scan identical
    to ``BannerDetector.scan_video``.

    Each detected timestamp means "a goal was scored ~10-25 s before this".
    """

    # The assists banner sits in the same top-left HUD zone as the GOAL banner
    ASSISTS_ROI_X = 0.0
    ASSISTS_ROI_Y = 0.0
    ASSISTS_ROI_W = 0.55   # wider — the assists text + player name is ~950px
    ASSISTS_ROI_H = 0.15

    def __init__(
        self,
        template_path: str | Path | None = None,
        threshold: float = 0.65,
        scale_factors: list[float] | None = None,
    ) -> None:
        self.threshold = threshold
        self.scale_factors = scale_factors or [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        if template_path is None:
            # Auto-discover any after_goal_assists_template*.png in apps/reel_builder/configs/
            candidates = sorted(Path("apps/reel_builder/configs").glob("after_goal_assists_template*.png"))
            if not candidates:
                raise FileNotFoundError(
                    "No assists banner templates found in apps/reel_builder/configs/after_goal_assists_template*.png"
                )
            tp = candidates[-1]  # use the latest version
        else:
            tp = Path(template_path)
        if not tp.exists():
            raise FileNotFoundError(f"Assists banner template not found: {tp}")
        img = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read assists template image: {tp}")
        self.template_gray = img
        logger.info(
            "AssistsBannerDetector loaded: %s (%dx%d), threshold=%.2f",
            tp.name, img.shape[1], img.shape[0], threshold,
        )

    def scan_video(
        self,
        video_path: str | Path,
        interval_s: float = 1.0,
        merge_gap_s: float = 15.0,
    ) -> list[float]:
        """
        Scan a normalised video for assists-banner frames.

        Returns sorted list of timestamps (seconds) where the assists banner
        was detected.  Consecutive hits within *merge_gap_s* are merged.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        step = max(1, int(fps * interval_s))
        indices = range(0, total_frames, step)
        logger.info(
            "AssistsBannerDetector: scanning %s — %d samples at %.1fs intervals",
            video_path.name, len(indices), interval_s,
        )

        raw_times: list[float] = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            conf = self._match_frame(frame)
            if conf >= self.threshold:
                t = i / fps
                raw_times.append(t)
                logger.info(
                    "  Assists banner hit: t=%.2fs  frame=%d  conf=%.3f",
                    t, i, conf,
                )
        cap.release()

        merged: list[float] = []
        for t in raw_times:
            if not merged or t - merged[-1] > merge_gap_s:
                merged.append(t)

        logger.info(
            "AssistsBannerDetector: %d raw hit(s) → %d merged assists event(s)",
            len(raw_times), len(merged),
        )
        return merged

    def _match_frame(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        x1 = int(self.ASSISTS_ROI_X * w)
        y1 = int(self.ASSISTS_ROI_Y * h)
        x2 = int((self.ASSISTS_ROI_X + self.ASSISTS_ROI_W) * w)
        y2 = int((self.ASSISTS_ROI_Y + self.ASSISTS_ROI_H) * h)
        roi = frame[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        best = 0.0
        for scale in self.scale_factors:
            th, tw = self.template_gray.shape
            new_w = int(tw * scale)
            new_h = int(th * scale)
            if new_w > roi_gray.shape[1] or new_h > roi_gray.shape[0]:
                continue
            scaled = cv2.resize(self.template_gray, (new_w, new_h))
            result = cv2.matchTemplate(roi_gray, scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best:
                best = max_val
            if best >= self.threshold:
                break
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
