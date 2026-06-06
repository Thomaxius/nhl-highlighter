"""
detection/classifier.py

Runs a fine-tuned VideoMAE (or SlowFast) model over video segments and
returns a highlight label + confidence score for each segment.
"""

from pathlib import Path
from typing import Optional
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Labels must match the order used during training.
DEFAULT_LABELS = ["goal", "celebration", "goal_replay", "other_replay", "scoring_chance", "faceoff_cutscene", "faceoff", "other"]


class HighlightClassifier:
    """
    Wraps a fine-tuned VideoMAE / TimeSformer model loaded from a local
    checkpoint for inference.

    Example
    -------
    >>> clf = HighlightClassifier("models/checkpoints/videomae_nhl.pt")
    >>> results = clf.classify_segments(segment_paths)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        labels: list[str] = None,
        device: Optional[str] = None,
        num_frames: int = 16,
    ) -> None:
        self.num_frames = num_frames
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor, self.labels = self._load_model(checkpoint_path, labels)
        logger.info("Classifier loaded on %s with labels: %s", self.device, self.labels)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Clips longer than this (seconds) get sliding-window inference so a short
    # goal moment at the end of a long scene isn't drowned out by uniform sampling.
    WINDOW_THRESHOLD_S = 10.0
    WINDOW_SIZE_S      = 4.0   # match typical training clip length
    WINDOW_STRIDE_S    = 2.0   # 50 % overlap

    def classify_segment(self, video_path: str | Path) -> dict:
        """
        Classify a single video segment.

        For clips longer than WINDOW_THRESHOLD_S, runs the model over overlapping
        windows matching training clip length and takes the highest-confidence
        non-other prediction (or best other if everything else is low).

        Returns:
            {"path": str, "label": str, "confidence": float, "scores": dict}
        """
        import cv2 as _cv2
        path = Path(video_path)
        cap = _cv2.VideoCapture(str(path))
        fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration_s = total_frames / fps

        if duration_s > self.WINDOW_THRESHOLD_S:
            return self._classify_windowed(path, fps, total_frames, duration_s)

        return self._classify_frames(path, self._load_frames(path))

    def _classify_frames(self, video_path: Path, frames) -> dict:
        """Run a single model forward pass on a pre-loaded frame list."""
        if frames is None:
            return {"path": str(video_path), "label": "other", "confidence": 0.0, "scores": {}}

        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0].cpu().numpy()

        best_idx = int(np.argmax(probs))
        return {
            "path": str(video_path),
            "label": self.labels[best_idx],
            "confidence": float(probs[best_idx]),
            "scores": {label: float(probs[i]) for i, label in enumerate(self.labels)},
        }

    def _classify_windowed(self, video_path: Path, fps: float, total_frames: int, duration_s: float) -> dict:
        """
        Slide a fixed-length window over a long clip and return the prediction
        with the highest non-other confidence.  Falls back to the best overall
        prediction if every window confidently says 'other'.
        """
        import cv2 as _cv2

        window_frames_n = int(self.WINDOW_SIZE_S * fps)
        stride_frames_n = int(self.WINDOW_STRIDE_S * fps)
        window_frames_n = max(window_frames_n, self.num_frames)  # must have enough frames to sample

        cap = _cv2.VideoCapture(str(video_path))

        best_result = None         # best non-other result
        best_other_result = None   # fallback if every window says other
        best_window_start_s = 0.0  # absolute start time (s) of the winning window

        start = 0
        while start < total_frames:
            end = min(start + window_frames_n, total_frames)
            window_start_s = start / fps
            indices = np.linspace(start, end - 1, self.num_frames, dtype=int)
            frames = []
            for idx in indices:
                cap.set(_cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frame = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                    frame = _cv2.resize(frame, (224, 224))
                    frames.append(frame)

            if len(frames) == self.num_frames:
                result = self._classify_frames(video_path, frames)
                if result["label"] != "other":
                    if best_result is None or result["confidence"] > best_result["confidence"]:
                        best_result = result
                        best_window_start_s = window_start_s
                else:
                    if best_other_result is None or result["confidence"] > best_other_result["confidence"]:
                        best_other_result = result

            if end >= total_frames:
                break
            start += stride_frames_n

        cap.release()

        # TODO(multi-peak): instead of a single winner, collect all windows above a
        # confidence threshold (e.g. 85%), deduplicate by proximity, and emit multiple
        # trim ranges from one segment — so a 64s scene with 3 scoring chances surfaces
        # all 3 clips instead of just the highest-confidence one.
        winner = best_result if best_result is not None else best_other_result
        if winner is None:
            return {"path": str(video_path), "label": "other", "confidence": 0.0, "scores": {}}

        # For scoring_chance, store the window timestamp so the reel builder
        # can trim to just the relevant action (7s before → 3s after window end).
        SC_PRE_S  = 7.0
        SC_POST_S = 3.0
        if winner["label"] == "scoring_chance" and best_result is not None:
            trim_start = max(0.0, best_window_start_s - SC_PRE_S)
            trim_end   = min(duration_s, best_window_start_s + self.WINDOW_SIZE_S + SC_POST_S)
            winner["trim_start_s"] = trim_start
            winner["trim_end_s"]   = trim_end
            logger.debug(
                "  SC trim: window at %.1fs → trim [%.1fs, %.1fs]",
                best_window_start_s, trim_start, trim_end,
            )

        logger.debug(
            "  Windowed inference: %s → %s (%.0f%%) over %.1fs",
            video_path.name, winner["label"], winner["confidence"] * 100, duration_s,
        )
        return winner

    def classify_segments(self, video_paths: list[Path]) -> list[dict]:
        """Classify a list of segment paths and return results."""
        results = []
        for path in video_paths:
            logger.info("Classifying: %s", Path(path).name)
            results.append(self.classify_segment(path))
        return results

    def is_highlight(self, result: dict, min_confidence: float = 0.55) -> bool:
        """Return True if a segment is highlight-worthy."""
        highlight_labels = {"goal", "save", "hit", "fight", "celebration", "goal_replay", "scoring_chance"}
        return (
            result["label"] in highlight_labels
            and result["confidence"] >= min_confidence
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint_path: str | Path, labels: list[str] = None):
        from transformers import VideoMAEForVideoClassification, AutoProcessor
        import json

        checkpoint_path = Path(checkpoint_path)

        # Read labels from the saved checkpoint config first
        config_file = checkpoint_path / "config.json"
        if labels is None and config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            id2label = cfg.get("id2label", {})
            if id2label:
                labels = [id2label[str(i)] for i in range(len(id2label))]
                logger.info("Labels loaded from checkpoint config: %s", labels)

        if labels is None:
            labels = DEFAULT_LABELS
            logger.warning("Could not read labels from checkpoint, using defaults: %s", labels)

        if checkpoint_path.exists():
            logger.info("Loading fine-tuned checkpoint: %s", checkpoint_path)
            model = VideoMAEForVideoClassification.from_pretrained(
                str(checkpoint_path),
                num_labels=len(labels),
                ignore_mismatched_sizes=True,
            )
            processor = AutoProcessor.from_pretrained(str(checkpoint_path))
        else:
            logger.warning(
                "Checkpoint not found at %s — loading base VideoMAE weights.", checkpoint_path
            )
            model_id = "MCG-NJU/videomae-base"
            model = VideoMAEForVideoClassification.from_pretrained(
                model_id,
                num_labels=len(labels),
                ignore_mismatched_sizes=True,
            )
            processor = AutoProcessor.from_pretrained(model_id)

        model.to(self.device).eval()
        return model, processor, labels

    def _load_frames(self, video_path: Path) -> Optional[list]:
        """Sample *num_frames* evenly spaced frames from the video."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            logger.warning("Could not read frames from %s", video_path.name)
            cap.release()
            return None

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)

        cap.release()

        if len(frames) < self.num_frames:
            # Pad with last frame if video is short
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

        return frames
