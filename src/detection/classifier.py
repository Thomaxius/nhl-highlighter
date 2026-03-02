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
DEFAULT_LABELS = ["goal", "celebration", "replay", "other"]


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

    def classify_segment(self, video_path: str | Path) -> dict:
        """
        Classify a single video segment.

        Returns:
            {"path": str, "label": str, "confidence": float, "scores": dict}
        """
        frames = self._load_frames(Path(video_path))
        if frames is None:
            return {"path": str(video_path), "label": "other", "confidence": 0.0, "scores": {}}

        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        best_idx = int(np.argmax(probs))
        scores = {label: float(probs[i]) for i, label in enumerate(self.labels)}

        return {
            "path": str(video_path),
            "label": self.labels[best_idx],
            "confidence": float(probs[best_idx]),
            "scores": scores,
        }

    def classify_segments(self, video_paths: list[Path]) -> list[dict]:
        """Classify a list of segment paths and return results."""
        results = []
        for path in video_paths:
            logger.info("Classifying: %s", Path(path).name)
            results.append(self.classify_segment(path))
        return results

    def is_highlight(self, result: dict, min_confidence: float = 0.55) -> bool:
        """Return True if a segment is highlight-worthy."""
        highlight_labels = {"goal", "save", "hit", "fight", "celebration", "replay"}
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
