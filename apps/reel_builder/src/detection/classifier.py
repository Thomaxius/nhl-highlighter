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

    # Multi-peak windowing: a long scene can hold more than one scoring chance,
    # and the chance that scored highest is not always the one the viewer cares
    # about (a quiet breakaway loses to a louder scrum later in the same 30s
    # scene). Collect every window that agrees with the winning label and clears
    # one of these gates, merge them into time-clusters, and emit one trim per
    # cluster instead of a single winner.
    PEAK_MIN_CONF      = 0.45   # a window counts as a peak at/above this confidence…
    PEAK_REL_MARGIN    = 0.20   # …or within this margin of the best window
    PEAK_CLUSTER_GAP_S = 4.0    # peaks closer than this in time merge into one clip
    PEAK_MAX_CLUSTER_S = 20.0   # a single cluster can't grow past this many seconds
    PEAK_MAX_EXTRA     = 2      # cap on extra clips emitted from one scene
    SC_PRE_S           = 7.0    # scoring-chance lead-in kept before the peak
    SC_POST_S          = 5.0    # …and tail kept after it (a shot / save often lands
                                 # just past the last confidently-classified window)

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
        Slide a fixed-length window over a long clip. The headline label is the
        single most-confident non-other window; falls back to the best 'other'
        window if nothing else fires.

        For a scoring chance, the trim is not anchored on that top window —
        instead every window sharing the winning label is clustered by time
        (see the PEAK_* constants) and the EARLIEST cluster becomes the primary
        clip, because a chance builds at its first shot and the rest is
        aftermath. Any further clusters are attached as ``sc_extra_windows`` so
        the reel builder can emit them as their own clips.
        """
        import cv2 as _cv2

        window_frames_n = int(self.WINDOW_SIZE_S * fps)
        stride_frames_n = int(self.WINDOW_STRIDE_S * fps)
        window_frames_n = max(window_frames_n, self.num_frames)  # must have enough frames to sample

        cap = _cv2.VideoCapture(str(video_path))

        windows: list[dict] = []   # every non-other window, in time order
        best_other_result = None   # fallback if every window says other

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
                result["window_start_s"] = window_start_s
                result["window_end_s"]   = min(duration_s, window_start_s + self.WINDOW_SIZE_S)
                if result["label"] != "other":
                    windows.append(result)
                elif best_other_result is None or result["confidence"] > best_other_result["confidence"]:
                    best_other_result = result

            if end >= total_frames:
                break
            start += stride_frames_n

        cap.release()

        if not windows:
            if best_other_result is None:
                return {"path": str(video_path), "label": "other", "confidence": 0.0, "scores": {}}
            return best_other_result

        # Headline label/confidence: the single most confident non-other window.
        best = max(windows, key=lambda w: w["confidence"])
        winner = dict(best)

        # ── Cluster every window that agrees with the winning label ──────────
        # A goal-mouth scramble flickers between 'goal' and 'scoring_chance'
        # window to window — to the model they're the same crease-crowding
        # phenomenon read at slightly different confidence, and a 'goal' winner
        # is heading for exactly this fate itself (Step 4c demotes any
        # banner-less goal, Step 4f.5 rescues it back as a scoring_chance). So
        # when the winner isn't already scoring_chance, pool both labels when
        # clustering — otherwise the climax can sit in a run of windows the
        # model tagged scoring_chance and never join the 'goal' cluster the
        # trim gets anchored on, cutting the reel clip before the shot.
        peak_labels = {best["label"]} if best["label"] == "scoring_chance" else {best["label"], "scoring_chance"}
        # "Or" means the LOWER of the two bars (min), not the higher one (max).
        # max() made the gate *stricter* whenever the scene had one very
        # confident window — backwards, since that's exactly when the quieter
        # build-up/follow-through windows around the real peak most need the
        # margin clause to let them in. (Confirmed against a real reel: a
        # scoring-chance clip was ending a few seconds before the actual shot
        # because the window that showed it fell just short of this gate.)
        conf_gate = min(self.PEAK_MIN_CONF, best["confidence"] - self.PEAK_REL_MARGIN)
        peaks = sorted(
            (w for w in windows if w["label"] in peak_labels and w["confidence"] >= conf_gate),
            key=lambda w: w["window_start_s"],
        )
        # A run of merely-adequate windows (generic zone possession that never
        # becomes a real chance) can otherwise bridge gap after gap and pull an
        # entire 30-60s scene into "one cluster" — confirmed on a real reel,
        # where two whole scenes (58.3s of raw footage) became a single 57.4s
        # clip with no real trim at all. Refuse to grow a cluster past
        # PEAK_MAX_CLUSTER_S; a window that would push it over starts a new
        # cluster instead; real scoring chances observed so far run 8-16s.
        clusters: list[dict] = []
        for w in peaks:
            if (
                clusters
                and w["window_start_s"] - clusters[-1]["end_s"] <= self.PEAK_CLUSTER_GAP_S
                and max(clusters[-1]["end_s"], w["window_end_s"]) - clusters[-1]["start_s"]
                    <= self.PEAK_MAX_CLUSTER_S
            ):
                clusters[-1]["end_s"]      = max(clusters[-1]["end_s"], w["window_end_s"])
                clusters[-1]["confidence"] = max(clusters[-1]["confidence"], w["confidence"])
            else:
                clusters.append({
                    "start_s": w["window_start_s"],
                    "end_s": w["window_end_s"],
                    "confidence": w["confidence"],
                })
        if not clusters:  # best window itself failed the gate (margin only) — use it
            clusters = [{
                "start_s": best["window_start_s"],
                "end_s": best["window_end_s"],
                "confidence": best["confidence"],
            }]

        # A scoring chance is anchored on its FIRST peak (the shot that started
        # it); other labels keep the highest-confidence cluster.
        if best["label"] == "scoring_chance":
            primary, extra = clusters[0], clusters[1:]
        else:
            primary = max(clusters, key=lambda c: c["confidence"])
            extra = [c for c in clusters if c is not primary]

        winner["window_start_s"] = primary["start_s"]
        winner["window_end_s"]   = primary["end_s"]

        # Extra, temporally-separate peaks get attached regardless of the
        # winning label — not only when the winner is already scoring_chance.
        # A 'goal' winner here is routinely a banner-less false read that Step
        # 4c demotes and Step 4f.5 rescues back to scoring_chance later in the
        # pipeline, and a second, unrelated chance elsewhere in the same long
        # scene (e.g. an early net-front scramble, then an unconnected shot 10s
        # later) is exactly as real either way. Harmless for a segment that
        # stays a genuine banner-confirmed goal or gets dropped as 'other' —
        # the reel builder only reads this key off a segment whose final label
        # is scoring_chance.
        if extra:
            winner["sc_extra_windows"] = [
                {
                    "trim_start_s": max(0.0, c["start_s"] - self.SC_PRE_S),
                    "trim_end_s":   min(duration_s, c["end_s"] + self.SC_POST_S),
                    "confidence":   c["confidence"],
                }
                for c in extra[: self.PEAK_MAX_EXTRA]
            ]

        if winner["label"] == "scoring_chance":
            winner["trim_start_s"] = max(0.0, primary["start_s"] - self.SC_PRE_S)
            winner["trim_end_s"]   = min(duration_s, primary["end_s"] + self.SC_POST_S)

        logger.debug(
            "  Windowed inference: %s → %s (%.0f%%) over %.1fs  [%d peak(s), window %.1f-%.1fs%s]",
            video_path.name, winner["label"], winner["confidence"] * 100, duration_s, len(clusters),
            winner["window_start_s"], winner["window_end_s"],
            f", +{len(winner['sc_extra_windows'])} extra" if extra else "",
        )
        return winner

    def classify_segments(self, video_paths: list[Path]) -> list[dict]:
        """Classify a list of segment paths and return results in original order.

        Uses a ThreadPoolExecutor so frame decoding (OpenCV, GIL-released) and
        PyTorch CPU inference (also GIL-released) overlap across clips.
        Workers are capped at 3 to avoid CPU oversubscription on inference;
        --local mode (NHL_LOCAL=1) raises the cap to 6 and uses all cores.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if os.environ.get("NHL_LOCAL") == "1":
            workers = min(6, max(1, os.cpu_count() or 2))
        else:
            workers = min(3, max(1, (os.cpu_count() or 2) // 2))

        def _run(path):
            logger.info("Classifying: %s", Path(path).name)
            return self.classify_segment(path)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run, p): i for i, p in enumerate(video_paths)}
            indexed = [None] * len(video_paths)
            for future in as_completed(futures):
                indexed[futures[future]] = future.result()

        return indexed

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
