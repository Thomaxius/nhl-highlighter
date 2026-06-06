"""
training/dataset.py

PyTorch Dataset for loading labelled video segments.

Expected directory layout in data/labeled/:
    data/labeled/
        goal/
            clip_001.mp4
            clip_002.mp4
        save/
            clip_010.mp4
        hit/
            ...
        other/
            ...
"""

from pathlib import Path
from typing import Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import logging

logger = logging.getLogger(__name__)


class NHLSegmentDataset(Dataset):
    """
    Loads short video segments from a labelled folder structure.

    Args:
        root_dir:    Path to data/labeled/ (or any dir with class subfolders).
        labels:      Ordered list of class names; inferred from subfolders if None.
        num_frames:  Number of frames to uniformly sample from each clip.
        transform:   Optional transform applied to the list of frames.
        split:       "train", "val", or "all" — applies an 80/20 train-val split.
        seed:        Random seed for reproducible splits.
    """

    def __init__(
        self,
        root_dir: str | Path,
        labels: Optional[list[str]] = None,
        num_frames: int = 16,
        transform: Optional[Callable] = None,
        split: str = "all",
        seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform

        # Discover classes
        if labels is None:
            self.labels = sorted(
                [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            )
        else:
            self.labels = labels
        self.label2idx = {name: i for i, name in enumerate(self.labels)}

        # Build sample list
        all_samples: list[tuple[Path, int]] = []
        video_extensions = ["*.mp4", "*.webm", "*.mov", "*.mkv"]
        for label in self.labels:
            class_dir = self.root_dir / label
            if not class_dir.exists():
                logger.warning("Label directory not found: %s", class_dir)
                continue
            for ext in video_extensions:
                for video_file in class_dir.glob(ext):
                    all_samples.append((video_file, self.label2idx[label]))

        # Split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_samples))
        split_at = int(len(indices) * 0.8)

        if split == "train":
            self.samples = [all_samples[i] for i in indices[:split_at]]
        elif split == "val":
            self.samples = [all_samples[i] for i in indices[split_at:]]
        else:
            self.samples = all_samples

        logger.info(
            "Dataset (%s): %d samples across %d classes",
            split, len(self.samples), len(self.labels),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        video_path, label_idx = self.samples[idx]

        npy_path = video_path.with_suffix(".npy")
        if npy_path.exists():
            # Fast path: load pre-decoded frame cache (T, H, W, C) uint8
            arr = np.load(npy_path)
            frames = [arr[i] for i in range(arr.shape[0])]
        else:
            frames = self._load_frames(video_path)

        if self.transform is not None:
            frames = self.transform(frames)

        # Frames: list of HxWxC numpy arrays → Tensor [T, C, H, W]
        frame_tensors = [
            torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float() / 255.0
            for f in frames
        ]
        video_tensor = torch.stack(frame_tensors)  # [T, C, H, W]

        return {
            "pixel_values": video_tensor,
            "labels": torch.tensor(label_idx, dtype=torch.long),
            "path": str(video_path),
        }

    def _load_frames(self, video_path: Path) -> list:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(total, 1)

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        cap.release()

        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        return frames
