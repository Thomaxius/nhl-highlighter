"""
scripts/preprocess_videos.py

Pre-decodes every video in data/labeled/ into a .npy frame cache so that
training loads arrays directly instead of seeking/decoding video on the fly.

Each .npy file is saved next to its source video with the same stem:
    data/labeled/goal/clip_001.mp4  →  data/labeled/goal/clip_001.npy

Array shape: (num_frames, 224, 224, 3)  dtype: uint8

Usage
-----
    python -m scripts.preprocess_videos
    python -m scripts.preprocess_videos --data_dir data/labeled --num_frames 16
    python -m scripts.preprocess_videos --force   # re-encode even if .npy exists
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = [".mp4", ".webm", ".mov", ".mkv"]


def decode_frames(video_path: Path, num_frames: int) -> np.ndarray:
    """Decode *num_frames* uniformly-sampled frames from *video_path*.

    Returns an array of shape (num_frames, 224, 224, 3) uint8.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, 1)

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()

    # Pad with last frame (or black) if decode failed for some frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    return np.stack(frames, axis=0)  # (T, H, W, C)


def _process_one(args_tuple: tuple) -> tuple[str, str]:
    """Worker function: decode one video and save .npy. Returns (status, message)."""
    video_path, num_frames, force = args_tuple
    npy_path = Path(video_path).with_suffix(".npy")
    if npy_path.exists() and not force:
        return "skipped", str(video_path)
    try:
        frames = decode_frames(Path(video_path), num_frames)
        np.save(npy_path, frames)
        return "ok", f"{video_path} → {npy_path.name}  {frames.shape}"
    except Exception as exc:
        return "error", f"{video_path}: {exc}"


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-decode videos to .npy frame caches")
    p.add_argument("--data_dir",   default="data/labeled", help="Root of labeled data")
    p.add_argument("--num_frames", type=int, default=16,   help="Frames per clip (must match training)")
    p.add_argument("--workers",    type=int, default=8,    help="Parallel worker processes (default: 8)")
    p.add_argument("--force",      action="store_true",    help="Re-encode even if .npy already exists")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("data_dir not found: %s", data_dir)
        sys.exit(1)

    videos = [
        f for f in data_dir.rglob("*")
        if f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        logger.error("No video files found under %s", data_dir)
        sys.exit(1)

    logger.info("Found %d video files — using %d workers", len(videos), args.workers)

    skipped = processed = errors = 0
    work = [(str(v), args.num_frames, args.force) for v in videos]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, w): w[0] for w in work}
        for done_idx, future in enumerate(as_completed(futures), 1):
            status, msg = future.result()
            if status == "ok":
                processed += 1
                logger.info("[%d/%d] %s", done_idx, len(videos), msg)
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                logger.error("[%d/%d] FAILED %s", done_idx, len(videos), msg)

    logger.info(
        "Done. processed=%d  skipped=%d  errors=%d",
        processed, skipped, errors,
    )
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
