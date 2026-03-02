"""
ingestion/ingest.py

Handles importing raw .mp4 clips from a folder (USB dump, PS App export, etc.)
and normalising them to a consistent format using FFmpeg.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Target spec for all processed clips
TARGET_FPS = 30
TARGET_RESOLUTION = "1920x1080"
TARGET_CODEC = "libx264"
TARGET_AUDIO_CODEC = "aac"
TARGET_AUDIO_RATE = 44100


def ingest_clips(
    source_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """
    Walk *source_dir* for .mp4 / .mov / .mkv files, normalise each one
    with FFmpeg, and write the result to *output_dir*.

    Returns a list of successfully processed output paths.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    clips = [p for p in source_dir.rglob("*") if p.suffix.lower() in video_extensions]

    if not clips:
        logger.warning("No video files found in %s", source_dir)
        return []

    processed: list[Path] = []
    for clip in clips:
        out_path = output_dir / (clip.stem + "_norm.mp4")
        if out_path.exists() and not overwrite:
            if out_path.stat().st_size > 10_000:  # skip only if file looks valid (>10KB)
                logger.info("Skipping (already exists): %s", out_path.name)
                processed.append(out_path)
                continue
            else:
                logger.warning("Existing file looks broken (too small), re-encoding: %s", out_path.name)
                out_path.unlink()

        logger.info("Normalising: %s → %s", clip.name, out_path.name)
        success = _ffmpeg_normalise(clip, out_path)
        if success:
            processed.append(out_path)

    return processed


def _ffmpeg_normalise(src: Path, dst: Path) -> bool:
    """Run FFmpeg to normalise a single clip."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1:color=black,fps={TARGET_FPS}",
        "-c:v", TARGET_CODEC,
        "-preset", "fast",
        "-crf", "18",
        "-c:a", TARGET_AUDIO_CODEC,
        "-ar", str(TARGET_AUDIO_RATE),
        "-movflags", "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg error for %s:\n%s", src.name, result.stderr[-500:])
        return False
    return True


def copy_raw_clips(source_dir: str | Path, raw_dir: str | Path) -> list[Path]:
    """
    Convenience function: copy clips from a USB / external drive into
    data/raw/ without any processing.
    """
    source_dir = Path(source_dir)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for item in source_dir.iterdir():
        if item.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            dest = raw_dir / item.name
            shutil.copy2(item, dest)
            copied.append(dest)
            logger.info("Copied: %s", item.name)

    return copied
