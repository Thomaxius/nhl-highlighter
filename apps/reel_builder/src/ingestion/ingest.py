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
        # If the file is already normalised (stem ends with _norm), skip re-encoding
        # and treat it as the output directly — UNLESS it's still HDR, in which
        # case "_norm" is a misnomer (the detectors need SDR) and we tone-map it.
        if clip.stem.endswith("_norm"):
            if _is_hdr(clip):
                out_path = output_dir / (clip.stem + "_sdr.mp4")
                logger.info("'%s' is tagged _norm but is HDR — tone-mapping to SDR: %s",
                            clip.name, out_path.name)
                if out_path.exists() and out_path.stat().st_size > 10_000 and not overwrite:
                    processed.append(out_path)
                elif _ffmpeg_normalise(clip, out_path):
                    processed.append(out_path)
                continue
            logger.info("Already normalised, using as-is: %s", clip.name)
            processed.append(clip)
            continue

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


def _is_hdr(src: Path) -> bool:
    """True if the source uses an HDR transfer function (PQ or HLG).

    PS5 captures are often 10-bit HDR (smpte2084/bt2020); YouTube downloads are
    8-bit SDR (bt709). The detectors and OCR are calibrated on SDR, so HDR must
    be tone-mapped during normalisation — see _ffmpeg_normalise.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True,
        ).stdout.strip().lower()
    except Exception:
        return False
    return out in {"smpte2084", "arib-std-b67"}


def _ffmpeg_normalise(src: Path, dst: Path) -> bool:
    """Run FFmpeg to normalise a single clip to 1080p30 8-bit SDR (bt709)."""
    base_vf = (
        f"scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:-1:-1:color=black,fps={TARGET_FPS}"
    )
    out_color: list[str] = []
    if _is_hdr(src):
        # Tone-map HDR (PQ/HLG, bt2020) → SDR bt709 so the colour/brightness-
        # calibrated detectors (banner, stars, stats OCR) see what they expect.
        # Needs ffmpeg built with zscale (libzimg).
        logger.info("HDR source detected — tone-mapping to SDR bt709: %s", src.name)
        tonemap = (
            "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
            "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
        )
        vf = f"{tonemap},{base_vf}"
        out_color = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    else:
        vf = base_vf

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", TARGET_CODEC,
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",          # force 8-bit output (no-op for 8-bit SDR input)
        *out_color,
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
