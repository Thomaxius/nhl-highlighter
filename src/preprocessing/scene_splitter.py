"""
preprocessing/scene_splitter.py

Uses PySceneDetect to split a normalised clip into individual scenes /
segments, then saves them to disk for labelling and model training.
"""

from pathlib import Path
import logging
import subprocess

logger = logging.getLogger(__name__)


def split_into_scenes(
    video_path: str | Path,
    output_dir: str | Path,
    threshold: float = 27.0,
    min_scene_len: int = 15,
) -> list[Path]:
    """
    Detect scene cuts in *video_path* and split into separate segment files.

    Args:
        video_path:    Path to a normalised .mp4 file.
        output_dir:    Directory to write segment files into.
        threshold:     Content-detection threshold (lower = more sensitive).
        min_scene_len: Minimum scene length in frames.

    Returns:
        List of paths to the saved segment files.
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    from scenedetect.video_splitter import split_video_ffmpeg

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )

    logger.info("Detecting scenes in %s …", video_path.name)
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    logger.info("Found %d scenes", len(scene_list))

    if not scene_list:
        return []

    split_video_ffmpeg(
        str(video_path),
        scene_list,
        output_dir=str(output_dir),
        output_file_template=f"{video_path.stem}-scene-$SCENE_NUMBER.mp4",
        show_progress=True,
    )

    segments = sorted(output_dir.glob(f"{video_path.stem}-scene-*.mp4"))
    return segments


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 1.0,
) -> list[Path]:
    """
    Extract frames from a video at *fps* frames-per-second using FFmpeg.
    Useful for building an image-level training dataset.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir) / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(output_dir / "frame_%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Frame extraction error:\n%s", result.stderr[-300:])
        return []

    return sorted(output_dir.glob("frame_*.jpg"))
