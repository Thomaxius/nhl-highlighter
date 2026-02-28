"""
assembly/reel_builder.py

Takes a ranked list of highlight segments and assembles them into a
broadcast-style highlight reel with transitions, optional text overlays,
and a music bed — all via FFmpeg and MoviePy.
"""

from pathlib import Path
from typing import Optional
import subprocess
import logging
import tempfile

logger = logging.getLogger(__name__)

# Default intro/outro duration in seconds
FADE_DURATION = 0.5


def build_reel(
    segments: list[dict],
    output_path: str | Path,
    music_path: Optional[str | Path] = None,
    max_clips: int = 10,
    min_confidence: float = 0.55,
    add_overlays: bool = True,
) -> Path:
    """
    Assemble a highlight reel from classified segments.

    Args:
        segments:        List of classifier result dicts with "path", "label",
                         "confidence", and optionally "score".
        output_path:     Path for the final .mp4 reel.
        music_path:      Optional background music file (.mp3 / .wav).
        max_clips:       Maximum number of clips to include.
        min_confidence:  Minimum confidence to include a clip.
        add_overlays:    Whether to burn label text onto each clip.

    Returns:
        Path to the assembled reel.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter & rank
    highlight_labels = {"goal", "save", "hit", "fight"}
    filtered = [
        s for s in segments
        if s.get("label") in highlight_labels and s.get("confidence", 0) >= min_confidence
    ]
    ranked = sorted(filtered, key=lambda s: s.get("score", s.get("confidence", 0)), reverse=True)
    selected = ranked[:max_clips]

    if not selected:
        logger.warning("No highlight segments passed the filter — reel not created.")
        return output_path

    logger.info("Building reel from %d clips …", len(selected))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Step 1: Optionally add text overlays to each clip
        processed_clips: list[Path] = []
        for i, seg in enumerate(selected):
            src = Path(seg["path"])
            dst = tmp / f"clip_{i:03d}.mp4"
            if add_overlays:
                _burn_overlay(src, dst, label=seg["label"], confidence=seg["confidence"])
            else:
                dst = src  # Use original
            processed_clips.append(dst)

        # Step 2: Add fade-in / fade-out to each clip
        faded_clips: list[Path] = []
        for i, clip in enumerate(processed_clips):
            faded = tmp / f"faded_{i:03d}.mp4"
            _add_fades(clip, faded, fade_duration=FADE_DURATION)
            faded_clips.append(faded)

        # Step 3: Concatenate
        concat_list = tmp / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{str(c)}'" for c in faded_clips)
        )
        concat_out = tmp / "concat.mp4"
        _ffmpeg_concat(concat_list, concat_out)

        # Step 4: Mix in background music (optional)
        if music_path and Path(music_path).exists():
            _mix_music(concat_out, Path(music_path), output_path)
        else:
            import shutil
            shutil.copy2(concat_out, output_path)

    logger.info("Reel saved → %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _burn_overlay(src: Path, dst: Path, label: str, confidence: float) -> None:
    overlay_text = f"{label.upper()} ({confidence:.0%})"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf",
        (
            f"drawtext=text='{overlay_text}'"
            ":fontsize=36:fontcolor=white"
            ":x=30:y=h-th-30"
            ":box=1:boxcolor=black@0.5:boxborderw=8"
        ),
        "-codec:a", "copy",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Overlay failed for %s, using original.", src.name)
        import shutil
        shutil.copy2(src, dst)


def _add_fades(src: Path, dst: Path, fade_duration: float) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"fade=t=in:d={fade_duration},fade=t=out:st=99999:d={fade_duration}",
        "-af", f"afade=t=in:d={fade_duration},afade=t=out:st=99999:d={fade_duration}",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        import shutil
        shutil.copy2(src, dst)


def _ffmpeg_concat(concat_list: Path, output: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Concat failed:\n{result.stderr[-300:]}")


def _mix_music(video: Path, music: Path, output: Path) -> None:
    """Mix background music underneath the original audio."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(music),
        "-filter_complex",
        "[0:a]volume=1.0[orig];[1:a]volume=0.3[music];[orig][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-shortest",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Music mix failed:\n{result.stderr[-300:]}")
