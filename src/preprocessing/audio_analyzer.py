"""
preprocessing/audio_analyzer.py

Detects audio energy spikes (crowd roars, goal horns) as a secondary
highlight signal alongside video classification.
"""

from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_audio(video_path: str | Path, output_path: str | Path) -> Path:
    """Extract the audio track from a video file to a .wav file using FFmpeg."""
    import subprocess

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{result.stderr[-300:]}")
    return output_path


def detect_energy_spikes(
    audio_path: str | Path,
    frame_length: int = 2048,
    hop_length: int = 512,
    top_db: float = 15.0,
    min_duration_s: float = 1.0,
) -> list[dict]:
    """
    Detect sustained high-energy segments (crowd roar, goal horn) in an
    audio file.

    Returns a list of dicts: {"start_s": float, "end_s": float, "energy": float}
    """
    import librosa

    audio_path = Path(audio_path)
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.power_to_db(rms, ref=np.max)

    # Frames above threshold
    above = rms_db > (rms_db.max() - top_db)

    # Convert to time segments
    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )

    spikes: list[dict] = []
    in_spike = False
    start_t = 0.0

    for i, is_above in enumerate(above):
        if is_above and not in_spike:
            in_spike = True
            start_t = float(times[i])
        elif not is_above and in_spike:
            in_spike = False
            duration = float(times[i]) - start_t
            if duration >= min_duration_s:
                avg_energy = float(np.mean(rms[max(0, i - 10) : i]))
                spikes.append(
                    {"start_s": start_t, "end_s": float(times[i]), "energy": avg_energy}
                )

    logger.info("Detected %d energy spikes in %s", len(spikes), audio_path.name)
    return spikes


def score_segments_by_audio(
    segments: list[dict], spikes: list[dict], window_s: float = 3.0
) -> list[dict]:
    """
    Boost the highlight score of video segments that overlap with audio
    energy spikes.

    Args:
        segments: List of {"start_s", "end_s", "score", ...} dicts.
        spikes:   Output from detect_energy_spikes().
        window_s: Extra context window (seconds) around each spike.

    Returns:
        Updated segments list with boosted scores.
    """
    for seg in segments:
        for spike in spikes:
            overlap = (
                seg["start_s"] < spike["end_s"] + window_s
                and seg["end_s"] > spike["start_s"] - window_s
            )
            if overlap:
                seg["score"] = seg.get("score", 0.0) + spike["energy"] * 10
                break
    return segments
