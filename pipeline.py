"""
pipeline.py  —  End-to-end highlight reel generator.

Usage
-----
    python pipeline.py --input data/raw --output data/exports/reel.mp4

The pipeline runs:
    1. Ingest & normalise clips  (FFmpeg)
    2. Split into scenes         (PySceneDetect)
    3. Audio spike detection     (librosa)
    4. Video classification      (VideoMAE fine-tune)
    5. Score & rank segments
    6. Assemble highlight reel   (FFmpeg + MoviePy)
"""

import argparse
import logging
from pathlib import Path

from src.ingestion.ingest import ingest_clips
from src.preprocessing.scene_splitter import split_into_scenes
from src.preprocessing.audio_analyzer import extract_audio, detect_energy_spikes, score_segments_by_audio
from src.detection.classifier import HighlightClassifier
from src.assembly.reel_builder import build_reel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run_pipeline(
    input_dir: str,
    output_path: str,
    checkpoint: str = "models/checkpoints/videomae_nhl",
    music_path: str | None = None,
    max_clips: int = 10,
    min_confidence: float = 0.55,
) -> None:
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    processed_dir = Path("data/processed")
    segments_dir = Path("data/processed/segments")
    audio_dir = Path("data/processed/audio")

    # ── Step 1: Ingest & normalise ───────────────────────────────────────────
    logger.info("━━━  Step 1: Ingesting clips  ━━━")
    normalised = ingest_clips(input_dir, processed_dir)
    logger.info("Normalised %d clips.", len(normalised))

    # ── Step 2: Scene splitting ──────────────────────────────────────────────
    logger.info("━━━  Step 2: Splitting scenes  ━━━")
    all_segments: list[Path] = []
    for clip in normalised:
        segs = split_into_scenes(clip, segments_dir / clip.stem)
        all_segments.extend(segs)
    logger.info("Total segments: %d", len(all_segments))

    # ── Step 3: Audio analysis ───────────────────────────────────────────────
    logger.info("━━━  Step 3: Audio analysis  ━━━")
    all_spikes = []
    for clip in normalised:
        wav_path = audio_dir / (clip.stem + ".wav")
        try:
            extract_audio(clip, wav_path)
            spikes = detect_energy_spikes(wav_path)
            all_spikes.extend(spikes)
        except Exception as e:
            logger.warning("Audio analysis failed for %s: %s", clip.name, e)

    # ── Step 4: Video classification ─────────────────────────────────────────
    logger.info("━━━  Step 4: Classifying segments  ━━━")
    classifier = HighlightClassifier(checkpoint_path=checkpoint)
    results = classifier.classify_segments(all_segments)

    # ── Step 5: Score with audio boosts ──────────────────────────────────────
    logger.info("━━━  Step 5: Scoring  ━━━")
    for r in results:
        r["score"] = r["confidence"]

    # Attach rough timing from filename index (real app would parse FFmpeg metadata)
    results = score_segments_by_audio(results, all_spikes)

    # ── Step 6: Assemble reel ────────────────────────────────────────────────
    logger.info("━━━  Step 6: Building reel  ━━━")
    build_reel(
        segments=results,
        output_path=output_path,
        music_path=music_path,
        max_clips=max_clips,
        min_confidence=min_confidence,
    )
    logger.info("Done!  Reel → %s", output_path)


def main():
    p = argparse.ArgumentParser(description="NHL 25 Highlight Reel Generator")
    p.add_argument("--input", required=True, help="Directory of raw .mp4 clips from PS5")
    p.add_argument("--output", default="data/exports/reel.mp4", help="Output reel path")
    p.add_argument("--checkpoint", default="models/checkpoints/videomae_nhl")
    p.add_argument("--music", default=None, help="Optional background music file")
    p.add_argument("--max_clips", type=int, default=10)
    p.add_argument("--min_confidence", type=float, default=0.55)
    args = p.parse_args()

    run_pipeline(
        input_dir=args.input,
        output_path=args.output,
        checkpoint=args.checkpoint,
        music_path=args.music,
        max_clips=args.max_clips,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
