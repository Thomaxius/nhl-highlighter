"""
scripts/stitch_transitions.py

Stitches short transition clips into ~4s training samples and saves them to
training-data/ea_transition/ (mapped to 'other' in LABEL_MAP).

Usage
-----
    python scripts/stitch_transitions.py [--max N] [--src FOLDER] [--dst FOLDER]

Options
-------
    --max   Maximum number of stitched output clips (default: 100)
    --src   Source folder (default: training-data/transition)
    --dst   Destination folder (default: training-data/ea_transition)
    --seed  Random seed for reproducibility (default: 42)
"""

import argparse
import random
import subprocess
import tempfile
from pathlib import Path

BATCH_SIZE = 3          # clips per stitched output
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def get_duration(path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if n > 0 else 0.0


def stitch(clips: list[Path], output: Path) -> bool:
    """Concatenate clips with ffmpeg stream copy. Returns True on success."""
    lines = [f"file '{c.resolve().as_posix()}'" for c in clips]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines))
        concat_list = Path(f.name)

    result = subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", str(output), "-y"],
        capture_output=True,
    )
    concat_list.unlink(missing_ok=True)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max",  type=int, default=100)
    p.add_argument("--src",  default="apps/shared/training-data/transition")
    p.add_argument("--dst",  default="apps/shared/training-data/ea_transition")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    clips = sorted([f for f in src.iterdir() if f.suffix.lower() in VIDEO_EXTS])
    if not clips:
        print(f"No clips found in {src}")
        return

    random.seed(args.seed)
    random.shuffle(clips)

    # Split into batches of BATCH_SIZE
    batches = [clips[i:i + BATCH_SIZE] for i in range(0, len(clips), BATCH_SIZE)]
    # Drop last batch if it's a singleton
    batches = [b for b in batches if len(b) >= 2]
    batches = batches[:args.max]

    print(f"Source clips : {len(clips)}")
    print(f"Batches      : {len(batches)} (capped at {args.max})")
    print(f"Output folder: {dst}")
    print()

    ok = 0
    for i, batch in enumerate(batches):
        out = dst / f"transition_stitched_{i:04d}.mp4"
        if out.exists():
            print(f"  [{i+1}/{len(batches)}] Skip (exists): {out.name}")
            ok += 1
            continue
        success = stitch(batch, out)
        if success:
            dur = get_duration(out)
            print(f"  [{i+1}/{len(batches)}] OK  {out.name}  ({len(batch)} clips, {dur:.1f}s)")
            ok += 1
        else:
            print(f"  [{i+1}/{len(batches)}] FAIL {out.name}")

    print(f"\nDone. {ok}/{len(batches)} stitched clips saved to {dst}")


if __name__ == "__main__":
    main()
