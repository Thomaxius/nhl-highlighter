"""
scripts/map_training_data.py

Copies clips from training-data/<category>/ into data/labeled/<label>/
according to the label taxonomy used by the VideoMAE classifier.

Files are renamed to <original_stem>_<label>.mp4 so the label is
immediately visible in the filename.

Usage
-----
    python scripts/map_training_data.py [--dry_run]

Flags
-----
    --dry_run   Print what would be copied without touching any files.
    --src       Source root (default: training-data)
    --dst       Destination root (default: data/labeled)
"""

import argparse
import shutil
from pathlib import Path

# ── Label mapping ─────────────────────────────────────────────────────────────
# Key   = subfolder name inside training-data/
# Value = target label (None = discard)
LABEL_MAP = {
    "goal":                 "goal",
    "close_goal":           "scoring_chance",
    "offensive_zone":       "scoring_chance",
    "celebration":          "celebration",
    "replay":               "replay",
    "hit_replay":           "replay",
    "closecall_replay":     "replay",
    "faceoff":              "other",
    "faceoff_cutscene":     "other",
    "hit":                  "other",
    "coach_reactions":      "other",
    "referee_cutscene":     "other",
    "injury_cutscene":      "other",
    "intro_cutscene":       "other",
    "menu":                 "other",
    "misc_cutscene":        "other",
    "other":                "other",
    # Discarded
    "transition":           None,   # sub-second clips — unreliable for VideoMAE
    "empty_net_cutscene":   None,   # 0 clips
    "output":               None,   # pipeline results, not training data
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def map_data(src_root: Path, dst_root: Path, dry_run: bool) -> None:
    counts: dict[str, int] = {}
    skipped_folders: list[str] = []
    collision_renames: int = 0

    for src_folder in sorted(src_root.iterdir()):
        if not src_folder.is_dir():
            continue

        folder_name = src_folder.name

        if folder_name not in LABEL_MAP:
            print(f"  [UNKNOWN]  {folder_name}/ — not in LABEL_MAP, skipping")
            skipped_folders.append(folder_name)
            continue

        label = LABEL_MAP[folder_name]

        if label is None:
            files = list(src_folder.iterdir())
            print(f"  [DISCARD]  {folder_name}/ ({len(files)} file(s))")
            continue

        dst_folder = dst_root / label
        files = [f for f in src_folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]

        if not files:
            print(f"  [EMPTY]    {folder_name}/ → {label}/ (0 files)")
            continue

        print(f"  [COPY]     {folder_name}/ → {label}/ ({len(files)} file(s))")

        if not dry_run:
            dst_folder.mkdir(parents=True, exist_ok=True)

        for src_file in sorted(files):
            stem = src_file.stem

            # Strip any existing label suffix to avoid doubling up
            # e.g. "scene-005_scoring_chance" → "scene-005"
            for existing_label in LABEL_MAP.values():
                if existing_label and stem.endswith(f"_{existing_label}"):
                    stem = stem[: -(len(existing_label) + 1)]
                    break

            dst_name = f"{stem}_{label}{src_file.suffix.lower()}"
            dst_file = dst_folder / dst_name

            # Handle collisions — prefix with source folder name
            if dst_file.exists() and not dry_run:
                dst_name = f"{folder_name}__{stem}_{label}{src_file.suffix.lower()}"
                dst_file = dst_folder / dst_name
                collision_renames += 1
                print(f"    [COLLISION] renamed to {dst_name}")

            if dry_run:
                print(f"    {src_file.name}  →  {dst_name}")
            else:
                shutil.copy2(src_file, dst_file)

            counts[label] = counts.get(label, 0) + 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("── Summary " + ("(DRY RUN) " if dry_run else "") + "─" * 40)
    total = 0
    for label in sorted(counts):
        print(f"  {label:<20} {counts[label]:>4} file(s)")
        total += counts[label]
    print(f"  {'TOTAL':<20} {total:>4} file(s)")
    if collision_renames:
        print(f"  {collision_renames} collision(s) were renamed with source folder prefix")
    if skipped_folders:
        print(f"  Skipped unknown folders: {', '.join(skipped_folders)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Map training-data → data/labeled")
    parser.add_argument("--src", default="training-data", help="Source root folder")
    parser.add_argument("--dst", default="data/labeled", help="Destination root folder")
    parser.add_argument("--dry_run", action="store_true", help="Print plan without copying")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"ERROR: source folder not found: {src_root}")
        return

    print(f"Source : {src_root.resolve()}")
    print(f"Dest   : {dst_root.resolve()}")
    print(f"Mode   : {'DRY RUN' if args.dry_run else 'COPY'}")
    print()

    map_data(src_root, dst_root, args.dry_run)


if __name__ == "__main__":
    main()
