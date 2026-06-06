# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
scripts/map_training_data.py

Copies clips from training-data/<category>/ into data/labeled/<label>/
according to the label taxonomy used by the VideoMAE classifier.

Files are renamed to <original_stem>_<label>.mp4 so the label is
immediately visible in the filename.

Usage
-----
    python scripts/map_training_data.py [--dry_run] [--clear_dst] [--max_other N]

Flags
-----
    --dry_run       Print what would be copied without touching any files.
    --clear_dst     Wipe data/labeled before copying (fresh start).
    --max_other N   Cap the number of files copied into the 'other' label (random sample).
    --src           Source root (default: training-data)
    --dst           Destination root (default: data/labeled)
"""

import argparse
import random
import shutil
from pathlib import Path

# ── Label mapping ─────────────────────────────────────────────────────────────
# Key   = subfolder name inside training-data/
# Value = target label (None = discard)
LABEL_MAP = {
    "goal":                  "goal",
    "close_goal":            "scoring_chance",
    "good_scoring_chance":   "scoring_chance",
    "scoring_chance":        "scoring_chance",
    "celebration":           "celebration",
    "replay":                "goal_replay",
    "hit_replay":            "other_replay",
    "closecall_replay":      "other_replay",
    "faceoff":               "faceoff",
    "faceoff_cutscene":      "faceoff_cutscene",
    "hit":                   "other",
    "coach_reactions":       "other_replay",
    "referee_cutscene":      "other_replay",
    "injury_cutscene":       "other_replay",
    "intro_cutscene":        "other_replay",
    "menu":                  "other",
    "misc_cutscene":         "other_replay",
    "other":                 "other",
    "ejection_cutscene":     "other_replay",
    "ea_transition":         "other",
    # Discarded
    "offensive_zone":        None,   # replaced by good_scoring_chance
    "transition":            None,   # sub-second clips — unreliable for VideoMAE
    "empty_net_cutscene":    None,   # 0 clips
    "combine_clips":         None,   # utility folder, not training data
    "output":                None,   # pipeline results, not training data
}

# Per-folder clip cap (applied before the global max_other cap).
# Use this to include only a fraction of a large folder.
FOLDER_CAP: dict = {}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def map_data(src_root: Path, dst_root: Path, dry_run: bool, max_other: int | None = None) -> None:
    # Collect all files per label first so we can apply max_other cap
    plan: dict[str, list[tuple[Path, str, str]]] = {}  # label -> [(src_file, stem, dst_name)]

    skipped_folders: list[str] = []

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
            files = [f for f in src_folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
            print(f"  [DISCARD]  {folder_name}/ ({len(files)} file(s))")
            continue

        files = [f for f in src_folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
        if not files:
            print(f"  [EMPTY]    {folder_name}/ -> {label}/ (0 files)")
            continue

        # Apply per-folder cap before adding to plan
        if folder_name in FOLDER_CAP and len(files) > FOLDER_CAP[folder_name]:
            random.seed(42)
            files = random.sample(files, FOLDER_CAP[folder_name])
            print(f"  [PLAN]     {folder_name}/ -> {label}/ ({FOLDER_CAP[folder_name]} file(s), capped from {len(list(src_folder.iterdir()))})")
        else:
            print(f"  [PLAN]     {folder_name}/ -> {label}/ ({len(files)} file(s))")

        for src_file in sorted(files):
            stem = src_file.stem
            # Strip any existing label suffix to avoid doubling up
            for existing_label in LABEL_MAP.values():
                if existing_label and stem.endswith(f"_{existing_label}"):
                    stem = stem[: -(len(existing_label) + 1)]
                    break
            dst_name = f"{stem}_{label}{src_file.suffix.lower()}"
            plan.setdefault(label, []).append((src_file, stem, dst_name))

    # Apply cap to 'other' and 'other_replay'
    for cap_label in ("other", "other_replay"):
        if max_other is not None and cap_label in plan:
            original = len(plan[cap_label])
            if original > max_other:
                random.seed(42)
                plan[cap_label] = random.sample(plan[cap_label], max_other)
                print(f"\n  [CAP]  {cap_label}: {original} -> {max_other} (random sample, seed=42)")

    # Execute copies
    counts: dict[str, int] = {}
    collision_renames = 0

    for label, entries in plan.items():
        dst_folder = dst_root / label
        if not dry_run:
            dst_folder.mkdir(parents=True, exist_ok=True)

        for src_file, stem, dst_name in entries:
            dst_file = dst_folder / dst_name

            if dst_file.exists() and not dry_run:
                dst_name = f"{src_file.parent.name}__{dst_name}"
                dst_file = dst_folder / dst_name
                collision_renames += 1
                print(f"    [COLLISION] renamed to {dst_name}")

            if dry_run:
                print(f"    {src_file.name}  ->  {dst_name}")
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
    parser = argparse.ArgumentParser(description="Map training-data -> data/labeled")
    parser.add_argument("--src", default="apps/shared/training-data", help="Source root folder")
    parser.add_argument("--dst", default="apps/shared/data/labeled", help="Destination root folder")
    parser.add_argument("--dry_run", action="store_true", help="Print plan without copying")
    parser.add_argument("--clear_dst", action="store_true", help="Wipe dst before copying")
    parser.add_argument("--max_other", type=int, default=None, help="Cap files copied into 'other'")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"ERROR: source folder not found: {src_root}")
        return

    if args.clear_dst and dst_root.exists() and not args.dry_run:
        print(f"Clearing {dst_root.resolve()} ...")
        shutil.rmtree(dst_root)

    print(f"Source : {src_root.resolve()}")
    print(f"Dest   : {dst_root.resolve()}")
    print(f"Mode   : {'DRY RUN' if args.dry_run else 'COPY'}")
    if args.max_other:
        print(f"Cap    : other <= {args.max_other}")
    print()

    map_data(src_root, dst_root, args.dry_run, max_other=args.max_other)


if __name__ == "__main__":
    main()
