#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

VIDEO_EXTS = {".webm", ".mp4", ".mkv", ".mov", ".avi", ".m4v"}

# Suffix at END of filename stem, requiring at least one segment:
# _0010-end
# _0115-0215
# _0250-0315;1135-1200
# _0250-0315;1135-end
TIMESTAMP_SUFFIX_RE = re.compile(
    r"_(?P<spec>(?:\d{3,4}-(?:\d{3,4}|end))(?:;(?:\d{3,4}-(?:\d{3,4}|end)))*)$",
    re.IGNORECASE,
)

SEGMENT_RE = re.compile(r"^(?P<start>\d{3,4})-(?P<end>\d{3,4}|end)$", re.IGNORECASE)

def mmss_to_seconds(mmss: str) -> int:
    if not mmss.isdigit() or not (3 <= len(mmss) <= 4):
        raise ValueError(f"Invalid time '{mmss}' (expected 3-4 digits, e.g. 010, 0010, 0115, 1135).")
    sec = int(mmss[-2:])
    mins = int(mmss[:-2])
    if sec >= 60:
        raise ValueError(f"Invalid time '{mmss}': seconds must be < 60.")
    return mins * 60 + sec

def parse_mark(stem: str) -> Tuple[bool, Optional[str], Optional[str], Optional[List[Tuple[int, Optional[int], str]]]]:
    """
    Returns:
      (is_marked, base_stem, spec, segments)
    is_marked True if:
      - stem ends with "_"  (copy whole file), OR
      - stem ends with "_<timestamp spec>" (cut segments)
    """
    if stem.endswith("_"):
        base_stem = stem[:-1]
        return True, base_stem, None, None

    m = TIMESTAMP_SUFFIX_RE.search(stem)
    if not m:
        return False, None, None, None

    spec = m.group("spec")
    base_stem = stem[: m.start()]  # everything before "_spec"

    segments: List[Tuple[int, Optional[int], str]] = []
    for seg in spec.split(";"):
        sm = SEGMENT_RE.match(seg.strip())
        if not sm:
            raise ValueError(f"Bad segment '{seg}'")

        start_s = mmss_to_seconds(sm.group("start"))
        end_part = sm.group("end").lower()
        end_s = None if end_part == "end" else mmss_to_seconds(end_part)

        if end_s is not None and end_s <= start_s:
            raise ValueError(f"Segment '{seg}' has end <= start")

        segments.append((start_s, end_s, seg))

    return True, base_stem, spec, segments

def run_ffmpeg_cut(src: Path, dst: Path, start_s: int, end_s: Optional[int], dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(start_s), "-i", str(src)]
    if end_s is not None:
        cmd += ["-t", str(end_s - start_s)]
    cmd += ["-c", "copy", str(dst)]

    if dry_run:
        print("    CMD:", " ".join(map(str, cmd)))
        return

    subprocess.run(cmd, check=True)

def copy_whole_file(src: Path, dst: Path, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"    COPY: {src} -> {dst}")
        return
    shutil.copy2(src, dst)

def main() -> None:
    ap = argparse.ArgumentParser(description="Cut/copy marked NHL clips into ./goals using ffmpeg.")
    ap.add_argument("folder", nargs="?", default=".", help="Folder to scan (default: current directory)")
    ap.add_argument("--goals-dir", default="goals", help="Output subfolder name (default: goals)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions/commands without executing")
    args = ap.parse_args()

    src_dir = Path(args.folder).resolve()
    out_dir = src_dir / args.goals_dir

    if not src_dir.is_dir():
        raise SystemExit(f"Not a directory: {src_dir}")

    video_files = [p for p in sorted(src_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]

    print(f"Scanning: {src_dir}")
    print(f"Found {len(video_files)} video files\n")

    marked_files = 0
    clips_created = 0
    files_copied = 0

    for f in video_files:
        # Extra guard (useful if later you change the input source)
        if not f.exists():
            print(f"[SKIP] {f} (does not exist)")
            continue

        try:
            is_marked, base_stem, spec, segments = parse_mark(f.stem)
        except ValueError as e:
            print(f"[SKIP] {f.name} (bad suffix: {e})")
            continue

        if not is_marked:
            print(f"[SKIP] {f.name} (not marked)")
            continue

        marked_files += 1

        # Case 1: ends with "_" -> copy whole file
        if segments is None:
            dst = out_dir / f"{base_stem}{f.suffix}"
            print(f"[COPY] {f.name} -> goals/{dst.name}")
            copy_whole_file(f, dst, args.dry_run)
            if not args.dry_run:
                files_copied += 1
            continue

        # Case 2: has timestamp spec -> cut segments
        print(f"[MARKED] {f.name}  suffix=_{spec}")
        for idx, (start_s, end_s, label) in enumerate(segments, start=1):
            out_name = f"{base_stem}__clip{idx:02d}_{label}{f.suffix}"
            dst = out_dir / out_name
            print(f"  [CUT] {label} -> goals/{dst.name}")
            run_ffmpeg_cut(f, dst, start_s, end_s, args.dry_run)
            if not args.dry_run:
                clips_created += 1

        print()

    print("Summary:")
    print(f"  Marked files processed: {marked_files}")
    print(f"  Whole files copied:     {files_copied}")
    print(f"  Clips created:          {clips_created}")
    print(f"  Output folder:          {out_dir.resolve()}")

if __name__ == "__main__":
    main()