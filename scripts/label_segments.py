"""
scripts/label_segments.py

Interactive terminal labeler — plays each unlabeled segment with
QuickLook (macOS) or mpv, then lets you press a key to assign its class.

Usage
-----
    python scripts/label_segments.py --segments_dir data/processed/segments \
                                      --labeled_dir  data/labeled

Controls
--------
    g → goal        c → celebration  r → replay
    t → transition  o → other
    SPACE → skip    q → quit
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

KEYMAP = {
    "g": "goal",
    "c": "celebration",
    "r": "replay",
    "t": "transition",
    "o": "other",
}


def get_keypress() -> str:
    import tty
    import termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def play_clip(path: Path) -> None:
    """Open clip for preview — uses mpv if available, otherwise QuickLook."""
    if shutil.which("mpv"):
        subprocess.Popen(
            ["mpv", "--loop=yes", "--ontop", "--geometry=800x450", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        # macOS QuickLook
        subprocess.Popen(["qlmanage", "-p", str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def kill_players() -> None:
    for name in ("mpv", "qlmanage"):
        subprocess.run(["pkill", "-x", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def label_segments(segments_dir: Path, labeled_dir: Path) -> None:
    # Gather unlabeled segments (not already in any labeled subfolder)
    # Strip trailing _{label} suffix so we match both old and new naming styles
    labeled_stems: set[str] = set()
    for p in labeled_dir.rglob("*.mp4"):
        stem = p.stem
        for lbl in KEYMAP.values():
            if stem.endswith(f"_{lbl}"):
                stem = stem[: -(len(lbl) + 1)]
                break
        labeled_stems.add(stem)

    segments = sorted(
        p for p in segments_dir.rglob("*.mp4")
        if p.stem not in labeled_stems
    )

    if not segments:
        print("No unlabeled segments found.")
        return

    print(f"\nFound {len(segments)} unlabeled segments.\n")
    print("Controls:")
    for key, label in KEYMAP.items():
        print(f"  [{key}] {label}")
    print("  [SPACE] skip    [q] quit\n")

    for i, seg in enumerate(segments):
        print(f"[{i+1}/{len(segments)}]  {seg.name}", end="  →  ", flush=True)
        kill_players()
        play_clip(seg)

        key = get_keypress()

        if key == "q":
            kill_players()
            print("\nQuit.")
            break
        elif key == " ":
            print("(skipped)")
            continue
        elif key in KEYMAP:
            label = KEYMAP[key]
            dest_dir = labeled_dir / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            new_name = f"{seg.stem}_{label}{seg.suffix}"
            shutil.copy2(seg, dest_dir / new_name)
            # Rename the source segment in-place so the segments folder reflects the label
            seg.rename(seg.parent / new_name)
            print(label)
        else:
            print("(unknown key — skipped)")

    kill_players()
    print("\nDone! Summary:")
    for label_dir in sorted(labeled_dir.iterdir()):
        if label_dir.is_dir():
            count = len(list(label_dir.glob("*.mp4")))
            print(f"  {label_dir.name:12s} {count:>4d} clips")


def main():
    p = argparse.ArgumentParser(description="Interactive segment labeler")
    p.add_argument("--segments_dir", default="data/processed/segments")
    p.add_argument("--labeled_dir", default="data/labeled")
    args = p.parse_args()

    label_segments(
        segments_dir=Path(args.segments_dir),
        labeled_dir=Path(args.labeled_dir),
    )


if __name__ == "__main__":
    main()
