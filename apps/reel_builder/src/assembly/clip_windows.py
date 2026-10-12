"""
assembly/clip_windows.py

Pure helpers for planning scoring-chance clip windows (no torch / cv2 imports, so the
classifier, the pipeline and the reel builder can all share them and tests stay fast).

All times are seconds into one scene file. A "window" is a dict:
    {"trim_start_s": float, "trim_end_s": float, "confidence": float, "primary": bool}
"""
from __future__ import annotations

from typing import Iterable


def cap_window(start_s: float, end_s: float, center_s: float, max_len_s: float) -> tuple[float, float]:
    """Shrink [start_s, end_s] to at most *max_len_s*, centred as close to *center_s*
    as the original bounds allow. Windows already short enough come back unchanged."""
    if end_s - start_s <= max_len_s:
        return start_s, end_s
    start = min(max(center_s - max_len_s / 2.0, start_s), end_s - max_len_s)
    return start, start + max_len_s


def merge_windows(windows: Iterable[dict], gap_s: float = 1.0) -> list[dict]:
    """Union windows that overlap or sit within *gap_s* of each other, so no footage is
    shown twice. Returns new dicts sorted by start; the merged confidence is the max, and
    "primary" stays set if any merged window was the primary one."""
    merged: list[dict] = []
    for w in sorted(windows, key=lambda w: (w["trim_start_s"], w["trim_end_s"])):
        if merged and w["trim_start_s"] <= merged[-1]["trim_end_s"] + gap_s:
            last = merged[-1]
            last["trim_end_s"] = max(last["trim_end_s"], w["trim_end_s"])
            last["confidence"] = max(last.get("confidence", 0.0), w.get("confidence", 0.0))
            last["primary"] = bool(last.get("primary") or w.get("primary"))
        else:
            merged.append({**w, "confidence": w.get("confidence", 0.0), "primary": bool(w.get("primary"))})
    return merged


def shot_windows(
    shot_times_s: Iterable[float],
    duration_s: float,
    pre_s: float,
    post_s: float,
    max_len_s: float,
    ignore_before_s: float = 0.0,
) -> list[tuple[float, float]]:
    """Clip windows anchored on detected shots: [shot - pre_s, shot + post_s] per shot.

    Shots whose windows overlap are folded into one continuous window, up to *max_len_s*;
    past that, the next window starts exactly where the previous one ended, so consecutive
    clips never repeat footage. Shot times at or before *ignore_before_s* are ignored: the
    scoreboard detector reports a shot that happened in the PREVIOUS scene as occurring at
    ~0s of the scene where it first read the new count, which says nothing about where
    the shot really is.
    """
    times = sorted(t for t in shot_times_s if ignore_before_s < t <= duration_s)
    windows: list[list[float]] = []
    for t in times:
        start, end = max(0.0, t - pre_s), min(duration_s, t + post_s)
        if windows and start <= windows[-1][1]:
            if end - windows[-1][0] <= max_len_s:
                windows[-1][1] = max(windows[-1][1], end)
                continue
            start = windows[-1][1]          # too long to merge: continue where the last ended
            if end - start < 1.0:
                continue
        windows.append([start, end])
    return [(s, e) for s, e in windows]
