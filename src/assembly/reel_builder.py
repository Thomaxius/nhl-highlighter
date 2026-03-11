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
    highlight_labels = {"goal", "save", "hit", "fight", "celebration", "goal_replay", "other_replay"}
    filtered = [
        s for s in segments
        if s.get("label") in highlight_labels and s.get("confidence", 0) >= min_confidence
    ]

    # Build goal sequences: each goal pulls its chained celebration + replay along with it.
    # celebration and replay are ONLY included when chained to a banner-detected goal.
    groups: list[list[dict]] = []
    used_ids: set[int] = set()

    # Collect intro segments — always go first regardless of label/confidence
    intro_segs = sorted(
        [s for s in segments if s.get("intro")],
        key=lambda s: str(s["path"]),
    )
    for seg in intro_segs:
        used_ids.add(id(seg))

    # Collect game_end segments separately — they always go last, unscored
    game_end_segs = [s for s in segments if s.get("game_end")]
    for seg in game_end_segs:
        used_ids.add(id(seg))

    for i, seg in enumerate(segments):
        if id(seg) in used_ids or seg not in filtered:
            continue
        if seg.get("banner_detected"):
            # Start a goal group and collect chained follow-ups in order
            group = [seg]
            used_ids.add(id(seg))
            for j in range(i + 1, min(i + 10, len(segments))):
                follow = segments[j]
                if follow.get("chain_goal_idx") == i:
                    group.append(follow)
                    used_ids.add(id(follow))
            groups.append(group)
        elif seg.get("label") in {"celebration", "goal_replay", "other_replay"}:
            # Never include celebration/replay unless chained to a goal — skip
            used_ids.add(id(seg))
            continue
        else:
            # Non-goal highlight (save, hit etc.) — standalone
            groups.append([seg])
            used_ids.add(id(seg))

    # Keep groups in chronological order (original segment discovery order).
    # Prepend intro group at the very start; append game_end group at the end.
    if intro_segs:
        groups.insert(0, intro_segs)

    if game_end_segs:
        # Sort by path name to preserve chronological order.
        # (game_end_segs were already added to used_ids above to keep them
        # out of the goal-chain loop — no need to filter again here.)
        game_end_segs_ordered = sorted(
            game_end_segs,
            key=lambda s: str(s["path"]),
        )
        if game_end_segs_ordered:
            groups.append(game_end_segs_ordered)

    if not groups:
        logger.warning("No highlight segments passed the filter — reel not created.")
        return output_path

    # intro and game_end groups are outside the max_clips limit — always included.
    has_intro_group    = bool(intro_segs) and groups and groups[0][0].get("intro")
    has_game_end_group = bool(game_end_segs) and groups and groups[-1][0].get("game_end")
    head_groups        = groups[:1]  if has_intro_group    else []
    tail_groups        = groups[-1:] if has_game_end_group else []
    middle_groups      = groups[len(head_groups): len(groups) - len(tail_groups)]
    selected_groups    = head_groups + middle_groups[:max_clips] + tail_groups

    logger.info(
        "Building reel from %d groups (%d intro + %d highlights + %d game_end) …",
        len(selected_groups), len(head_groups), len(middle_groups[:max_clips]), len(tail_groups),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Step 1: Process each group — goal sequences become ONE rolling clip
        processed_clips: list[Path] = []
        for group_idx, group in enumerate(selected_groups):
            lead = group[0]

            if lead.get("banner_detected") and len(group) > 1:
                # ── Merge goal → celebration → replay into a single continuous clip ──
                merged = tmp / f"group_{group_idx:03d}_merged.mp4"
                _merge_goal_sequence(group, merged, tmp)
                src = merged
            elif lead.get("intro"):
                # ── Concat all intro segments, then trim to VS+6s … faceoff+7s ──
                concat_intro = tmp / f"group_{group_idx:03d}_intro_raw.mp4"
                paths = [Path(s["path"]).resolve() for s in group]
                concat_txt = tmp / f"intro_list_{group_idx:03d}.txt"
                concat_txt.write_text("\n".join(f"file '{p}'" for p in paths))
                r_concat = subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(concat_txt), "-c", "copy", str(concat_intro)],
                    capture_output=True, text=True,
                )
                if r_concat.returncode != 0 or not concat_intro.exists():
                    logger.warning(
                        "Intro concat failed (rc=%d): %s",
                        r_concat.returncode, r_concat.stderr[-300:],
                    )
                    src = Path(lead["path"])  # fallback: just the first segment
                else:
                    trim_start = lead.get("intro_clip_start_s", 0.0)
                    trim_end   = lead.get("intro_clip_end_s")
                    if trim_end is not None:
                        trimmed_intro = tmp / f"group_{group_idx:03d}_intro.mp4"
                        _ffmpeg_trim(concat_intro, trimmed_intro, trim_start, trim_end)
                        src = trimmed_intro if trimmed_intro.exists() else concat_intro
                    else:
                        src = concat_intro
            elif lead.get("game_end"):
                # ── Concatenate all game-end segments then trim 20s post-event ──
                if len(group) > 1:
                    raw_gameend = tmp / f"group_{group_idx:03d}_gameend_raw.mp4"
                    paths       = [Path(s["path"]).resolve() for s in group]
                    concat_txt  = tmp / f"gameend_list_{group_idx:03d}.txt"
                    concat_txt.write_text("\n".join(f"file '{p}'" for p in paths))
                    r = subprocess.run(
                        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                         "-i", str(concat_txt), "-c", "copy", str(raw_gameend)],
                        capture_output=True, text=True,
                    )
                    if r.returncode != 0 or not raw_gameend.exists():
                        logger.warning(
                            "game_end concat failed (rc=%d): %s",
                            r.returncode, r.stderr[-300:],
                        )
                        raw_gameend = Path(lead["path"]).resolve()  # fallback
                    base = raw_gameend
                else:
                    base = Path(lead["path"]).resolve()
                trim_start = lead.get("game_end_trim_start_s", 0.0)
                trim_end   = lead.get("game_end_trim_end_s")
                logger.info(
                    "game_end clip: %d segment(s), trim=[%.1fs, %.1fs] (lead=%s)",
                    len(group), trim_start, trim_end if trim_end is not None else -1,
                    Path(lead["path"]).name,
                )
                if trim_end is not None:
                    trimmed_ge = tmp / f"group_{group_idx:03d}_gameend.mp4"
                    _ffmpeg_trim(base, trimmed_ge, trim_start, trim_end)
                    src = trimmed_ge if trimmed_ge.exists() else base
                else:
                    src = base
            else:
                # ── Standalone clip (solo goal, save, hit, etc.) ──
                src = Path(lead["path"])
                if "trim_start_s" in lead and "trim_end_s" in lead:
                    trimmed = tmp / f"trimmed_{group_idx:03d}.mp4"
                    _ffmpeg_trim(src, trimmed, lead["trim_start_s"], lead["trim_end_s"])
                    src = trimmed

            dst = tmp / f"clip_{group_idx:03d}.mp4"
            if add_overlays:
                _burn_overlay(src, dst, label=lead["label"], confidence=lead["confidence"])
            else:
                import shutil as _shutil
                _shutil.copy2(src, dst)
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

def _merge_goal_sequence(group: list[dict], output: Path, tmp: Path) -> None:
    """
    Concatenate goal (trimmed from banner window start) + celebration + replay
    into a single seamless rolling clip, re-encoding each part to uniform
    codec / resolution before the concat.
    """
    import shutil

    parts: list[Path] = []
    for k, seg in enumerate(group):
        src = Path(seg["path"])
        if k == 0 and "trim_start_s" in seg:
            # Goal clip: trim from the pre-goal window start to end-of-file
            # (let it roll naturally into the goal moment and beyond)
            part = tmp / f"seq_raw_{k}_{output.stem}.mp4"
            _ffmpeg_trim_start(src, part, seg["trim_start_s"])
            parts.append(part)
        else:
            parts.append(src)

    if len(parts) == 1:
        shutil.copy2(parts[0], output)
        return

    # Re-encode each part to uniform 1080p / aac so stream-copy concat works
    reencoded: list[Path] = []
    for k, p in enumerate(parts):
        enc = tmp / f"seq_enc_{k}_{output.stem}.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", str(p),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-ar", "44100",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1,setsar=1",
            str(enc),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Re-encode failed for part %d, using original.", k)
            shutil.copy2(p, enc)
        reencoded.append(enc)

    concat_list = tmp / f"seq_concat_{output.stem}.txt"
    concat_list.write_text("\n".join(f"file '{str(p)}'" for p in reencoded))
    _ffmpeg_concat(concat_list, output)


def _ffmpeg_trim_start(src: Path, dst: Path, start_s: float) -> None:
    """Trim a clip from start_s to end-of-file (no encode — seek then copy)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", str(src),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Trim-start failed for %s, using original.", src.name)
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
        else:
            logger.error("Trim-start fallback failed — source does not exist: %s", src)


def _ffmpeg_trim(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    """Trim a clip to [start_s, end_s] using FFmpeg stream copy (fast, no re-encode)."""
    duration = end_s - start_s
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", str(src),
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Trim failed for %s, using original.", src.name)
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
        else:
            logger.error("Trim fallback failed — source does not exist: %s", src)


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
