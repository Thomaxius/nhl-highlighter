#!/usr/bin/env python3
"""
process_local.py

Processes recordings dropped into apps/shared/data/raw/ end to end:

    pipeline (reel_builder)  ->  upload to YouTube  ->  record in processed_videos.json
                             ->  archive the source to raw/processed_local/
                             ->  delete the exported reel

This is the "automatic uploader" half of the old YouTube poller. pipeline.py only
builds the reel and a description template; nothing uploads it unless something
runs upload_youtube.py afterwards, and this is that something.

Usage (from the repo root, with the project's python):
    python apps/processor/process_local.py                  # every pending file in raw/
    python apps/processor/process_local.py --file game.mp4  # one specific file
    python apps/processor/process_local.py --force          # reprocess files already marked done
    python apps/processor/process_local.py --local          # pass --local to pipeline.py

Safe to run from cron: a lock stops overlapping runs, so a new invocation while a
long pipeline is still going exits immediately instead of starting a second one.

Environment variables:
    APP_DIR            Project root (default: the repo this file lives in)
    UPLOAD_PRIVACY     unlisted | public | private (default: unlisted)
    LOCAL_SETTLE_SECS  Ignore files modified within this window, they may still
                       be copying (default: 30)

State file: apps/shared/data/processed_videos.json, shaped {"processed": {id: {...}},
"uploaded_reels": [...]}. Status values: processing, uploading, upload_failed, done, error.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
APP_DIR = Path(os.environ.get("APP_DIR") or Path(__file__).resolve().parents[2])
RAW_DIR = APP_DIR / "apps" / "shared" / "data" / "raw"
EXPORTS_DIR = APP_DIR / "apps" / "shared" / "data" / "exports"
PROCESSED_DIR = APP_DIR / "apps" / "shared" / "data" / "processed"
STATE_FILE = APP_DIR / "apps" / "shared" / "data" / "processed_videos.json"
PIPELINE_SCRIPT = APP_DIR / "apps" / "reel_builder" / "pipeline.py"
UPLOAD_SCRIPT = APP_DIR / "apps" / "uploader" / "upload_youtube.py"
LOCK_FILE = APP_DIR / "apps" / "shared" / "data" / ".process_local.lock"

UPLOAD_PRIVACY = os.environ.get("UPLOAD_PRIVACY", "unlisted")
LOCAL_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
LOCAL_SETTLE_SECONDS = int(os.environ.get("LOCAL_SETTLE_SECS", "30"))
UPLOAD_DESCRIPTION_TEMPLATE = (
    "{title}\n\n"
    "This highlight reel was automatically generated with NHL Highlighter:\n"
    "https://github.com/Thomaxius/nhl-highlighter"
)

# Always the interpreter running this script, so cron / venv / Windows all just work.
PYTHON = sys.executable

(APP_DIR / "logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(APP_DIR / "logs" / "process_local.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("process_local")


# ── Single-run lock ───────────────────────────────────────────────────────────

class AlreadyRunning(Exception):
    pass


class RunLock:
    """OS-level lock held for the life of the process. The OS drops it if the
    process dies, so a crash can never leave a stale lock behind."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            self.fh = None
            raise AlreadyRunning()
        return self

    def __exit__(self, *exc):
        if self.fh is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.fh.seek(0)
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self.fh.close()


# ── State helpers ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state.setdefault("processed", {})
        return state
    return {"processed": {}}


def save_state(state: dict) -> None:
    # Write to a temp file and atomically rename it into place, so a kill mid-write
    # (Ctrl+C, OOM, reboot) can never leave a truncated, corrupt JSON file behind.
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def local_video_id(path: Path) -> str:
    """Stable synthetic id for a local file ("local-<stem>"), tracked in the same
    state file as the old YouTube ids so each file is processed once."""
    return "local-" + re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)


# ── Pipeline + upload ─────────────────────────────────────────────────────────

def run_pipeline(input_file: Path, output_file: Path, local: bool = False) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(PIPELINE_SCRIPT), "--file", str(input_file), "--output", str(output_file)]
    if local:
        cmd.append("--local")
    result = subprocess.run(cmd, cwd=str(APP_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"pipeline.py exited with code {result.returncode}")


def upload_highlight(video_file: Path, title: str) -> str | None:
    """Upload the reel and return the new YouTube video id (None if unparseable)."""
    # pipeline.py saves a description template with "__TITLE__" as a placeholder.
    desc_file = video_file.with_suffix(".txt")
    if desc_file.exists():
        try:
            description = desc_file.read_text(encoding="utf-8").replace("__TITLE__", title, 1)
            logger.info("Using stats description from %s", desc_file.name)
        except Exception as exc:
            logger.warning("Could not read description file %s: %s - using default", desc_file, exc)
            description = UPLOAD_DESCRIPTION_TEMPLATE.format(title=title)
    else:
        description = UPLOAD_DESCRIPTION_TEMPLATE.format(title=title)

    result = subprocess.run(
        [
            PYTHON, str(UPLOAD_SCRIPT),
            "--file", str(video_file),
            "--title", title,
            "--description", description,
            "--privacy", UPLOAD_PRIVACY,
        ],
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in (result.stdout or "").strip().splitlines():
        logger.info("[uploader] %s", line)
    for line in (result.stderr or "").strip().splitlines():
        logger.warning("[uploader] %s", line)
    if result.returncode != 0:
        raise RuntimeError(f"upload_youtube.py exited with code {result.returncode}")
    for line in (result.stdout or "").splitlines():
        match = re.search(r"youtube\.com/watch\?v=([A-Za-z0-9_-]+)", line)
        if match:
            return match.group(1)
    return None


def _cleanup_after_success(output_file: Path, local_input: Path | None) -> None:
    """Archive the source and delete the exported reel after a successful upload.
    Segments in processed/ are kept for model training."""
    try:
        # The source isn't re-downloadable, so move (never delete) it out of the
        # scan path so it isn't picked up again on the next run.
        if local_input is not None and local_input.exists():
            archive_dir = RAW_DIR / "processed_local"
            archive_dir.mkdir(parents=True, exist_ok=True)
            dest = archive_dir / local_input.name
            shutil.move(str(local_input), str(dest))
            logger.info("Archived processed local file -> %s", dest)
        if output_file.exists():
            output_file.unlink()
            logger.info("Deleted export: %s", output_file)
    except Exception as exc:
        logger.warning("Cleanup failed (non-fatal): %s", exc)


def finalize_upload(video_id: str, title: str, output_file: Path, state: dict,
                    local_input: Path | None = None) -> bool:
    """Upload a finished reel and finalize state. Returns True on success.

    On failure the entry is marked 'upload_failed' (not 'error') and the reel +
    description stay on disk with their paths in the state file, so the next run
    retries just the upload instead of redoing the multi-hour pipeline.
    """
    entry = state["processed"][video_id]
    entry["status"] = "uploading"
    entry["reel_path"] = str(output_file)
    desc_path = output_file.with_suffix(".txt")
    if desc_path.exists():
        entry["description_path"] = str(desc_path)
    save_state(state)

    try:
        logger.info("Uploading: %s", title)
        reel_video_id = upload_highlight(output_file, title)
        logger.info("Upload done.")
    except Exception as exc:
        logger.error("Upload failed for %s: %s - the next run retries just the upload", video_id, exc)
        entry["status"] = "upload_failed"
        entry["error"] = str(exc)
        save_state(state)
        return False

    entry["status"] = "done"
    entry.pop("error", None)
    if reel_video_id:
        uploaded = state.setdefault("uploaded_reels", [])
        if reel_video_id not in uploaded:
            uploaded.append(reel_video_id)
        logger.info("Reel uploaded as https://youtube.com/watch?v=%s", reel_video_id)
    save_state(state)

    _cleanup_after_success(output_file, local_input)
    logger.info("=== Done: %s ===", video_id)
    return True


def retry_upload(video_id: str, state: dict) -> bool:
    """Re-attempt only the upload for a reel whose pipeline already succeeded."""
    entry = state["processed"].get(video_id, {})
    reel_path = entry.get("reel_path")
    title = entry.get("title", video_id)
    if not reel_path or not Path(reel_path).exists():
        logger.warning("Cannot retry upload for %s - reel file missing (%s); will reprocess", video_id, reel_path)
        return False
    local_source = entry.get("local_source")
    local_input = Path(local_source) if local_source and Path(local_source).exists() else None
    logger.info("Retrying upload for %s: %s", video_id, reel_path)
    return finalize_upload(video_id, title, Path(reel_path), state, local_input)


def process_file(path: Path, state: dict, local: bool = False) -> bool:
    """Run the full pipeline + upload for one recording. Returns True on success."""
    video_id = local_video_id(path)
    title = path.stem
    output_file = EXPORTS_DIR / f"{video_id}_highlights.mp4"

    logger.info("=== Processing: %s  (%s) ===", title, video_id)
    state["processed"][video_id] = {"status": "processing", "title": title, "local_source": str(path)}
    save_state(state)

    try:
        # An already-normalised copy lets the pipeline skip the slow FFmpeg re-encode.
        norm = sorted(glob.glob(str(PROCESSED_DIR / (glob.escape(path.stem) + "*_norm.mp4"))))
        if norm:
            pipeline_input = Path(norm[0])
            logger.info("Normalised file already present, skipping re-encode: %s", pipeline_input.name)
        else:
            pipeline_input = path

        logger.info("Running pipeline (this may take a long time)...")
        run_pipeline(pipeline_input, output_file, local=local)
        logger.info("Pipeline finished: %s", output_file)

        # Upload is separate from the pipeline so an upload failure (expired token...)
        # is retried as upload-only next run.
        return finalize_upload(video_id, title, output_file, state, path)
    except Exception as exc:
        logger.error("Failed processing %s: %s", video_id, exc, exc_info=True)
        state["processed"][video_id]["status"] = "error"
        state["processed"][video_id]["error"] = str(exc)
        save_state(state)
        return False


# ── Selecting what to do ──────────────────────────────────────────────────────

def scan_local_files() -> list[Path]:
    """Top-level video files in raw/. Per-video subfolders and the processed_local
    archive are skipped. Files modified within the settle window may still be
    copying, so they wait for the next run."""
    if not RAW_DIR.exists():
        return []
    now = time.time()
    files: list[Path] = []
    for p in sorted(RAW_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in LOCAL_VIDEO_EXTS:
            continue
        if now - p.stat().st_mtime < LOCAL_SETTLE_SECONDS:
            logger.info("File still settling, will pick it up next run: %s", p.name)
            continue
        files.append(p)
    return files


def handle_file(path: Path, state: dict, force: bool, local: bool) -> bool | None:
    """Decide what to do for one file. Returns True/False for success/failure,
    or None if it was skipped. The caller holds the run lock, so any entry still
    marked 'processing' or 'uploading' is left over from an interrupted run."""
    video_id = local_video_id(path)
    status = state["processed"].get(video_id, {}).get("status")

    if status == "done" and not force:
        logger.info("Already done, skipping (use --force to redo): %s", path.name)
        return None
    if status in ("upload_failed", "uploading") and not force:
        reel = state["processed"][video_id].get("reel_path")
        if reel and Path(reel).exists():
            return retry_upload(video_id, state)   # pipeline already succeeded: upload only
        logger.warning("Reel for %s is gone (%s) - redoing it from scratch", path.name, reel)
    elif status == "processing":
        logger.warning("Found '%s' left in 'processing' by an interrupted run - redoing it", path.name)
    elif status == "error":
        logger.info("Retrying errored file: %s", path.name)

    return process_file(path, state, local=local)


def resolve_file(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p.resolve()
    if (RAW_DIR / arg).is_file():
        return (RAW_DIR / arg).resolve()
    raise SystemExit(f"File not found: {arg} (also looked in {RAW_DIR})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pipeline on recordings in raw/ and upload the reels.")
    parser.add_argument("--file", help="Process one specific file (path, or a name inside raw/)")
    parser.add_argument("--force", action="store_true", help="Reprocess files already marked done")
    parser.add_argument("--local", action="store_true", help="Pass --local (boosted parallelism) to pipeline.py")
    args = parser.parse_args()

    try:
        with RunLock(LOCK_FILE):
            state = load_state()
            files = [resolve_file(args.file)] if args.file else scan_local_files()
            if not files:
                logger.info("Nothing to do: no pending recordings in %s", RAW_DIR)
                return 0

            failures = 0
            for path in files:
                if handle_file(path, state, args.force, args.local) is False:
                    failures += 1
            if failures:
                logger.warning("%d file(s) failed - see the log above; the next run retries them.", failures)
            return 1 if failures else 0
    except AlreadyRunning:
        logger.info("Another process_local.py run is already in progress - exiting.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
