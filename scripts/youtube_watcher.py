#!/usr/bin/env python3
"""
youtube_watcher.py

Polls a YouTube channel for new uploads, downloads them with yt-dlp,
runs the NHL highlighter pipeline, and re-uploads the highlight reel.

Environment variables (from config/watcher.env):
    YOUTUBE_CHANNEL_ID      Your channel ID (required)
    YOUTUBE_API_KEY         YouTube Data API v3 key (required)
    APP_DIR                 Path to the app root (default: /opt/nhl-highlighter)
    POLL_INTERVAL_SECS      Seconds between polls (default: 900)
    MIN_VIDEO_DURATION_SECS Skip videos shorter than this (default: 1800)
    UPLOAD_PRIVACY          unlisted | public | private (default: unlisted)
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

# ── Configuration ─────────────────────────────────────────────────────────────
CHANNEL_ID = os.environ["YOUTUBE_CHANNEL_ID"]
API_KEY = os.environ["YOUTUBE_API_KEY"]
APP_DIR = Path(os.environ.get("APP_DIR", "/opt/nhl-highlighter"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECS", "900"))
MIN_DURATION = int(os.environ.get("MIN_VIDEO_DURATION_SECS", "1800"))
UPLOAD_PRIVACY = os.environ.get("UPLOAD_PRIVACY", "unlisted")

VENV_PYTHON = APP_DIR / "venv" / "bin" / "python"
PIPELINE_SCRIPT = APP_DIR / "pipeline.py"
UPLOAD_SCRIPT = APP_DIR / "scripts" / "upload_youtube.py"
STATE_FILE = APP_DIR / "data" / "processed_videos.json"
RAW_DIR = APP_DIR / "data" / "raw"
EXPORTS_DIR = APP_DIR / "data" / "exports"

# ── Logging ───────────────────────────────────────────────────────────────────
log_dir = APP_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "watcher.log"),
    ],
)
logger = logging.getLogger("watcher")


# ── State helpers ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed": {}}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── YouTube API helpers ───────────────────────────────────────────────────────

def get_uploads_playlist_id() -> str:
    resp = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={"part": "contentDetails", "id": CHANNEL_ID, "key": API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        raise RuntimeError(f"No channel found for ID: {CHANNEL_ID}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_latest_videos(playlist_id: str, max_results: int = 10) -> list[dict]:
    resp = requests.get(
        "https://www.googleapis.com/youtube/v3/playlistItems",
        params={
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": max_results,
            "key": API_KEY,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return [
        {
            "video_id": item["snippet"]["resourceId"]["videoId"],
            "title": item["snippet"]["title"],
            "published_at": item["snippet"]["publishedAt"],
        }
        for item in resp.json().get("items", [])
    ]


def get_video_duration_secs(video_id: str) -> int:
    resp = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part": "contentDetails", "id": video_id, "key": API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return 0
    duration_str = items[0]["contentDetails"]["duration"]  # ISO 8601: PT2H5M30S
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


# ── Pipeline steps ────────────────────────────────────────────────────────────

def download_video(video_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    result = subprocess.run(
        [
            "yt-dlp",
            "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "--output", output_template,
            "--merge-output-format", "mp4",
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr[-500:]}")
    downloaded = list(output_dir.glob(f"{video_id}.*"))
    if not downloaded:
        raise RuntimeError(f"Downloaded file not found for video {video_id}")
    return downloaded[0]


def run_pipeline(input_file: Path, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(VENV_PYTHON),
            str(PIPELINE_SCRIPT),
            "--file", str(input_file),
            "--output", str(output_file),
        ],
        cwd=str(APP_DIR),
    )
    if result.returncode != 0:
        raise RuntimeError(f"pipeline.py exited with code {result.returncode}")


def upload_highlight(video_file: Path, title: str):
    result = subprocess.run(
        [
            str(VENV_PYTHON),
            str(UPLOAD_SCRIPT),
            "--file", str(video_file),
            "--title", title,
            "--privacy", UPLOAD_PRIVACY,
        ],
        cwd=str(APP_DIR),
    )
    if result.returncode != 0:
        raise RuntimeError(f"upload_youtube.py exited with code {result.returncode}")


# ── Main processing loop ──────────────────────────────────────────────────────

def process_video(video_id: str, title: str, state: dict):
    raw_video_dir = RAW_DIR / video_id
    output_file = EXPORTS_DIR / f"{video_id}_highlights.mp4"

    logger.info("=== Processing: %s  (%s) ===", title, video_id)
    state["processed"][video_id] = {"status": "downloading", "title": title}
    save_state(state)

    try:
        logger.info("Downloading...")
        downloaded = download_video(video_id, raw_video_dir)
        logger.info("Downloaded: %s", downloaded.name)

        state["processed"][video_id]["status"] = "processing"
        save_state(state)

        logger.info("Running pipeline (this may take a long time)...")
        run_pipeline(downloaded, output_file)
        logger.info("Pipeline finished: %s", output_file)

        state["processed"][video_id]["status"] = "uploading"
        save_state(state)

        highlight_title = f"NHL 25 Highlights – {title}"
        logger.info("Uploading: %s", highlight_title)
        upload_highlight(output_file, highlight_title)
        logger.info("Upload done.")

        state["processed"][video_id]["status"] = "done"
        save_state(state)
        logger.info("=== Done: %s ===", video_id)

    except Exception as exc:
        logger.error("Failed processing %s: %s", video_id, exc, exc_info=True)
        state["processed"][video_id]["status"] = "error"
        state["processed"][video_id]["error"] = str(exc)
        save_state(state)


def main():
    logger.info("NHL Highlighter watcher starting. Channel: %s", CHANNEL_ID)

    try:
        playlist_id = get_uploads_playlist_id()
        logger.info("Uploads playlist: %s", playlist_id)
    except Exception as exc:
        logger.error("Failed to get uploads playlist: %s", exc)
        sys.exit(1)

    while True:
        try:
            state = load_state()
            videos = get_latest_videos(playlist_id)
            logger.info("Checked uploads playlist — %d recent video(s) found", len(videos))

            for video in videos:
                video_id = video["video_id"]
                existing = state["processed"].get(video_id, {})
                existing_status = existing.get("status")

                if existing_status in ("done", "uploading", "processing", "downloading", "skipped_short"):
                    continue
                # Retry errored videos on the next cycle
                if existing_status == "error":
                    logger.info("Retrying errored video: %s (%s)", video["title"], video_id)
                elif existing_status is not None:
                    continue

                duration = get_video_duration_secs(video_id)
                if duration < MIN_DURATION:
                    logger.info(
                        "Skipping short video %s (%ds < %ds minimum)",
                        video_id, duration, MIN_DURATION,
                    )
                    state["processed"][video_id] = {
                        "status": "skipped_short",
                        "title": video["title"],
                        "duration": duration,
                    }
                    save_state(state)
                    continue

                process_video(video_id, video["title"], state)

        except Exception as exc:
            logger.error("Watcher poll error: %s", exc, exc_info=True)

        logger.info("Sleeping %ds until next poll...", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
