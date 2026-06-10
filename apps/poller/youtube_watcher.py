#!/usr/bin/env python3
"""
youtube_watcher.py

Polls a YouTube channel for new uploads, downloads them with yt-dlp,
runs the NHL highlighter pipeline, and re-uploads the highlight reel.

Environment variables (from config/watcher.env):
    YOUTUBE_CHANNEL_ID      Your channel ID (required)
    APP_DIR                 Path to the app root (default: /opt/nhl-highlighter)
    POLL_INTERVAL_SECS      Seconds between polls (default: 900)
    MIN_VIDEO_DURATION_SECS Skip videos shorter than this (default: 1800)
    UPLOAD_PRIVACY          unlisted | public | private (default: unlisted)
    OAUTH_CLIENT_SECRETS    Path to OAuth 2.0 client_secrets.json downloaded from
                            Google Cloud Console (default: <APP_DIR>/config/client_secrets.json)
    OAUTH_TOKEN_FILE        Path where the refreshable OAuth token is cached
                            (default: <APP_DIR>/config/token.json)

First-run OAuth flow:
    Run this script interactively once to complete the browser-based OAuth consent
    and cache a token. Subsequent runs (including on the server) will refresh
    the token automatically without user interaction.
"""

import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import google.auth.transport.requests
import google.oauth2.credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ── Configuration ─────────────────────────────────────────────────────────────
CHANNEL_ID = os.environ["YOUTUBE_CHANNEL_ID"]
APP_DIR = Path(os.environ.get("APP_DIR", "/opt/nhl-highlighter"))
OAUTH_CLIENT_SECRETS = Path(os.environ.get("OAUTH_CLIENT_SECRETS", str(APP_DIR / "config" / "client_secrets.json")))
OAUTH_TOKEN_FILE = Path(os.environ.get("OAUTH_TOKEN_FILE", str(APP_DIR / "config" / "token.json")))
OAUTH_SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECS", "900"))
MIN_DURATION = int(os.environ.get("MIN_VIDEO_DURATION_SECS", "1800"))
UPLOAD_PRIVACY = os.environ.get("UPLOAD_PRIVACY", "unlisted")
UPLOAD_DESCRIPTION_TEMPLATE = (
    "{title}\n\n"
    "This highlight reel was automatically generated with NHL Highlighter:\n"
    "https://github.com/Thomaxius/nhl-highlighter"
)

_is_windows = platform.system() == "Windows"
_venv_root = APP_DIR / ".venv"
VENV_PYTHON = _venv_root / ("Scripts\\python.exe" if _is_windows else "bin/python")
VENV_YTDLP  = _venv_root / ("Scripts\\yt-dlp.exe" if _is_windows else "bin/yt-dlp")
YTDLP_COOKIES_FILE = os.environ.get("YTDLP_COOKIES_FILE", "")
PIPELINE_SCRIPT = APP_DIR / "apps" / "reel_builder" / "pipeline.py"
UPLOAD_SCRIPT = APP_DIR / "apps" / "uploader" / "upload_youtube.py"
STATE_FILE = APP_DIR / "apps" / "shared" / "data" / "processed_videos.json"
RAW_DIR = APP_DIR / "apps" / "shared" / "data" / "raw"
EXPORTS_DIR = APP_DIR / "apps" / "shared" / "data" / "exports"

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


# ── YouTube OAuth service ─────────────────────────────────────────────────────

def build_youtube_service():
    creds = None
    if OAUTH_TOKEN_FILE.exists():
        creds = google.oauth2.credentials.Credentials.from_authorized_user_file(
            str(OAUTH_TOKEN_FILE), OAUTH_SCOPES
        )
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            if not OAUTH_CLIENT_SECRETS.exists():
                raise RuntimeError(
                    f"OAuth client secrets not found at {OAUTH_CLIENT_SECRETS}. "
                    "Download client_secrets.json from the Google Cloud Console "
                    "(APIs & Services > Credentials) and place it there, or set "
                    "OAUTH_CLIENT_SECRETS. Then run this script interactively once "
                    "to complete the browser-based consent flow."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(OAUTH_CLIENT_SECRETS), OAUTH_SCOPES
            )
            creds = flow.run_local_server(port=8080, open_browser=False)
        OAUTH_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        OAUTH_TOKEN_FILE.write_text(creds.to_json())
        logger.info("OAuth token saved to %s", OAUTH_TOKEN_FILE)
    return build("youtube", "v3", credentials=creds)


# ── YouTube API helpers ───────────────────────────────────────────────────────

def get_uploads_playlist_id(service) -> str:
    resp = service.channels().list(part="contentDetails", id=CHANNEL_ID).execute()
    items = resp.get("items", [])
    if not items:
        raise RuntimeError(f"No channel found for ID: {CHANNEL_ID}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_latest_videos(service, playlist_id: str, max_results: int = 10) -> list[dict]:
    resp = service.playlistItems().list(
        part="snippet",
        playlistId=playlist_id,
        maxResults=max_results,
    ).execute()
    return [
        {
            "video_id": item["snippet"]["resourceId"]["videoId"],
            "title": item["snippet"]["title"],
            "published_at": item["snippet"]["publishedAt"],
        }
        for item in resp.get("items", [])
    ]


def get_video_duration_secs(service, video_id: str) -> int | None:
    """Return duration in seconds, or None if the API returned no data (e.g. video
    still processing). Never returns 0 for a missing video — callers should treat
    None as "unknown, retry later"."""
    resp = service.videos().list(part="contentDetails", id=video_id).execute()
    items = resp.get("items", [])
    if not items:
        return None
    duration_str = items[0]["contentDetails"]["duration"]  # ISO 8601: PT2H5M30S
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


# ── Pipeline steps ────────────────────────────────────────────────────────────

def download_video(video_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    cmd = [
        str(VENV_YTDLP),
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "--output", output_template,
        "--merge-output-format", "mp4",
        "--js-runtimes", "node",
    ]
    if YTDLP_COOKIES_FILE:
        cmd += ["--cookies", YTDLP_COOKIES_FILE]
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")
    result = subprocess.run(cmd, capture_output=True, text=True)
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


def upload_highlight(video_file: Path, title: str) -> str | None:
    """Upload the reel and return the new YouTube video ID (or None if unparseable)."""
    # pipeline.py saves a description template with "__TITLE__" as a placeholder.
    # Use it if present; otherwise fall back to the built-in template.
    desc_file = video_file.with_suffix(".txt")
    if desc_file.exists():
        try:
            description = desc_file.read_text(encoding="utf-8").replace("__TITLE__", title, 1)
            logger.info("Using stats description from %s", desc_file.name)
        except Exception as exc:
            logger.warning("Could not read description file %s: %s — using default", desc_file, exc)
            description = UPLOAD_DESCRIPTION_TEMPLATE.format(title=title)
    else:
        description = UPLOAD_DESCRIPTION_TEMPLATE.format(title=title)
    result = subprocess.run(
        [
            str(VENV_PYTHON),
            str(UPLOAD_SCRIPT),
            "--file", str(video_file),
            "--title", title,
            "--description", description,
            "--privacy", UPLOAD_PRIVACY,
        ],
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
    )
    # Always surface uploader output so it appears in our log
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info("[uploader] %s", line)
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            logger.warning("[uploader] %s", line)
    if result.returncode != 0:
        raise RuntimeError(f"upload_youtube.py exited with code {result.returncode}")
    # Parse the video ID from the "Upload complete!" line so we can record it
    # in state and avoid re-processing our own uploads on the next poll.
    for line in result.stdout.splitlines():
        match = re.search(r"youtube\.com/watch\?v=([A-Za-z0-9_-]+)", line)
        if match:
            return match.group(1)
    return None


# ── Main processing loop ──────────────────────────────────────────────────────

def process_video(video_id: str, title: str, state: dict):
    raw_video_dir = RAW_DIR / video_id
    output_file = EXPORTS_DIR / f"{video_id}_highlights.mp4"
    processed_dir = APP_DIR / "apps" / "shared" / "data" / "processed"

    logger.info("=== Processing: %s  (%s) ===", title, video_id)
    state["processed"][video_id] = {"status": "downloading", "title": title}
    save_state(state)

    try:
        # Check for an already-normalised file first — if present we can skip
        # both the download and the FFmpeg re-encode entirely.
        norm_candidates = list(processed_dir.glob(f"{video_id}*_norm.mp4")) if processed_dir.exists() else []
        if norm_candidates:
            pipeline_input = norm_candidates[0]
            logger.info("Normalised file already present, skipping download + re-encode: %s", pipeline_input.name)
        else:
            # Fall back to checking for the raw downloaded file.
            existing_raw = (
                list(raw_video_dir.glob("*.mp4")) + list(raw_video_dir.glob("*.webm"))
                if raw_video_dir.exists() else []
            )
            if existing_raw:
                pipeline_input = existing_raw[0]
                logger.info("Raw file already present, skipping download: %s", pipeline_input.name)
            else:
                logger.info("Downloading...")
                pipeline_input = download_video(video_id, raw_video_dir)
                logger.info("Downloaded: %s", pipeline_input.name)

        state["processed"][video_id]["status"] = "processing"
        save_state(state)

        logger.info("Running pipeline (this may take a long time)...")
        run_pipeline(pipeline_input, output_file)
        logger.info("Pipeline finished: %s", output_file)

        state["processed"][video_id]["status"] = "uploading"
        save_state(state)

        logger.info("Uploading: %s", title)
        reel_video_id = upload_highlight(output_file, title)
        logger.info("Upload done.")

        state["processed"][video_id]["status"] = "done"
        if reel_video_id:
            # Track the uploaded reel's video ID so the poller skips it on
            # future polls instead of treating it as a new video to process.
            uploaded = state.setdefault("uploaded_reels", [])
            if reel_video_id not in uploaded:
                uploaded.append(reel_video_id)
            logger.info("Reel uploaded as https://youtube.com/watch?v=%s", reel_video_id)
        save_state(state)

        # Clean up large files now that the reel is uploaded.
        # Segments in processed/ are kept for model training.
        try:
            import shutil
            if raw_video_dir.exists():
                shutil.rmtree(raw_video_dir)
                logger.info("Deleted raw dir: %s", raw_video_dir)
            if output_file.exists():
                output_file.unlink()
                logger.info("Deleted export: %s", output_file)
        except Exception as cleanup_exc:
            logger.warning("Cleanup failed (non-fatal): %s", cleanup_exc)

        logger.info("=== Done: %s ===", video_id)

    except Exception as exc:
        logger.error("Failed processing %s: %s", video_id, exc, exc_info=True)
        state["processed"][video_id]["status"] = "error"
        state["processed"][video_id]["error"] = str(exc)
        save_state(state)


def main():
    logger.info("NHL Highlighter watcher starting. Channel: %s", CHANNEL_ID)

    try:
        service = build_youtube_service()
        playlist_id = get_uploads_playlist_id(service)
        logger.info("Uploads playlist: %s", playlist_id)
    except Exception as exc:
        logger.error("Failed to initialise YouTube service: %s", exc)
        sys.exit(1)

    while True:
        try:
            state = load_state()
            videos = get_latest_videos(service, playlist_id)
            logger.info("Checked uploads playlist — %d recent video(s) found", len(videos))

            for video in videos:
                video_id = video["video_id"]

                # Skip videos we uploaded ourselves (highlight reels)
                if video_id in state.get("uploaded_reels", []):
                    logger.debug("Skipping our own reel: %s", video_id)
                    continue

                existing = state["processed"].get(video_id, {})
                existing_status = existing.get("status")

                if existing_status in ("done", "uploading", "processing", "downloading", "skipped_short"):
                    continue
                # Retry errored videos on the next cycle
                if existing_status == "error":
                    logger.info("Retrying errored video: %s (%s)", video["title"], video_id)
                elif existing_status is not None:
                    continue

                duration = get_video_duration_secs(service, video_id)
                if duration is None:
                    logger.warning(
                        "Could not retrieve duration for %s — video may still be processing; will retry next poll",
                        video_id,
                    )
                    continue
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
