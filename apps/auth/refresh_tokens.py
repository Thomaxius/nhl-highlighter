#!/usr/bin/env python3
"""Background OAuth token refresher for NHL Highlighter.

Keeps both YouTube tokens warm and persisted:

  * poller   — config/token.json          (scope: youtube.readonly)
  * uploader — config/youtube_token.json   (scope: youtube.upload)

For each token file it loads the stored credentials, refreshes the access
token using the long-lived refresh token, and writes the refreshed
credentials back to disk (mode 0600). This runs as a *separate process* from
the poller/uploader so the tokens are always fresh on disk — a freshly
restarted poller or a per-upload uploader never has to do a cold refresh, and
a dead refresh token is detected here (early) instead of mid-upload.

IMPORTANT: a background process can only refresh a *living* refresh token. If
the refresh token has been revoked/expired (``invalid_grant``), only an
interactive browser consent can recover it — run ``apps/auth/reauth.py``. To
stop this happening weekly, publish the OAuth app to "In production" in the
Google Cloud Console (Testing-mode refresh tokens are force-expired after 7 days).

Usage (from the repo root):
    python apps/auth/refresh_tokens.py               # refresh once and exit
    python apps/auth/refresh_tokens.py --interval 21600  # loop every 6h
    python apps/auth/refresh_tokens.py --check       # report status only

Exit code is 0 when every token is healthy, 1 when at least one needs reauth.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

APP_DIR = Path(os.environ.get("APP_DIR", "/opt/nhl-highlighter"))
CONFIG_DIR = APP_DIR / "config"
# Sentinel touched when any token needs interactive reauth; cleared when healthy.
REAUTH_FLAG = CONFIG_DIR / "reauth_required.flag"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  token-refresher  %(message)s",
)
logger = logging.getLogger("token-refresher")


@dataclass(frozen=True)
class TokenSpec:
    name: str
    token_file: Path
    scopes: tuple[str, ...]


TOKENS: list[TokenSpec] = [
    TokenSpec(
        name="poller",
        token_file=Path(os.environ.get("OAUTH_TOKEN_FILE", str(CONFIG_DIR / "token.json"))),
        scopes=("https://www.googleapis.com/auth/youtube.readonly",),
    ),
    TokenSpec(
        name="uploader",
        token_file=CONFIG_DIR / "youtube_token.json",
        scopes=("https://www.googleapis.com/auth/youtube.upload",),
    ),
]


def refresh_one(spec: TokenSpec, *, write: bool = True) -> bool:
    """Refresh a single token file. Returns True if the token is healthy
    afterwards, False if it needs interactive reauth (or is missing)."""
    if not spec.token_file.exists():
        logger.warning("[%s] no token file at %s — run reauth.py", spec.name, spec.token_file)
        return False

    try:
        creds = Credentials.from_authorized_user_file(str(spec.token_file), list(spec.scopes))
    except (ValueError, KeyError) as exc:
        logger.error("[%s] token file is malformed (%s) — run reauth.py", spec.name, exc)
        return False

    if creds.valid and not creds.expired:
        logger.info("[%s] token still valid — no refresh needed", spec.name)
        return True

    if not creds.refresh_token:
        logger.error("[%s] no refresh token stored — run reauth.py", spec.name)
        return False

    try:
        creds.refresh(Request())
    except RefreshError as exc:
        logger.error(
            "[%s] refresh failed (%s) — refresh token revoked/expired; run reauth.py",
            spec.name, exc,
        )
        return False

    if write:
        spec.token_file.parent.mkdir(parents=True, exist_ok=True)
        spec.token_file.write_text(creds.to_json())
        spec.token_file.chmod(0o600)
        logger.info("[%s] refreshed and saved (%s)", spec.name, spec.token_file)
    else:
        logger.info("[%s] refresh OK (check mode — not written)", spec.name)
    return True


def refresh_all(*, write: bool = True) -> bool:
    """Refresh all tokens. Returns True only if every token is healthy."""
    results = {spec.name: refresh_one(spec, write=write) for spec in TOKENS}
    all_healthy = all(results.values())

    if write:
        unhealthy = [n for n, ok in results.items() if not ok]
        if unhealthy:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            REAUTH_FLAG.write_text(
                "Reauth required for: " + ", ".join(unhealthy) + "\n"
                "Run:  python apps/auth/reauth.py " + (" ".join(unhealthy) if len(unhealthy) == 1 else "both") + "\n"
            )
            logger.error("Wrote reauth sentinel: %s", REAUTH_FLAG)
        elif REAUTH_FLAG.exists():
            REAUTH_FLAG.unlink()
            logger.info("All tokens healthy — cleared reauth sentinel")

    return all_healthy


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh NHL Highlighter YouTube OAuth tokens.")
    parser.add_argument(
        "--interval", type=int, default=0,
        help="If > 0, loop forever refreshing every N seconds (e.g. 21600 = 6h). "
             "Default 0 = refresh once and exit.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Report status only; never write tokens or the sentinel.",
    )
    args = parser.parse_args()

    if args.interval > 0:
        logger.info("Starting token refresher loop (every %ds)", args.interval)
        while True:
            try:
                refresh_all(write=not args.check)
            except Exception:  # never let the daemon die on a transient error
                logger.exception("Unexpected error during refresh cycle")
            time.sleep(args.interval)

    healthy = refresh_all(write=not args.check)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
