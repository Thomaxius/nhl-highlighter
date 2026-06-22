#!/usr/bin/env python3
"""Interactive OAuth re-consent for NHL Highlighter.

Use this when a token is dead (``invalid_grant: Token has been expired or
revoked``) and the background refresher cannot recover it — i.e. the refresh
token itself was revoked. This runs the browser consent flow and writes a
fresh token file. On a headless server it prints a URL: open it on any
machine, approve, and the consent redirects back to this process.

Usage (from the repo root):
    python apps/auth/reauth.py poller     # re-consent the read-only poller token
    python apps/auth/reauth.py uploader   # re-consent the upload token
    python apps/auth/reauth.py both       # re-consent both (default)

Run as the same user that owns config/ (so file permissions match).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

APP_DIR = Path(os.environ.get("APP_DIR", "/opt/nhl-highlighter"))
CONFIG_DIR = APP_DIR / "config"
CLIENT_SECRETS = Path(
    os.environ.get("OAUTH_CLIENT_SECRETS", str(CONFIG_DIR / "client_secrets.json"))
)
REAUTH_FLAG = CONFIG_DIR / "reauth_required.flag"


@dataclass(frozen=True)
class TokenSpec:
    name: str
    token_file: Path
    scopes: tuple[str, ...]
    port: int


SPECS: dict[str, TokenSpec] = {
    "poller": TokenSpec(
        "poller",
        Path(os.environ.get("OAUTH_TOKEN_FILE", str(CONFIG_DIR / "token.json"))),
        ("https://www.googleapis.com/auth/youtube.readonly",),
        8080,
    ),
    "uploader": TokenSpec(
        "uploader",
        CONFIG_DIR / "youtube_token.json",
        ("https://www.googleapis.com/auth/youtube.upload",),
        8081,
    ),
}


def reauth_one(spec: TokenSpec) -> None:
    if not CLIENT_SECRETS.exists():
        print(f"ERROR: OAuth client secrets not found at {CLIENT_SECRETS}", file=sys.stderr)
        print("Download client_secrets.json (APIs & Services → Credentials) and place it there.",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Re-authenticating: {spec.name}  (scope: {', '.join(spec.scopes)}) ===")
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), list(spec.scopes))
    # open_browser=False prints a URL to visit; the redirect returns to this port.
    # access_type=offline + prompt=consent guarantees a fresh refresh token.
    creds = flow.run_local_server(
        port=spec.port, open_browser=False, access_type="offline", prompt="consent",
    )
    spec.token_file.parent.mkdir(parents=True, exist_ok=True)
    spec.token_file.write_text(creds.to_json())
    spec.token_file.chmod(0o600)
    print(f"✓ Saved fresh token: {spec.token_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive OAuth re-consent.")
    parser.add_argument(
        "targets", nargs="*", default=["both"],
        help="Which token(s) to re-auth: poller, uploader, or both (default).",
    )
    args = parser.parse_args()

    targets = args.targets or ["both"]
    if "both" in targets:
        chosen = list(SPECS.values())
    else:
        unknown = [t for t in targets if t not in SPECS]
        if unknown:
            print(f"ERROR: unknown target(s): {', '.join(unknown)}. "
                  f"Choose from: poller, uploader, both.", file=sys.stderr)
            return 2
        chosen = [SPECS[t] for t in targets]

    for spec in chosen:
        reauth_one(spec)

    # Clear the sentinel the background refresher may have written.
    if REAUTH_FLAG.exists():
        REAUTH_FLAG.unlink()
        print(f"Cleared reauth sentinel: {REAUTH_FLAG}")

    print("\nDone. The background refresher will keep these tokens warm from here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
