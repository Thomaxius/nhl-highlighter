#!/usr/bin/env bash
#
# fetch_segments.sh — pull the server's logs, list the last 5 pipeline runs,
# let you pick one, and download that run's scene segments.
#
# Configuration lives in scripts/fetch_segments.env (committed with placeholders
# — edit it on each machine). Values already set in the environment still win,
# e.g.  NHL_SERVER=other@host ./scripts/fetch_segments.sh
#   NHL_SERVER       user@host of the pipeline server          (required)
#   SSH_KEY          path to the SSH private key               (optional; falls
#                    back to your ssh-agent / default identity if unset)
#   NHL_REMOTE_BASE  remote install dir   (default: /opt/nhl-highlighter)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root is one level up from scripts/ — no hardcoded local path.
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load settings from the env file next to this script (override path via FETCH_ENV).
FETCH_ENV="${FETCH_ENV:-$SCRIPT_DIR/fetch_segments.env}"
if [ -f "$FETCH_ENV" ]; then
    # shellcheck disable=SC1090
    source "$FETCH_ENV"
fi

SERVER="${NHL_SERVER:?NHL_SERVER is not set — edit $FETCH_ENV (or pass NHL_SERVER=user@host)}"
REMOTE_BASE="${NHL_REMOTE_BASE:-/opt/nhl-highlighter}"
LOCAL_BASE="$REPO_ROOT"
SEG_LOCAL="$LOCAL_BASE/apps/shared/data/segments"

# Pass -i only when a key is configured; otherwise let ssh/scp use the agent.
KEY_ARGS=()
if [ -n "${SSH_KEY:-}" ]; then
    KEY_ARGS=(-i "$SSH_KEY")
fi

# ── 1. Optionally fetch the latest logs ─────────────────────────────────────
read -r -p "Fetch latest logs from $SERVER first? [Y/n] " fetch
case "$fetch" in
    [Nn]*) echo "Skipping log fetch — using existing local logs." ;;
    *)     echo "==> Fetching logs from $SERVER …"
           scp -r ${KEY_ARGS[@]+"${KEY_ARGS[@]}"} "$SERVER:$REMOTE_BASE/logs" "$LOCAL_BASE/logs" ;;
esac

# ── 2. Find the 5 most recent pipeline logs (filename timestamp = run time) ──
# Sort by the timestamp in the FILENAME (not the full path), and de-duplicate
# by basename so a log present in both logs/ and logs/logs/ is counted once.
LOGS=()
while IFS= read -r line; do
    LOGS+=("$line")
done < <(find "$LOCAL_BASE/logs" -name 'pipeline_*.log' 2>/dev/null \
         | awk -F/ '{print $NF "\t" $0}' \
         | sort -r \
         | awk -F'\t' '!seen[$1]++ {print $2}' \
         | head -5)

if [ "${#LOGS[@]}" -eq 0 ]; then
    echo "No pipeline logs found under $LOCAL_BASE/logs" >&2
    exit 1
fi

# Extract the normalized-file stem (== the segment folder name) from a log.
stem_of() {
    local log="$1" s=""
    # Primary: the scene-splitter announces the exact file it splits.
    s=$(sed -n 's/.*Detecting scenes in \(.*\)\.mp4.*/\1/p' "$log" | head -1)
    # Fallbacks: the normalise / use-as-is lines.
    [ -z "$s" ] && s=$(sed -n 's/.*Normalising: .* → \(.*\)\.mp4.*/\1/p' "$log" | head -1)
    [ -z "$s" ] && s=$(sed -n "s/.*using as-is: \(.*\)\.mp4.*/\1/p" "$log" | head -1)
    printf '%s' "$s"
}

# ── 3. Present the menu ─────────────────────────────────────────────────────
echo
echo "Last ${#LOGS[@]} runs (newest first):"
STEMS=()
i=0
for log in "${LOGS[@]}"; do
    stem="$(stem_of "$log")"
    STEMS+=("$stem")
    if grep -q "Pipeline completed" "$log" 2>/dev/null; then
        status="done"
    else
        status="incomplete/running"
    fi
    printf "  [%d] %s\n        target: %s   (%s)\n" \
        "$((i + 1))" "$(basename "$log")" "${stem:-<unknown>}" "$status"
    i=$((i + 1))
done

# ── 4. Prompt for a choice ──────────────────────────────────────────────────
echo
read -r -p "Download segments for which run? [1-${#LOGS[@]}] (q to quit) " choice
[ "$choice" = "q" ] && exit 0

case "$choice" in
    ''|*[!0-9]*) echo "Not a number." >&2; exit 1 ;;
esac
idx=$((choice - 1))
if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#LOGS[@]}" ]; then
    echo "Choice out of range." >&2
    exit 1
fi

STEM="${STEMS[$idx]}"
if [ -z "$STEM" ]; then
    echo "Could not determine the segment folder for that run." >&2
    exit 1
fi

# ── 5. Download that run's segments ─────────────────────────────────────────
REMOTE_SEG="$REMOTE_BASE/apps/shared/data/processed/segments/$STEM"
mkdir -p "$SEG_LOCAL"
echo
echo "==> Downloading segments for: $STEM"
# Modern scp uses the SFTP protocol and takes the remote path literally, so we
# pass the path with its spaces intact (double-quoted for the LOCAL shell only)
# — adding remote-shell quotes here would become part of the filename.
scp -r ${KEY_ARGS[@]+"${KEY_ARGS[@]}"} "$SERVER:$REMOTE_SEG" "$SEG_LOCAL/"
echo "Done → $SEG_LOCAL/$STEM"
