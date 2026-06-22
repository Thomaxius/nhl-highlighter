#!/usr/bin/env bash
# Shared preflight checks for the poller and reel builder.
#
# Validates two prerequisites before kicking off a long-running job:
#   1. Free disk space on the data filesystem (clips/segments/exports land here).
#   2. OAuth tokens still refresh (no point running for an hour on a dead token).
#
# Sourced by run_poller.sh and run_reel_builder.sh, which call `preflight`
# after setting ROOT and PYTHON. Because the callers run with `set -e`, a
# non-zero return aborts the run.
#
# Tunables (env vars):
#   PREFLIGHT_MIN_FREE_KB   minimum free space in KiB (default 5 GiB)
#   PREFLIGHT_SKIP_TOKENS   set to 1 to skip the OAuth token check
#   PREFLIGHT_SKIP_DISK     set to 1 to skip the disk space check

# 5 GiB in 1-KiB blocks.
PREFLIGHT_MIN_FREE_KB="${PREFLIGHT_MIN_FREE_KB:-$((5 * 1024 * 1024))}"

preflight() {
    local root="${1:-${ROOT:-$PWD}}"
    local python="${2:-${PYTHON:-python}}"

    # ── 1. Disk space ───────────────────────────────────────────────────────
    if [ "${PREFLIGHT_SKIP_DISK:-0}" = "1" ]; then
        echo "Preflight: skipping disk check (PREFLIGHT_SKIP_DISK=1)."
    elif command -v df >/dev/null 2>&1; then
        # Check the filesystem holding the data dir; fall back to the repo root.
        local data_dir="$root/apps/shared/data"
        [ -d "$data_dir" ] || data_dir="$root"
        # -P = portable (one row per fs), -k = 1-KiB blocks; avail is column 4.
        local avail_kb
        avail_kb="$(df -Pk "$data_dir" 2>/dev/null | awk 'NR==2 {print $4}')"
        if [ -n "$avail_kb" ] && [ "$avail_kb" -lt "$PREFLIGHT_MIN_FREE_KB" ]; then
            echo "Preflight FAILED: only $((avail_kb / 1024 / 1024)) GiB free — need at least $((PREFLIGHT_MIN_FREE_KB / 1024 / 1024)) GiB." >&2
            df -h "$data_dir" >&2
            return 1
        fi
        echo "Preflight: disk OK (${avail_kb:-?} KiB / $((${avail_kb:-0} / 1024 / 1024)) GiB free)."
    else
        echo "Preflight: df not available — skipping disk space check." >&2
    fi

    # ── 2. OAuth tokens ─────────────────────────────────────────────────────
    if [ "${PREFLIGHT_SKIP_TOKENS:-0}" = "1" ]; then
        echo "Preflight: skipping token check (PREFLIGHT_SKIP_TOKENS=1)."
        return 0
    fi

    # refresh_tokens.py reads APP_DIR / OAUTH_TOKEN_FILE; default them to this
    # checkout so the check works whether or not the caller already exported them.
    export APP_DIR="${APP_DIR:-$root}"
    export OAUTH_CLIENT_SECRETS="${OAUTH_CLIENT_SECRETS:-$root/config/client_secrets.json}"
    export OAUTH_TOKEN_FILE="${OAUTH_TOKEN_FILE:-$root/config/token.json}"

    echo "Preflight: validating OAuth tokens …"
    # --check refreshes in-memory only (never writes); exit 0 = all healthy.
    if ! "$python" "$root/apps/auth/refresh_tokens.py" --check; then
        echo "Preflight FAILED: OAuth tokens need reauth — run: $python $root/apps/auth/reauth.py" >&2
        return 1
    fi
    echo "Preflight: tokens OK."
}
