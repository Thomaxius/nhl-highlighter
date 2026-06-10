"""
detection/stats_detector.py — Detect and parse end-of-game stats screens.

Detection strategy
------------------
Two template images (stored in configs/) are matched against each frame:

  stats_tab_all_skaters_template.png / stats_tab_goalies_template.png —
      crops of the active tab label ("ALL SKATERS" / "GOALIES").  These are
      team-independent UI strings.  Whichever scores higher tells us which
      table is currently visible.  Score > 0.75 on either confirms we are on
      a stats screen.

  stats_back_sort_template.png   — the BACK / SORT button bar that only
      becomes fully bright once the table has finished rendering.
      Score > 0.80 means the table is interactive and ready to parse.

Pass 1  Sequential 2-fps scan (VideoCapture grab-loop, no subprocess).
        For every sampled frame: run three template matches.  A frame is a
        *candidate* when tab > 0.75 AND back > 0.80.

Pass 2  Group consecutive candidates (< 1.5 s apart) → take the middle frame
        of each group.  For that frame: OCR the header for the team name, then
        parse the table.

Parsing
-------
  PLAYER column  — per-cell OCR with brightness-adaptive preprocessing (dark
                   bg = sharpen + threshold; light bg = raw grayscale).
  Numeric columns — single bulk pytesseract.image_to_data call on the full
                   table crop; words assigned to cells by (x, y) centre.

Only the last N seconds of the video are scanned (SCAN_FROM_END_S).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Scan parameters ───────────────────────────────────────────────────────────
SCAN_FROM_END_S: float = 300.0   # only scan the last 5 minutes
SCAN_FPS:        float = 2.0     # sample rate during the scan pass

# ── Template match thresholds ─────────────────────────────────────────────────
BACK_THRESHOLD: float = 0.80   # back/sort bar: table is fully rendered
TAB_THRESHOLD:  float = 0.75   # tab label ("ALL SKATERS" / "GOALIES"): on stats screen

# ── Grouping ──────────────────────────────────────────────────────────────────
GROUP_GAP_S: float = 1.5   # candidates > this apart → different screen

# ── Column boundaries for ALL SKATERS table (1920×1080) ──────────────────────
SKATER_COLS: list[tuple[str, int, int]] = [
    ("PLAYER", 20,   445),
    ("MIN",    445,  540),
    ("G",      540,  615),
    ("A",      615,  695),
    ("PTS",    775,  855),
    ("+/-",    855,  940),
    ("S",      940, 1060),
    ("S%",    1060, 1195),
    ("PPT",   1195, 1315),
    ("PIM",   1315, 1440),
    ("HITS",  1440, 1560),
    ("PPG",   1560, 1655),
    ("SHG",   1655, 1755),
    ("FC",    1755, 1900),
]
SKATER_ROW_Y: list[tuple[int, int]] = [
    (228 + i * 57, 228 + (i + 1) * 57) for i in range(12)
]

# ── GOALIES columns (1920×1080) ───────────────────────────────────────────────
GOALIE_COLS: list[tuple[str, int, int]] = [
    ("PLAYER", 40,   445),
    ("MIN",    445,  535),
    ("SA",     535,  628),
    ("S",      628,  734),
    ("SV%",    734,  860),
    ("GA",     860, 1050),
    ("GAA",   1050, 1175),
    ("ENG",   1175, 1295),
    ("PIM",   1295, 1415),
    ("G",     1415, 1545),
    ("A",     1545, 1625),
    ("PTS",   1625, 1750),
]
GOALIE_ROW_Y: list[tuple[int, int]] = [
    (251 + i * 57, 251 + (i + 1) * 57) for i in range(6)
]

# ── Header region for team-name OCR ──────────────────────────────────────────
HEADER_REGION = (0, 0, 800, 145)


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StatsScreen:
    """Parsed result of one stats screen (one team, one tab)."""
    team:      str
    tab:       str        # "ALL SKATERS" or "GOALIES"
    timestamp: float      # seconds into the video
    rows:      list[dict[str, str]] = field(default_factory=list)

    def to_csv_lines(self) -> list[str]:
        if not self.rows:
            return []
        headers = list(self.rows[0].keys())
        lines = [",".join(headers)]
        for row in self.rows:
            lines.append(",".join(row.get(h, "") for h in headers))
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# Template helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_templates(configs_dir: Path) -> dict[str, np.ndarray]:
    """Load all stats-screen templates from configs_dir.  Missing files are
    logged as warnings (detection will be skipped for that template)."""
    names = [
        "stats_back_sort_template",
        "stats_tab_all_skaters_template",
        "stats_tab_goalies_template",
    ]
    templates: dict[str, np.ndarray] = {}
    for name in names:
        p = configs_dir / f"{name}.png"
        if not p.exists():
            logger.warning("Template missing: %s", p)
            continue
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("Could not read template: %s", p)
            continue
        templates[name] = img
        logger.debug("Loaded template %s  shape=%s", name, img.shape)
    return templates


def _match(frame: np.ndarray, tmpl: np.ndarray) -> float:
    """Normalised cross-correlation score (0–1)."""
    res = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


# ─────────────────────────────────────────────────────────────────────────────
# OCR helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_player_column(
    frame: np.ndarray,
    player_x0: int,
    player_x1: int,
    row_ys: list[tuple[int, int]],
) -> list[str]:
    """OCR the entire PLAYER column in one psm-4 call and return one name per row.

    psm 4 ("single column of variable-size text") reads the column top-to-bottom
    and produces one text line per table row — no y-position maths needed.
    Line index i → row index i.

    Returns a list of length len(row_ys).  Missing / garbled entries are empty
    strings; the caller decides when to stop.
    """
    try:
        import pytesseract
    except ImportError:
        return [""] * len(row_ys)

    y_min = row_ys[0][0]
    y_max = row_ys[-1][1]
    col_crop = frame[y_min:y_max, player_x0:player_x1]

    scale = 3
    big  = cv2.resize(col_crop,
                      (col_crop.shape[1] * scale, col_crop.shape[0] * scale),
                      interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    raw   = pytesseract.image_to_string(gray, config="--psm 4 --oem 1").strip()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Clean each line
    cleaned: list[str] = []
    for line in lines:
        tokens = line.split()
        # Discard lines that are clearly noise (only punctuation / pipe chars)
        meaningful = [t for t in tokens if any(c.isalpha() for c in t)]
        if not meaningful:
            cleaned.append("")
            continue
        # Fix "0." initial → "O."
        if meaningful[0] == "0.":
            meaningful[0] = "O."
        # Join remaining tokens but cap at 3 words (initial + last + possible suffix)
        cleaned.append(" ".join(meaningful[:3]))

    # Pad / truncate to exactly len(row_ys) entries
    result = cleaned[:len(row_ys)]
    while len(result) < len(row_ys):
        result.append("")
    return result


def _ocr_team_name(frame: np.ndarray) -> str:
    """Extract team name from 'PAUSE <TEAM> STATS' header via OCR."""
    try:
        import pytesseract
    except ImportError:
        return ""

    x0, y0, x1, y1 = HEADER_REGION
    crop = frame[y0:y1, x0:x1]
    big  = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(
        big, config="--psm 7 --oem 1"
    ).strip().upper()

    m = re.search(r"PAUSE\s+(.+?)\s+STATS", text)
    return m.group(1).strip() if m else ""


def _clean_numeric(val: str, col_name: str) -> str:
    """Clean OCR output for a numeric/time column.

    Common letter-for-digit confusions are fixed in-place.  If the result
    still contains letters it can't be a valid stat → return '-'.

    col_name is used to skip the cleaning for text-like fields (SV%, MIN).
    """
    if not val:
        return ""

    # Map of common OCR substitutions: letter → digit (applied globally).
    # Only the highest-confidence swaps — O/o↔0 and l/I↔1 are very reliable;
    # others (S↔5, B↔8, Z↔2) are too risky for small-cell stats.
    _CHAR_MAP = str.maketrans({
        "O": "0", "o": "0",    # letter O → zero
        "I": "1", "l": "1",    # l / I → one
    })

    # SV% is stored as a decimal like ".921" — handle separately
    if col_name == "SV%":
        v = val.replace("O", "0").replace("o", "0")
        if v.startswith("-"):
            v = "." + v[1:]
        elif v and v[0].isdigit():
            v = "." + v
        return v

    # MIN is "MM:SS" — fix O→0, I→1, then validate
    if col_name == "MIN":
        v = val.translate(_CHAR_MAP)
        if re.fullmatch(r"\d{1,2}:\d{2}", v):
            return v
        # Try to recover "MMSS" → "MM:SS"
        digits = re.sub(r"\D", "", v)
        if len(digits) == 4:
            return f"{digits[:2]}:{digits[2:]}"
        return "-"  # unrecognisable → mark as missing

    # For all other columns: translate then check only digits/minus/dot remain
    v = val.translate(_CHAR_MAP)
    if re.fullmatch(r"-?\d+(\.\d+)?", v):
        return v
    # Mixed garbage → '-'
    return "-"


def _bulk_ocr_numerics(
    frame: np.ndarray,
    cols: list[tuple[str, int, int]],
    row_ys: list[tuple[int, int]],
) -> list[dict[str, str]]:
    """Run a single image_to_data call on the table area and assign tokens to
    cells by their (x, y) centre position.  Returns a grid as a list of row
    dicts.  Only numeric columns are populated here; PLAYER is left empty."""
    try:
        import pytesseract
    except ImportError:
        return [{c: "" for c, *_ in cols} for _ in row_ys]

    numeric_cols = [(c, x0, x1) for c, x0, x1 in cols if c != "PLAYER"]
    if not numeric_cols:
        return [{c: "" for c, *_ in cols} for _ in row_ys]

    x_min = min(x0 for _, x0, _ in numeric_cols)
    x_max = max(x1 for _, _, x1 in numeric_cols)
    y_min = row_ys[0][0]
    y_max = row_ys[-1][1]
    crop  = frame[y_min:y_max, x_min:x_max]

    scale = 2
    big   = cv2.resize(crop,
                       (crop.shape[1] * scale, crop.shape[0] * scale),
                       interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray, config="--psm 6 --oem 1",
        output_type=pytesseract.Output.DICT,
    )

    col_names  = [c for c, *_ in cols]
    all_col_cx = [(x0 + x1) / 2 for _, x0, x1 in cols]
    row_cy     = [(y0 + y1) / 2 for y0, y1 in row_ys]

    grid: list[dict[str, list[str]]] = [
        {c: [] for c in col_names} for _ in row_ys
    ]

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < 20:
            continue
        wx = data["left"][i] / scale + x_min + data["width"][i]  / (2 * scale)
        wy = data["top"][i]  / scale + y_min + data["height"][i] / (2 * scale)

        ri = min(range(len(row_cy)),     key=lambda j: abs(row_cy[j]     - wy))
        ci = min(range(len(all_col_cx)), key=lambda j: abs(all_col_cx[j] - wx))

        col_name = col_names[ci]
        if col_name != "PLAYER":
            grid[ri][col_name].append(text)

    # Collapse token lists → strings and clean
    rows: list[dict[str, str]] = []
    for ri, cell_tokens in enumerate(grid):
        row: dict[str, str] = {}
        for col_name in col_names:
            if col_name == "PLAYER":
                row[col_name] = ""  # filled separately
                continue
            raw_val = cell_tokens[col_name][0] if cell_tokens[col_name] else ""
            row[col_name] = _clean_numeric(raw_val, col_name)
        rows.append(row)
    return rows


def _parse_table(
    frame: np.ndarray,
    cols: list[tuple[str, int, int]],
    row_ys: list[tuple[int, int]],
) -> list[dict[str, str]]:
    """Parse a stats table.

    Hybrid approach:
      • PLAYER column — one psm-4 image_to_string call on the full column strip.
        Line index i maps directly to row index i.  Some names may be garbled
        (~80 % accuracy expected); that's acceptable.
      • Numeric columns — single bulk image_to_data call (fast, accurate for
        digits / time codes).  Letter-for-digit OCR errors are cleaned up by
        _clean_numeric.

    Stops after 3 consecutive rows where both the name and all numeric stats
    are empty (end-of-roster sentinel).
    """
    col_names = [c for c, *_ in cols]
    player_x0 = next(x0 for c, x0, _ in cols if c == "PLAYER")
    player_x1 = next(x1 for c, _, x1 in cols if c == "PLAYER")

    # --- numeric pass (one OCR call for all numeric columns) -----------------
    grid = _bulk_ocr_numerics(frame, cols, row_ys)

    # --- player name pass (one call for the whole column) --------------------
    name_lines = _ocr_player_column(frame, player_x0, player_x1, row_ys)

    rows: list[dict[str, str]] = []
    consecutive_empty = 0

    for ri, _ in enumerate(row_ys):
        name = name_lines[ri]
        row  = grid[ri].copy()
        row["PLAYER"] = name

        # Re-order so PLAYER is first
        ordered = {"PLAYER": name}
        ordered.update({k: row[k] for k in col_names if k != "PLAYER"})

        numeric_vals = [ordered[k] for k in col_names if k != "PLAYER"]
        has_any_data = bool(name) or any(v and v != "-" for v in numeric_vals)

        if not has_any_data:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                break   # end of roster
            rows.append(ordered)
        else:
            consecutive_empty = 0
            rows.append(ordered)

    # Strip trailing all-empty rows (blank roster slots)
    while rows:
        last = rows[-1]
        vals = [last.get(c, "") for c in col_names]
        if not any(v and v != "-" for v in vals):
            rows.pop()
        else:
            break

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_stats_screens(
    video_path: str | Path,
    scan_from_end_s: float = SCAN_FROM_END_S,
    configs_dir: str | Path | None = None,
) -> list[StatsScreen]:
    """
    Scan the last *scan_from_end_s* seconds of *video_path* for stats screens.

    Returns a list of :class:`StatsScreen` objects — one per detected
    team + tab combination — in chronological order.

    Algorithm
    ---------
    Pass 1  Sequential grab-loop at SCAN_FPS (default 2 fps).  Three template
            matches per sampled frame; no OCR.  A frame is a *candidate* when:
              • the ALL SKATERS or GOALIES tab label scores above TAB_THRESHOLD
                (we are on a stats screen looking at one of those two tabs), AND
              • the BACK/SORT button bar scores above BACK_THRESHOLD
                (the table has fully rendered — not mid-animation).

    Pass 2  Group consecutive candidates (gap ≤ GROUP_GAP_S).  Take the
            middle frame of each group.  OCR the team name from the header,
            then parse the table with the hybrid PLAYER/numeric approach.

    Dedup   Same (team, tab) detected multiple times → keep the one with the
            most rows.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return []

    # Locate template directory
    if configs_dir is None:
        here = Path(__file__).resolve().parent
        for ancestor in [here.parent, here.parent.parent]:
            candidate = ancestor / "configs"
            if candidate.is_dir():
                configs_dir = candidate
                break
        else:
            configs_dir = here.parent / "configs"

    configs_dir = Path(configs_dir)
    templates   = _load_templates(configs_dir)

    required = {"stats_back_sort_template",
                "stats_tab_all_skaters_template", "stats_tab_goalies_template"}
    missing = required - set(templates)
    if missing:
        logger.error("Missing templates: %s — aborting stats detection", missing)
        return []

    t_back = templates["stats_back_sort_template"]
    t_sk   = templates["stats_tab_all_skaters_template"]
    t_go   = templates["stats_tab_goalies_template"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_s   = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps_video
    start_s   = max(0.0, total_s - scan_from_end_s)

    logger.info(
        "Stats scan: %.1f s – %.1f s  (%.1f s window, sampling at %.0f fps)",
        start_s, total_s, total_s - start_s, SCAN_FPS,
    )

    # ── Pass 1: sequential frame scan ────────────────────────────────────────
    every_n = max(1, int(round(fps_video / SCAN_FPS)))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

    # candidates: list of (timestamp_s, tab_type)
    candidates: list[tuple[float, str]] = []
    frame_idx = 0

    while True:
        # Skip every_n-1 frames without decoding
        for _ in range(every_n - 1):
            if not cap.grab():
                break

        ok, frame = cap.read()
        if not ok:
            break

        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Tab label must match (team-independent: "ALL SKATERS" / "GOALIES" text)
        s_sk = _match(frame, t_sk)
        s_go = _match(frame, t_go)
        if max(s_sk, s_go) < TAB_THRESHOLD:
            frame_idx += 1
            continue

        # BACK/SORT bar must be bright — table fully rendered, not mid-animation
        if _match(frame, t_back) < BACK_THRESHOLD:
            frame_idx += 1
            continue

        tab_type = "ALL SKATERS" if s_sk >= s_go else "GOALIES"
        logger.debug("  Candidate t=%.2f s  tab=%s  sk=%.3f  go=%.3f",
                     ts, tab_type, s_sk, s_go)
        candidates.append((ts, tab_type))
        frame_idx += 1

    cap.release()
    logger.info("  Pass 1: %d loaded candidates", len(candidates))

    if not candidates:
        return []

    # ── Group consecutive candidates ─────────────────────────────────────────
    groups: list[list[tuple[float, str]]] = []
    current: list[tuple[float, str]] = [candidates[0]]

    for prev, nxt in zip(candidates, candidates[1:]):
        if nxt[0] - prev[0] <= GROUP_GAP_S:
            current.append(nxt)
        else:
            groups.append(current)
            current = [nxt]
    groups.append(current)

    logger.info("  Grouped into %d screen(s)", len(groups))

    # ── Pass 2: parse the middle frame of each group ──────────────────────────
    cap2    = cv2.VideoCapture(str(video_path))
    screens: list[StatsScreen] = []

    for group in groups:
        mid_ts, tab_type = group[len(group) // 2]
        logger.info("  Parsing group: tab=%s  t=%.2f s  (%d frame(s) in group)",
                    tab_type, mid_ts, len(group))

        cap2.set(cv2.CAP_PROP_POS_MSEC, mid_ts * 1000.0)
        ok, frame = cap2.read()
        if not ok:
            logger.warning("  Could not seek to t=%.2f s", mid_ts)
            continue

        team_name = _ocr_team_name(frame)
        logger.info("  Team: %r", team_name)

        if tab_type == "GOALIES":
            rows = _parse_table(frame, GOALIE_COLS, GOALIE_ROW_Y)
        else:
            rows = _parse_table(frame, SKATER_COLS, SKATER_ROW_Y)

        screens.append(StatsScreen(
            team=team_name,
            tab=tab_type,
            timestamp=mid_ts,
            rows=rows,
        ))

    cap2.release()

    # ── Dedup: same (team, tab) → keep most rows ─────────────────────────────
    seen: dict[tuple[str, str], StatsScreen] = {}
    for s in screens:
        key  = (s.team, s.tab)
        prev = seen.get(key)
        if prev is None or len(s.rows) > len(prev.rows):
            seen[key] = s
    screens = list(seen.values())

    logger.info("Found %d stats screen(s).", len(screens))
    return screens


def save_stats_csv(screens: list[StatsScreen], output_dir: str | Path) -> list[Path]:
    """Write each StatsScreen to a CSV file.  Returns written paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for screen in screens:
        stem = f"{screen.team}_{screen.tab}".lower().replace(" ", "_")
        path = output_dir / f"{stem}.csv"
        lines = screen.to_csv_lines()
        if not lines:
            logger.warning("No data for %s/%s — skipping CSV", screen.team, screen.tab)
            continue
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("  Stats CSV → %s", path)
        written.append(path)

    return written


DEFAULT_DISCLAIMER = (
    "This highlight reel was automatically generated with NHL Highlighter:\n"
    "https://github.com/Thomaxius/nhl-highlighter"
)


# YouTube descriptions render in a proportional font, so space-aligned column
# tables can never line up. Instead each player becomes one self-describing
# line with inline labels — and rows whose OCR came out garbled (fragment
# names, missing or implausible numbers) are dropped rather than printed
# under the wrong header.

def _plausible_name(name: str) -> bool:
    """True if the OCR'd player name looks like a real name rather than a
    stray fragment (e.g. '[s.chow', 'es,')."""
    name = name.strip()
    return bool(name) and name[0].isalpha() and sum(c.isalpha() for c in name) >= 3


def _int_or_none(value: str | None) -> int | None:
    value = (value or "").strip()
    return int(value) if value.isdigit() else None


def _star_stats_compact(stats: str) -> str:
    """Reformat star_detector's 'GOALS: 2 | ASSISTS: 0 | HITS: 8' to the same
    compact style as the skater lines: '2 G, 0 A, 8 hits'."""
    m = re.fullmatch(r"GOALS: (\d+) \| ASSISTS: (\d+) \| HITS: (\d+)", stats.strip())
    if not m:
        return stats.strip()
    g, a, h = m.groups()
    return f"{g} G, {a} A, {h} hits"


def _skater_lines(rows: list[dict[str, str]]) -> list[str]:
    """One labeled line per skater: 'M. Beniers — 15:52 | 0 G, 2 A, 2 P | 3 hits'."""
    lines: list[str] = []
    for r in rows:
        name = r.get("PLAYER", "").strip().replace(",", ".")
        if not _plausible_name(name):
            continue
        g, a, pts = (_int_or_none(r.get(c)) for c in ("G", "A", "PTS"))
        # G + A must equal PTS — catches shifted cells and OCR misreads
        if g is None or a is None or pts is None or g + a != pts:
            continue
        toi = r.get("MIN", "").strip()
        if not re.fullmatch(r"\d{1,2}:\d{2}", toi):
            toi = "-"
        line = f"{name} — {toi} | {g} G, {a} A, {pts} P"
        hits = _int_or_none(r.get("HITS"))
        if hits is not None:
            line += f" | {hits} hits"
        lines.append(line)
    return lines


def _goalie_lines(rows: list[dict[str, str]]) -> list[str]:
    """One labeled line per goalie: 'J. Daccord — 37 shots, .857 SV%, 3 GA'."""
    lines: list[str] = []
    for r in rows:
        name = r.get("PLAYER", "").strip().replace(",", ".")
        if not _plausible_name(name):
            continue
        stats: list[str] = []
        sa = _int_or_none(r.get("SA"))
        if sa is not None:
            stats.append(f"{sa} shots")
        sv = (r.get("SV%") or "").strip()
        if re.fullmatch(r"[01]?\.\d{2,3}", sv):
            stats.append(f"{sv} SV%")
        ga = _int_or_none(r.get("GA"))
        if ga is not None:
            stats.append(f"{ga} GA")
        if not stats:
            continue
        lines.append(f"{name} — {', '.join(stats)}")
    return lines


def format_description(
    title: str,
    screens: list[StatsScreen],
    disclaimer: str = DEFAULT_DISCLAIMER,
    stars: list | None = None,
) -> str:
    """Build a YouTube video description with game stats tables and stars.

    Format::

        {title}

        ⭐ Stars of the Game:
        🥇 FIRST STAR: Player Name
        🥈 SECOND STAR: Player Name
        🥉 THIRD STAR: Player Name

        Stats:
        {team1} Skaters:
        ...

        {team1} Goalies:
        ...

        {team2} Skaters / Goalies...

        {disclaimer}

    Teams appear in the order they were detected.  Only the most meaningful
    columns are shown to stay within YouTube's 5 000-character limit.
    """
    _RANK_STARS = {"FIRST": "⭐⭐⭐", "SECOND": "⭐⭐", "THIRD": "⭐"}

    parts: list[str] = [title]

    # Stars of the game — right under the title
    if stars:
        rank_order = ["FIRST", "SECOND", "THIRD"]
        by_rank = {s.rank: s for s in stars}
        stars_lines: list[str] = []
        for rank in rank_order:
            star = by_rank.get(rank)
            if star is None:
                continue
            emoji = _RANK_STARS.get(rank, "⭐")
            name_part = star.player_name or "(unknown)"
            line = f"{emoji} {name_part}"
            stats = _star_stats_compact(getattr(star, "player_stats", ""))
            if stats:
                line += f" — {stats}"
            stars_lines.append(line)
        if stars_lines:
            parts += [""] + stars_lines

    # Group screens by team (preserving detection order)
    teams_seen: list[str] = []
    by_team: dict[str, dict[str, StatsScreen]] = {}
    for s in screens:
        if s.team not in by_team:
            teams_seen.append(s.team)
            by_team[s.team] = {}
        by_team[s.team][s.tab] = s

    if teams_seen:
        parts += ["", "Stats:"]

    for team in teams_seen:
        tabs = by_team[team]
        # Skaters first, then Goalies
        for tab_key, formatter, label in [
            ("ALL SKATERS", _skater_lines, "Skaters"),
            ("GOALIES",     _goalie_lines, "Goalies"),
        ]:
            screen = tabs.get(tab_key)
            if screen is None:
                continue
            lines = formatter(screen.rows)
            if not lines:
                continue
            parts.append(f"\n{team} {label}:")
            parts.append("\n".join(lines))

    parts += ["", disclaimer]
    return "\n".join(parts)
