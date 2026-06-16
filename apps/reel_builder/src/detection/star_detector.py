"""
detection/star_detector.py — Detect end-of-game Three Stars screens.

Detection strategy
------------------
Template images (stored in configs/) are matched against the star-banner strip at
y=230–320 in each frame:

  star_third_star_template.png   — 1920×90 crop from confirmed frame at t≈27 s
  star_second_star_template.png  — 1920×90 crop from confirmed frame at t≈37 s
  star_first_star_template.png   — 1920×90 crop from confirmed frame at t≈52 s

  star_stats_label_template.png  — 22×360 crop of GOALS/ASSISTS/HITS header strip
                                   from y=475–497, x=550–910 at t≈27 s.
                                   Used to gate stats OCR until animation completes.

Two-phase algorithm
-------------------
Phase 1  Sequential 2-fps grab-loop. Records peak template-match score per rank.
         True star screens peak at ≥0.95.

Phase 2  Linear scan from (peak − 3 s) to (peak + 4 s) at ~6 fps (single seek,
         no random-access imprecision). For each frame passing the stats-animation
         gate, collect stats via two independent readers:
           • Per-digit: column brightness → 3 digit crops → PSM 8/10/13 (weight 2)
           • PSM 7 strip: multi-percentile full-strip OCR (weight 1)
         Final stats determined by weighted majority vote at each digit position.

OCR regions (1920×1080)
-----------------------
  Player first name  — y=635–688, x=90–300   CLAHE + OTSU + psm 11
  Player last  name  — y=635–740, x=90–300   top-15% R-channel + psm 11
  Game stats values  — y=505–560, x=570–870  combined per-digit + PSM 7 voting
  Stats label gate   — y=475–497, x=550–910  template match ≥ 0.65
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
    _TESS_AVAILABLE = True
except ImportError:
    _TESS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Scan parameters ────────────────────────────────────────────────────────────
SCAN_FROM_END_S: float = 300.0
SCAN_FPS:        float = 2.0     # Phase 1
STAR_THRESHOLD:  float = 0.85   # banner-match gate; lowered from 0.95 so star
                                # cards still pass on HDR sources that read a bit
                                # off against the SDR templates (~0.90 there)

# ── Regions ────────────────────────────────────────────────────────────────────
BANNER_Y0, BANNER_Y1 = 230, 320
BANNER_X0, BANNER_X1 = 0, 960      # left 960 px — the star card panel; the score HUD is centre/right

NAME_X0,   NAME_X1  = 90,  400   # X1 wide enough for long surnames ('Voronkov'); trailing
                                 # card background is harmless (thresholds to black)
NAME_Y0,   NAME_Y1  = 635, 740
NAME_MID_Y           = 688

STATS_X0, STATS_X1 = 570, 870
STATS_Y0, STATS_Y1 = 505, 560
STATS_LABELS        = ["GOALS", "ASSISTS", "HITS"]

STATS_LABEL_Y0, STATS_LABEL_Y1 = 475, 497
STATS_LABEL_X0, STATS_LABEL_X1 = 550, 910
STATS_LABEL_THRESHOLD: float   = 0.65

SCORE_X0, SCORE_X1 = 0,  900
SCORE_Y0, SCORE_Y1 = 40, 160


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StarScreen:
    rank:         str   = ""
    player_name:  str   = ""
    player_stats: str   = ""
    score_line:   str   = ""
    timestamp_s:  float = 0.0

    def __str__(self) -> str:
        parts = [f"{self.rank} STAR: {self.player_name or '(unknown)'}"]
        if self.player_stats:
            parts.append(f"  Stats: {self.player_stats}")
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# OCR helpers
# ─────────────────────────────────────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def _name_channel(crop: np.ndarray, mode: str) -> np.ndarray:
    """Single-channel image for name OCR.

    'red'  — the red channel; high contrast for light text on the dark/teal
             cards (e.g. the default home card).
    'min'  — per-pixel min(R,G,B); isolates near-white text on ANY team-coloured
             card. White text is high in all channels, while any saturated
             background (orange, red, …) has at least one low channel, so the
             min separates text from background regardless of hue.
    """
    if mode == "min":
        return crop.min(axis=2).astype(np.uint8)
    return crop[:, :, 2]


def _ocr_first_name(frame: np.ndarray, mode: str = "red") -> str:
    if not _TESS_AVAILABLE:
        return ""
    crop = frame[NAME_Y0:NAME_MID_Y, NAME_X0:NAME_X1]
    r = _name_channel(crop, mode)
    r_c = _clahe.apply(r)
    big = cv2.resize(r_c, (r_c.shape[1] * 4, r_c.shape[0] * 4),
                     interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw = pytesseract.image_to_string(thresh, config="--psm 11 --oem 1").strip()
    words = [w for w in raw.split() if sum(c.isalpha() for c in w) >= 3]
    if not words:
        return ""
    # The first-name row holds a single word; PSM 11 sometimes adds stray
    # fragments from the card background (e.g. 'Aat', 'OiQUONONOCN'). Keep
    # the most name-like token: title case beats mixed case, then longer
    # beats shorter, ties keep the earliest.
    return max(words, key=lambda w: (w[:1].isupper() and w[1:].islower(), len(w)))


def _ocr_last_name(frame: np.ndarray, mode: str = "red") -> str:
    if not _TESS_AVAILABLE:
        return ""
    crop = frame[NAME_Y0:NAME_Y1, NAME_X0:NAME_X1]
    r = _name_channel(crop, mode)
    thresh_val = int(np.percentile(r, 85))
    big = cv2.resize(r, (r.shape[1] * 4, r.shape[0] * 4),
                     interpolation=cv2.INTER_CUBIC)
    _, t = cv2.threshold(big, thresh_val, 255, cv2.THRESH_BINARY)
    # The crop spans both name rows (the percentile threshold needs the full
    # region's pixel distribution). The last name is the BOTTOM text row, so
    # pick by word position rather than by length — a longer first name would
    # otherwise win (e.g. 'Martin' over 'Necas').
    data = pytesseract.image_to_data(
        t, config="--psm 11 --oem 1", output_type=pytesseract.Output.DICT
    )
    cands = [
        (top, w.strip())
        for w, top in zip(data["text"], data["top"])
        if sum(c.isalpha() for c in w.strip()) >= 3
    ]
    if not cands:
        return ""
    bottom_top = max(top for top, _ in cands)
    same_row = [w for top, w in cands if top >= bottom_top - 60]  # ×4-scaled px
    return max(same_row, key=len)


def _stats_per_digit(r: np.ndarray) -> list[str] | None:
    """
    Locate the 3 digit columns by brightness peak, then OCR each in isolation.
    Returns ['goals', 'assists', 'hits'] or None if 3 peaks cannot be found.
    Per-digit isolation avoids PSM 7 confusing '8' for '3' in context.
    """
    global_max = int(r.max())
    if global_max < 100:
        return None

    col_max = r.max(axis=0).astype(float)
    peak_thresh = global_max * 0.50

    # Find contiguous bright column regions
    regions: list[tuple[int, float]] = []
    in_r = False
    start = 0
    for x, v in enumerate(col_max):
        if v > peak_thresh and not in_r:
            in_r = True; start = x
        elif v <= peak_thresh and in_r:
            in_r = False
            regions.append(((start + x) // 2, float(col_max[start:x].max())))
    if in_r:
        regions.append(((start + len(col_max)) // 2, float(col_max[start:].max())))

    # Keep top-3 brightest separated by ≥40 px
    regions.sort(key=lambda x: -x[1])
    selected: list[tuple[int, float]] = []
    for cx, bri in regions:
        if all(abs(cx - sc) >= 40 for sc, _ in selected):
            selected.append((cx, bri))
        if len(selected) == 3:
            break

    if len(selected) < 3:
        return None
    selected.sort(key=lambda x: x[0])  # left-to-right

    nums: list[str] = []
    for cx, _ in selected:
        x0 = max(0, cx - 28)
        x1 = min(r.shape[1], cx + 28)
        digit_r = r[:, x0:x1]
        # Threshold: higher of (65% of strip max) or (p75 of this digit crop)
        # The p75 guard handles thick digits (like '8') that need a tighter cut.
        thresh = max(int(global_max * 0.65), int(np.percentile(digit_r, 75)))
        big = cv2.resize(digit_r, (digit_r.shape[1] * 4, digit_r.shape[0] * 4),
                         interpolation=cv2.INTER_CUBIC)
        _, t = cv2.threshold(big, thresh, 255, cv2.THRESH_BINARY)
        found = ""
        for psm in [8, 10, 13]:
            raw = pytesseract.image_to_string(
                t, config=f"--psm {psm} --oem 1 -c tessedit_char_whitelist=0123456789"
            ).strip()
            d = re.findall(r"\d+", raw)
            if d:
                found = d[0]
                break
        nums.append(found)

    return nums if all(nums) else None


def _stats_psm7(r: np.ndarray) -> list[str] | None:
    """
    Multi-percentile PSM 7 on the full stats strip.
    Returns ['goals', 'assists', 'hits'] or None if <3 digit groups found.
    """
    for pct in [85, 80, 75]:
        thresh_val = int(np.percentile(r, pct))
        big = cv2.resize(r, (r.shape[1] * 4, r.shape[0] * 4),
                         interpolation=cv2.INTER_CUBIC)
        _, t = cv2.threshold(big, thresh_val, 255, cv2.THRESH_BINARY)
        raw = pytesseract.image_to_string(t, config="--psm 7 --oem 1").strip()
        cleaned = re.sub(r"[FIlO|]", "1", raw)
        nums = re.findall(r"\d+", cleaned)
        if len(nums) >= 3:
            return nums[:3]
    return None


def _ocr_stats_votes(
    r: np.ndarray,
    pos_votes: list[Counter],
) -> None:
    """
    Run both readers on the R-channel stats strip and add to shared vote counters.
    Per-digit readings get weight 2; PSM 7 readings get weight 1.
    """
    pd = _stats_per_digit(r)
    if pd:
        for i, v in enumerate(pd):
            pos_votes[i][v] += 2

    p7 = _stats_psm7(r)
    if p7:
        for i, v in enumerate(p7):
            pos_votes[i][v] += 1


def _votes_to_stats(pos_votes: list[Counter]) -> str:
    """Convert per-position vote counters to a stats string. Always returns digits."""
    nums: list[str] = []
    for ctr in pos_votes:
        winner = ctr.most_common(1)[0][0] if ctr else "0"
        nums.append(winner if winner.isdigit() else "0")
    return " | ".join(f"{label}: {num}" for label, num in zip(STATS_LABELS, nums))


def _ocr_score_line(frame: np.ndarray) -> str:
    if not _TESS_AVAILABLE:
        return ""
    crop = frame[SCORE_Y0:SCORE_Y1, SCORE_X0:SCORE_X1]
    big = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    raw = pytesseract.image_to_string(gray, config="--psm 7 --oem 1").strip()
    for line in raw.splitlines():
        if re.search(r"(END|GAME|OVER|\d\s*[-–]\s*\d)", line, re.I):
            return line.strip()
    return raw.splitlines()[0].strip() if raw else ""


# ─────────────────────────────────────────────────────────────────────────────
# Template loading + matching
# ─────────────────────────────────────────────────────────────────────────────

def _load_templates(
    configs_dir: Path,
) -> tuple[dict[str, list[np.ndarray]], np.ndarray | None]:
    """Load star-card banner templates, allowing multiple per rank.

    Any file matching ``star_<rank>_star_template*.png`` is used, so extra
    variants (e.g. star_first_star_template2.png) can be dropped in and the
    best match across them wins. Note: variants should be cut from SDR
    (tone-mapped) footage to match what the pipeline feeds the detector.
    """
    templates: dict[str, list[np.ndarray]] = {}
    for rank, stem in [
        ("FIRST",  "star_first_star_template"),
        ("SECOND", "star_second_star_template"),
        ("THIRD",  "star_third_star_template"),
    ]:
        imgs: list[np.ndarray] = []
        for p in sorted(configs_dir.glob(f"{stem}*.png")):
            img = cv2.imread(str(p))
            if img is not None:
                imgs.append(img)
                logger.debug("Loaded %s (%dx%d)", p.name, img.shape[1], img.shape[0])
            else:
                logger.warning("Cannot read template: %s", p)
        if imgs:
            templates[rank] = imgs
            logger.debug("%s: %d template(s)", rank, len(imgs))
        else:
            logger.warning("No templates found for %s (%s*.png)", rank, stem)

    label_p = configs_dir / "star_stats_label_template.png"
    stats_label_tmpl: np.ndarray | None = None
    if label_p.exists():
        img = cv2.imread(str(label_p))
        if img is not None:
            stats_label_tmpl = img
            logger.debug("Loaded stats-label template (%dx%d)",
                         img.shape[1], img.shape[0])
        else:
            logger.warning("Cannot read stats-label template: %s", label_p)

    return templates, stats_label_tmpl


def _match_banner(frame: np.ndarray, template: np.ndarray) -> float:
    region = frame[BANNER_Y0:BANNER_Y1, BANNER_X0:BANNER_X1]
    if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
        return 0.0
    return float(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED).max())


def _stats_animation_done(
    frame: np.ndarray, label_tmpl: np.ndarray | None
) -> bool:
    """Return True when the stats label row and digit strip are both fully rendered.

    Primary gate: template match against the label template (works on dark/teal
    cards).  Fallback: brightness-based heuristic that works on any team-colour
    card (e.g. orange Oilers card).  The heuristic checks two conditions:

      1. Label row (y=475-497) has non-trivial std-dev, meaning text is present.
      2. Stats digit strip (y=505-560) has a bright pixel (max of min-channel
         > 140), meaning at least one digit has fully faded in.

    Using the min-channel (per-pixel min of R,G,B) isolates near-white/gray
    text from any saturated background (orange, teal, …).
    """
    region = frame[STATS_LABEL_Y0:STATS_LABEL_Y1, STATS_LABEL_X0:STATS_LABEL_X1]

    # Primary: template match (dark/teal cards)
    if label_tmpl is not None:
        th, tw = label_tmpl.shape[:2]
        if region.shape[0] >= th and region.shape[1] >= tw:
            score = float(cv2.matchTemplate(region, label_tmpl, cv2.TM_CCOEFF_NORMED).max())
            if score >= STATS_LABEL_THRESHOLD:
                return True

    # Fallback: brightness heuristic for team-coloured cards
    # Label row must show visible text (std > 10 in the min channel)
    label_min = region.min(axis=2)
    if label_min.std() < 10:
        return False
    # Digit strip must have at least one bright pixel (animation complete)
    digit_strip = frame[STATS_Y0:STATS_Y1, STATS_X0:STATS_X1]
    digit_min = digit_strip.min(axis=2)
    return int(digit_min.max()) > 140


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: linear OCR scan
# ─────────────────────────────────────────────────────────────────────────────

def _clean_name(name: str) -> str:
    """Keep only plausible name tokens: alpha-only, 3–15 chars, contains a vowel."""
    cleaned = re.sub(r"[^A-Za-z\s'-]", " ", name)
    tokens = []
    for w in cleaned.split():
        if not w.isalpha() or not (3 <= len(w) <= 15):
            continue
        if not re.search(r"[AEIOUaeiou]", w):  # filter OCR noise like "CYC", "lUYl"
            continue
        tokens.append(w)
    return " ".join(tokens[:3])


def _name_quality(name: str) -> int:
    return len([w for w in name.split() if w.isalpha() and len(w) >= 3][:4])


def _ocr_around_peak(
    cap: cv2.VideoCapture,
    peak_ts: float,
    duration_s: float,
    stats_label_tmpl: np.ndarray | None = None,
) -> tuple[str, str, str]:
    """
    Two-pass strategy around peak_ts:

    Stats — linear scan from (peak−3 s) to (peak+4 s) at ~6 fps with a single
    seek. Accumulates weighted votes (per-digit w=2, PSM7 w=1) across all
    gate-passing frames for maximum robustness.

    Names — outward-step seeks (0, ±0.5, ±1.0 … ±3.0 s) from the peak. These
    targeted seeks land on the frames with the clearest name rendering; the
    linear scan's many gate-passing frames introduce too much OCR noise.
    """
    # ── Stats: linear scan ───────────────────────────────────────────────────
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every_n    = max(1, round(fps / 6.0))
    scan_start = max(0.0,        peak_ts - 3.0)
    scan_end   = min(duration_s, peak_ts + 4.0)

    cap.set(cv2.CAP_PROP_POS_MSEC, scan_start * 1000.0)
    pos_votes: list[Counter] = [Counter(), Counter(), Counter()]
    best_score_l = ""
    got_score_l  = False

    while True:
        for _ in range(every_n - 1):
            if not cap.grab():
                break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts > scan_end:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if not _stats_animation_done(frame, stats_label_tmpl):
            continue
        # Use per-pixel min(R,G,B) channel: isolates near-white/gray digits
        # from any team-colour background (orange, teal, dark, …).
        # On a dark/teal card min≈R≈G≈B for white text; on an orange card
        # the background has low B so min(R,G,B)≈B while digits remain bright.
        r = frame[STATS_Y0:STATS_Y1, STATS_X0:STATS_X1].min(axis=2).astype(np.uint8)
        _ocr_stats_votes(r, pos_votes)
        if not got_score_l:
            best_score_l = _ocr_score_line(frame)
            got_score_l  = True

    stats_str = _votes_to_stats(pos_votes) if any(pos_votes) else ""

    # ── Names: forward-biased seeks × colour channels ────────────────────────
    # The banner fires the instant the card appears, but the LAST name fades in
    # over the next ~1 s — so earlier frames only ever have a partial last name.
    # Sample forward-densely (with a couple of fallback look-backs) so the fully
    # rendered name is captured; the length tiebreaker then prefers it.
    # Each frame is read with both colour channels ('red' for dark/teal cards,
    # 'min' for team-coloured cards) and all readings compete in one pool.
    steps = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.5, -0.5, -1.0]

    def _title_score(name: str) -> int:
        """Count tokens that are properly title-cased (first upper, rest lower)."""
        return sum(
            1 for w in name.split()
            if w and w[0].isupper() and w[1:].islower()
        )

    name_candidates: list[tuple[int, int, int, str]] = []  # (quality, title_score, alpha_len, name)
    for step in steps:
        ts = min(max(0.0, peak_ts + step), duration_s)
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        # No gate here — player name is visible before stats finish animating.
        for mode in ("red", "min"):
            first = _ocr_first_name(frame, mode)
            last  = _ocr_last_name(frame, mode)
            name  = _clean_name(f"{first} {last}".strip())
            q     = _name_quality(name)
            tc    = _title_score(name)
            alpha_len = sum(c.isalpha() for c in name)
            name_candidates.append((q, tc, alpha_len, name))
            logger.debug("  name-seek t=%.2fs [%s]  q=%d  tc=%d  len=%d  name=%r",
                         ts, mode, q, tc, alpha_len, name)

    # Primary: quality; secondary: title-case tokens; tertiary: total alpha length.
    best_name = max(name_candidates, key=lambda x: (x[0], x[1], x[2]))[3] if name_candidates else ""

    return best_name, stats_str, best_score_l


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_star_screens(
    video_path: str | Path,
    configs_dir: str | Path | None = None,
) -> list[StarScreen]:
    """Detect FIRST / SECOND / THIRD STAR screens. Returns up to 3 StarScreen objects."""
    video_path  = Path(video_path)
    configs_dir = Path(configs_dir or Path(__file__).parent.parent.parent / "configs")

    templates, stats_label_tmpl = _load_templates(configs_dir)
    if not templates:
        logger.warning("No star templates loaded — skipping star detection")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_s   = total_frames / fps
    scan_start_s = max(0.0, duration_s - SCAN_FROM_END_S)
    every_n      = max(1, int(round(fps / SCAN_FPS)))

    logger.info("Star detector: phase 1 scan %.1f–%.1f s  every %d frames",
                scan_start_s, duration_s, every_n)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    peak_by_rank: dict[str, tuple[float, float]] = {}
    cap.set(cv2.CAP_PROP_POS_MSEC, scan_start_s * 1000.0)
    while True:
        for _ in range(every_n - 1):
            if not cap.grab():
                break
        ok, frame = cap.read()
        if not ok:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        for rank, tmpls in templates.items():
            score = max(_match_banner(frame, t) for t in tmpls)
            if score > peak_by_rank.get(rank, (0, 0.0))[1]:
                peak_by_rank[rank] = (ts, score)

    logger.info("Phase 1 peaks: %s",
                {r: f"{s:.3f}@{t:.1f}s" for r, (t, s) in peak_by_rank.items()})

    confirmed = {r: v for r, v in peak_by_rank.items() if v[1] >= STAR_THRESHOLD}
    if not confirmed:
        cap.release()
        return []

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    screens: list[StarScreen] = []
    for rank in ["THIRD", "SECOND", "FIRST"]:
        if rank not in confirmed:
            continue
        peak_ts, peak_score = confirmed[rank]
        logger.info("Phase 2: %s STAR  t=%.1fs  score=%.3f", rank, peak_ts, peak_score)

        player_name, player_stats, score_line = _ocr_around_peak(
            cap, peak_ts, duration_s, stats_label_tmpl
        )
        screen = StarScreen(
            rank=rank,
            player_name=player_name,
            player_stats=player_stats,
            score_line=score_line,
            timestamp_s=peak_ts,
        )
        screens.append(screen)
        logger.info("  → %s", screen)

    cap.release()
    return screens
