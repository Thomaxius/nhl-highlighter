"""
tests/test_endgame_detection.py

Automated end-to-end test for the end-of-game detectors: Three Stars screen
(star_detector) and team stats screens (stats_detector), plus the YouTube
description they produce together. Asserts against known-good values for the
fixture clip, so it can run in test automation.

Run (from repo root, takes a few minutes — it OCR-scans the whole clip twice):
    .venv/bin/python -m pytest tests/test_endgame_detection.py -v

Fixture: tests/fixtures/eqh-IQga1CM_last2min.mp4 — the last ~2 minutes of
"San Jose - Seattle Preseason #4 [eqh-IQga1CM].mp4" (contains the Three Stars
sequence and both teams' stats screens). Not in git (*.mp4 is ignored); the
tests skip if it is missing. Recreate it from the full video with:
    ffmpeg -sseof -120 -i "San Jose - Seattle Preseason #4 [eqh-IQga1CM].mp4" \
        -c copy tests/fixtures/eqh-IQga1CM_last2min.mp4
"""

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "eqh-IQga1CM_last2min.mp4"

pytestmark = [
    pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture clip missing: {FIXTURE}"),
    pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed"),
]

# Known-good ground truth for the fixture clip (verified by eye against the
# actual Three Stars screens): rank → (name, stats, timestamp ±2s).
EXPECTED_STARS = {
    "FIRST":  ("Martin Necas",    "GOALS: 2 | ASSISTS: 0 | HITS: 8", 52.0),
    "SECOND": ("Matthew Tkachuk", "GOALS: 1 | ASSISTS: 1 | HITS: 8", 37.0),
    "THIRD":  ("Sam Chow",        "GOALS: 1 | ASSISTS: 1 | HITS: 3", 27.0),
}


@pytest.fixture(scope="module")
def stars():
    from apps.reel_builder.src.detection.star_detector import detect_star_screens
    return detect_star_screens(FIXTURE)


@pytest.fixture(scope="module")
def screens():
    from apps.reel_builder.src.detection.stats_detector import detect_stats_screens
    return detect_stats_screens(FIXTURE)


# ── Three Stars ────────────────────────────────────────────────────────────────

def test_all_three_stars_detected(stars):
    assert {s.rank for s in stars} == set(EXPECTED_STARS)


@pytest.mark.parametrize("rank", list(EXPECTED_STARS))
def test_star_name_stats_and_timing(stars, rank):
    expected_name, expected_stats, expected_t = EXPECTED_STARS[rank]
    star = next(s for s in stars if s.rank == rank)
    assert star.player_name == expected_name
    assert star.player_stats == expected_stats
    assert abs(star.timestamp_s - expected_t) <= 2.0


# ── Stats screens ──────────────────────────────────────────────────────────────

def test_all_four_stats_screens_detected(screens):
    found = {(s.team, s.tab) for s in screens}
    assert found == {
        ("SEATTLE KRAKEN", "ALL SKATERS"),
        ("SEATTLE KRAKEN", "GOALIES"),
        ("SAN JOSE SHARKS", "ALL SKATERS"),
        ("SAN JOSE SHARKS", "GOALIES"),
    }


def _rows(screens, team, tab):
    return next(s.rows for s in screens if s.team == team and s.tab == tab)


def test_skater_spot_checks(screens):
    sjs = _rows(screens, "SAN JOSE SHARKS", "ALL SKATERS")
    tkachuk = next(r for r in sjs if r["PLAYER"] == "M. Tkachuk")
    assert (tkachuk["G"], tkachuk["A"], tkachuk["PTS"], tkachuk["HITS"]) == ("1", "1", "2", "8")

    sea = _rows(screens, "SEATTLE KRAKEN", "ALL SKATERS")
    beniers = next(r for r in sea if r["PLAYER"] == "M. Beniers")
    assert (beniers["MIN"], beniers["A"], beniers["PTS"], beniers["HITS"]) == ("15:52", "2", "2", "3")


def test_goalie_spot_checks(screens):
    sjs = _rows(screens, "SAN JOSE SHARKS", "GOALIES")
    ravensbergen = next(r for r in sjs if r["PLAYER"] == "J. Ravensbergen")
    assert (ravensbergen["SA"], ravensbergen["SV%"], ravensbergen["GA"]) == ("21", ".857", "3")


# ── YouTube description ────────────────────────────────────────────────────────

def test_description_output(stars, screens):
    from apps.reel_builder.src.detection.stats_detector import format_description

    desc = format_description("__TITLE__", screens, stars=stars)

    # Stars section, with stats in the compact style
    assert "⭐⭐⭐ Martin Necas — 2 G, 0 A, 8 hits" in desc
    assert "⭐⭐ Matthew Tkachuk — 1 G, 1 A, 8 hits" in desc
    assert "⭐ Sam Chow — 1 G, 1 A, 3 hits" in desc

    # Labeled player lines (no space-aligned tables — YouTube fonts are
    # proportional) and no markdown fences
    assert "M. Tkachuk — 24:06 | 1 G, 1 A, 2 P | 8 hits" in desc
    assert "J. Ravensbergen — 21 shots, .857 SV%, 3 GA" in desc
    assert "```" not in desc

    # OCR junk rows must be filtered out
    for junk in ("fe i OD", "be oe |r", "OiQUONONOCN", "[s.chow"):
        assert junk not in desc

    # Stays well inside YouTube's 5000-character description limit
    assert len(desc) < 4000
