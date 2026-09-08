"""
tests/test_period_detection.py

Regression test for game-clock period / overtime detection
(apps/reel_builder/src/detection/game_clock_detector.py).

The reel built from "NHL 25_20260903192725_norm.mp4" (a 1–0 game that finished
1–3 in regulation) got two wrong period cards:

  * a spurious OVERTIME transition — a scoreboard frame was OCR'd as "2OT" and,
    with no confidence gate on ot_period_end, it confirmed overtime;
  * a 3RD PERIOD card ~90 s after 3rd-period play had already started — at the
    real 3rd-period puck drop the period-start matcher took the first template
    over threshold (1st/2nd period) instead of the highest-confidence one (3rd),
    and a later "20:00 / 3RD" OCR misread (no gate on OCR period_start) fired
    again mid-period.

Fixture: tests/fixtures/period_detection_20260903192725.mp4 — six short 1080p
windows spliced from that recording:

  A  330–356 s  — scoreboard frame that OCRs as "2OT"
  B  588–604 s  — real 2nd-period puck drop  (20:00 / 2ND)
  C  916–936 s  — real 3rd-period puck drop  (20:00 / 3RD, unambiguous)
  D 1066–1086 s — mid-3rd; the 2nd- and 3rd-period clock templates both score
                  high on this frame (the tie-break)
  E 1372–1392 s — a "1:00" clock OCR'd as "20:00 / 3RD"  (false period start)
  F 1480–1500 s — real end-of-regulation buzzer  (kept last for the reverse scan)

Not in git (*.mp4 is ignored). Recreate from the normalised recording with:

    tests/fixtures/make_period_fixture.sh /path/to/NHL\\ 25_20260903192725_norm.mp4

Pre-fix, the detector reads this as game_start(1) → period_start(2) →
game_start(1) → period_start(3) → period_start(2) → period_start(?) : a second
game_start mid-stream and a period counter that runs backwards. After the fix it
is a clean 1 → 2 → 3 with a single regulation buzzer and no overtime.
"""

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "apps" / "reel_builder"))

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "period_detection_20260903192725.mp4"

pytestmark = [
    pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture clip missing: {FIXTURE}"),
    pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed"),
]


def _detector():
    import yaml
    from src.detection.game_clock_detector import GameClockDetector

    cfg = yaml.safe_load((REPO_ROOT / "apps/reel_builder/configs/config.yaml").read_text())
    g = cfg.get("game_clock", {})
    return GameClockDetector(
        template_path=REPO_ROOT / "apps/reel_builder/configs/game_end_template.png",
        min_game_end_conf=g.get("min_game_end_conf", 0.85),
        min_period_end_conf=g.get("min_period_end_conf", 0.85),
        min_period_start_conf=g.get("min_period_start_conf", 0.85),
        period_end_confirm_window_s=g.get("period_end_confirm_window_s", 10.0),
        period_end_confirm_min_hits=int(g.get("period_end_confirm_min_hits", 2)),
        period_end_confirm_interval_s=g.get("period_end_confirm_interval_s", 1.0),
    )


@pytest.fixture(scope="module")
def events():
    return _detector().scan_video(str(FIXTURE))


# ── End-to-end: the whole clip must read as a clean 3-period regulation game ──

def test_no_overtime_detected(events):
    """The 2OT scoreboard misread must not confirm an overtime period."""
    assert not any(e["event"] == "ot_period_end" for e in events), events


def test_single_regulation_end_and_it_is_last(events):
    reg = [e for e in events if e["event"] == "regulation_end"]
    assert len(reg) == 1, events
    assert events[-1]["event"] == "regulation_end", events


def test_game_start_only_appears_once_at_the_front(events):
    """Pre-fix, the 1st-period clock template matched the real 3rd-period puck
    drop (20:00 / 3RD) and emitted a second game_start mid-game."""
    gs = [i for i, e in enumerate(events) if e["event"] == "game_start"]
    assert gs in ([], [0]), events


def test_period_progression_is_monotonic_up_to_the_third(events):
    """The core bug: the period counter ran 1 → 2 → 1 → 3 → 2. The puck-drop
    events must climb monotonically and stop at the 3rd period."""
    periods = [e.get("period") for e in events
               if e["event"] in ("game_start", "period_start")]
    assert None not in periods, f"an ungated OCR period_start slipped through: {events}"
    assert periods == sorted(periods), f"period counter went backwards: {periods}"
    assert 2 in periods and 3 in periods, periods
    assert max(periods) == 3, f"phantom period past regulation: {periods}"


# ── Unit level: the matcher fixes, independent of OCR / tesseract ────────────

def _frame_at(seconds):
    import cv2

    cap = cv2.VideoCapture(str(FIXTURE))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * (cap.get(cv2.CAP_PROP_FPS) or 30.0)))
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read fixture frame at {seconds}s"
    return frame


def test_fixture_still_exercises_the_period_template_ambiguity():
    """The tie-break fix only bites while both the 2nd- and 3rd-period clock
    templates clear the gate on the mid-3rd frame and the 3rd scores at least as
    high. If a template is re-cut and that stops holding, this fixture no longer
    guards the regression — fail loudly so it gets refreshed."""
    det = _detector()
    frame = _frame_at(74)  # window D — mid-3rd, source ~1078 s
    scores = {p: det._match_period_start_template(frame, tmpl)
              for p, _ev, tmpl in det._period_start_templates}
    assert scores[2] >= det.min_period_start_conf, scores
    assert scores[3] >= det.min_period_start_conf, scores
    assert scores[3] >= scores[2], scores


def test_ocr_only_period_start_is_below_the_confidence_gate():
    """The "20:00 / 3RD" misread frame carries a low clock-template confidence,
    so gating OCR period_start by min_period_start_conf drops it."""
    det = _detector()
    frame = _frame_at(90)  # window E — source ~1380 s
    assert det._template_conf(frame) < det.min_period_start_conf


def test_ot_period_end_needs_a_sustained_buzzer_not_one_frame():
    """A lone "2OT"-ish frame must not confirm overtime: the dense rescan finds
    no run of near-zero OT-clock frames around it."""
    import cv2

    det = _detector()
    cap = cv2.VideoCapture(str(FIXTURE))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        assert not det._confirm_period_end(cap, fps, 10.0, accept={"ot_period_end"})
    finally:
        cap.release()
