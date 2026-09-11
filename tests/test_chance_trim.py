"""
tests/test_chance_trim.py

_trim_chance_clips must anchor on the classifier's winning-window timestamp
whenever one is available, and only fall back to the whole-clip audio peak
when it isn't.

Two regressions from the 20260903192725 game, same scene (126), same
Blackhawks chance:
  1. No audio spike at all (the game only spiked on goals) → the clip played
     in full instead of trimming to the action.
  2. Fixed #1 by adding an audio-peak fallback, but _audio_peak_time picks the
     single loudest RMS frame in the WHOLE (up to 45s) clip — on a scoring
     chance that's often commentary/a hit/crowd noise from an unrelated
     stretch, not the shot. It out-ranked the real window, so the reel clip
     still opened after the breakaway instead of showing it. Window must win.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "apps" / "reel_builder"))

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "period_detection_20260903192725.mp4"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture clip missing: {FIXTURE}")


def test_window_start_used_when_no_audio_peak(monkeypatch):
    import pipeline

    # fixture carries no audio track → real _audio_peak_time returns None
    monkeypatch.setattr(pipeline, "_get_clip_duration", lambda _p: 32.0)

    seg = {
        "path": str(FIXTURE),
        "label": "scoring_chance",
        "window_start_s": 4.0,
        "window_end_s": 8.0,
    }
    pipeline._trim_chance_clips([seg])

    assert "trim_start_s" in seg and "trim_end_s" in seg
    assert seg["trim_start_s"] == pytest.approx(0.0)          # max(0, 4 - 5)
    assert seg["trim_end_s"] == pytest.approx(11.0)           # 8 + 3
    assert seg["trim_end_s"] - seg["trim_start_s"] < 32.0     # not the whole scene


def test_no_trim_without_peak_or_window(monkeypatch):
    import pipeline

    monkeypatch.setattr(pipeline, "_get_clip_duration", lambda _p: 32.0)
    seg = {"path": str(FIXTURE), "label": "scoring_chance"}
    pipeline._trim_chance_clips([seg])
    assert "trim_start_s" not in seg


def test_window_wins_over_a_misleading_audio_peak(monkeypatch):
    """The real bug: a genuine audio peak (e.g. a hit at 12.2s) sits well after
    the actual chance (a 4-8s window) — trusting it clips out the breakaway."""
    import pipeline

    monkeypatch.setattr(pipeline, "_get_clip_duration", lambda _p: 32.0)
    monkeypatch.setattr(pipeline, "_audio_peak_time", lambda _p: 12.2)

    seg = {
        "path": str(FIXTURE),
        "label": "scoring_chance",
        "window_start_s": 4.0,
        "window_end_s": 8.0,
    }
    pipeline._trim_chance_clips([seg])

    # anchored on the window (0–11s), not the misleading peak (7.2–15.2s)
    assert seg["trim_start_s"] == pytest.approx(0.0)
    assert seg["trim_end_s"] == pytest.approx(11.0)


def test_audio_peak_still_used_when_no_window(monkeypatch):
    import pipeline

    monkeypatch.setattr(pipeline, "_get_clip_duration", lambda _p: 32.0)
    monkeypatch.setattr(pipeline, "_audio_peak_time", lambda _p: 12.2)

    seg = {"path": str(FIXTURE), "label": "scoring_chance"}
    pipeline._trim_chance_clips([seg])

    assert seg["trim_start_s"] == pytest.approx(7.2)
    assert seg["trim_end_s"] == pytest.approx(15.2)
