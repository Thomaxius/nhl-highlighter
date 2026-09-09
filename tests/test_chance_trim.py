"""
tests/test_chance_trim.py

_trim_chance_clips must fall back to the classifier's winning-window timestamp
when there is no audio spike — otherwise a scoring chance rescued from a long
(~30 s) demoted-goal segment plays in full. Regression for the 20260903192725
game, where the Blackhawks' chance lived in a 32 s scene the audio never spiked.
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
