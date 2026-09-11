"""
tests/test_classifier_windowing.py

Regression for scene 131 (20260903192725 game): the reel clip cut off the
crash-the-net shot ~3-5s before it happened. The scene's overall winner was
a banner-less 'goal' window mid-clip; Step 4c demotes that to 'other' and
Step 4f.5 rescues it back to a scoring_chance, anchored on window_start_s /
window_end_s from _classify_windowed's clustering.

The actual scramble/shot continued in windows the model tagged
'scoring_chance' rather than 'goal' — a few seconds later, same play, same
net-mouth crowding, just a wobble in which of two near-identical classes won
that window. _classify_windowed only clustered windows sharing the exact
winning label, so those later scoring_chance windows never joined the 'goal'
cluster the trim was anchored on, and the reel cut right where the build-up
ended instead of where the shot happened.

This test drives _classify_windowed end-to-end (real cv2 frame reads against
an existing fixture clip, only long enough to matter — content is irrelevant)
with _classify_frames mocked to return a scripted sequence of window labels,
so it doesn't depend on the actual model (whose exact softmax outputs vary
by torch/transformers build and aren't reproducible off the training box).
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "apps" / "reel_builder"))

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "period_detection_20260903192725.mp4"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture clip missing: {FIXTURE}")


def _make_classifier(monkeypatch, scripted_windows):
    """A HighlightClassifier whose _classify_frames pops the next scripted
    (label, confidence) in call order, without ever loading the real model."""
    from src.detection.classifier import HighlightClassifier

    clf = HighlightClassifier.__new__(HighlightClassifier)  # skip __init__ / model load
    clf.num_frames = 16
    queue = list(scripted_windows)

    def fake_classify_frames(_self_or_path, maybe_frames=None):
        label, conf = queue.pop(0)
        scores = {l: 0.0 for l in
                  ["goal", "celebration", "goal_replay", "other_replay",
                   "scoring_chance", "faceoff_cutscene", "faceoff", "other"]}
        scores[label] = conf
        return {"path": "x", "label": label, "confidence": conf, "scores": scores}

    monkeypatch.setattr(
        HighlightClassifier, "_classify_frames",
        lambda self, video_path, frames: fake_classify_frames(video_path, frames),
    )
    return clf


def test_goal_window_absorbs_a_later_scoring_chance_climax(monkeypatch):
    """12s of scripted windows: a 'goal' peak at [4,8) then 'faceoff' filler,
    then the real climax reads as 'scoring_chance' at [8,12). The trim window
    must extend to cover it instead of stopping at the 'goal' cluster alone."""
    import cv2

    # 6 windows for a 12s/30fps clip: starts at 0,2,4,6,8,10 (4s window, 2s stride)
    scripted = [
        ("faceoff", 0.30),          # [ 0- 4s] build-up, ignored (low conf)
        ("goal", 0.55),             # [ 2- 6s]
        ("goal", 0.75),             # [ 4- 8s]  <- single highest-confidence window (`best`)
        ("faceoff", 0.20),          # [ 6-10s] dip between the two labels
        ("scoring_chance", 0.65),   # [ 8-12s] the actual shot — tagged the sibling label
        ("scoring_chance", 0.60),   # [10-12s]
    ]
    clf = _make_classifier(monkeypatch, scripted)

    cap = cv2.VideoCapture(str(FIXTURE))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    total_frames = int(12.0 * fps)

    winner = clf._classify_windowed(FIXTURE, fps, total_frames, 12.0)

    assert winner["label"] == "goal"
    assert winner["window_start_s"] == pytest.approx(2.0)
    # Old behaviour (goal-only clustering) stopped at 8.0s, cutting the shot.
    assert winner["window_end_s"] == pytest.approx(12.0)


def test_goal_winner_still_exposes_a_second_unrelated_chance(monkeypatch):
    """Regression for scene 120 (same game): an early net-front scramble wins
    the scene as 'goal' (demoted/rescued like scene 131), but a second, fully
    separate chance ~10s later — a real shot ("SHOT SPEED 82.5 MPH" on
    screen) — sits well outside that cluster's reach (unlike scene 131, the
    two are >4s apart, so pooling doesn't merge them into one window). Before
    this fix, sc_extra_windows was only ever attached when the winning label
    was already scoring_chance, so this second chance was silently dropped
    for every demoted-goal rescue instead."""
    import cv2

    # 10 windows for a 22s/30fps clip: starts at 0,2,4,...,18
    scripted = [
        ("faceoff", 0.30),          # [ 0- 4s]
        ("goal", 0.65),             # [ 2- 6s]  cluster 0 (primary)
        ("goal", 0.80),             # [ 4- 8s]  <- single highest-confidence window (`best`)
        ("faceoff", 0.20),          # [ 6-10s]  gap
        ("faceoff", 0.20),          # [ 8-12s]  gap
        ("faceoff", 0.20),          # [10-14s]  gap (>4s clear of cluster 0's end at 8s)
        ("faceoff", 0.20),          # [12-16s]  gap
        ("scoring_chance", 0.65),   # [14-18s]  cluster 1 (extra) — the real second chance
        ("scoring_chance", 0.62),   # [16-20s]  (clears the 0.60 gate with margin — avoid a float-equality edge)
        ("faceoff", 0.20),          # [18-22s]
    ]
    clf = _make_classifier(monkeypatch, scripted)

    cap = cv2.VideoCapture(str(FIXTURE))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    total_frames = int(22.0 * fps)

    winner = clf._classify_windowed(FIXTURE, fps, total_frames, 22.0)

    assert winner["label"] == "goal"
    assert winner["window_start_s"] == pytest.approx(2.0)   # primary (highest-confidence) cluster
    assert winner["window_end_s"] == pytest.approx(8.0)
    extra = winner.get("sc_extra_windows")
    assert extra, "the second, later chance must not be silently dropped"
    assert extra[0]["trim_start_s"] == pytest.approx(7.0)   # max(0, 14 - 7)
    assert extra[0]["trim_end_s"] == pytest.approx(22.0)    # min(22, 20 + 3)


def test_scoring_chance_winner_still_anchors_on_its_first_peak(monkeypatch):
    """Unaffected case: when the winner already IS scoring_chance, clustering
    must not start pooling other labels in — still anchor on the first peak."""
    import cv2

    scripted = [
        ("scoring_chance", 0.65),   # [0-4s]  <- first peak, chance builds here
        ("scoring_chance", 0.80),   # [2-6s]  <- highest confidence, but NOT first
        ("other", 0.90),            # [4-8s]
        ("goal", 0.40),             # [6-10s] below the confidence gate, ignored
    ]
    clf = _make_classifier(monkeypatch, scripted)

    cap = cv2.VideoCapture(str(FIXTURE))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    total_frames = int(10.0 * fps)

    winner = clf._classify_windowed(FIXTURE, fps, total_frames, 10.0)

    assert winner["label"] == "scoring_chance"
    assert winner["window_start_s"] == pytest.approx(0.0)
    assert "trim_start_s" in winner and "trim_end_s" in winner
