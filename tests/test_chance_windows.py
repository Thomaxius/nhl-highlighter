"""
Regression tests for how scoring-chance clips are windowed (reel 20260903192725, 01:57-04:05).

That stretch of the reel was ~107s of continuous play that the scene splitter cut into
scenes 052-055 (26.1s / 26.1s / 27.5s / 27.5s). It showed 128s of footage: each scene kept
almost entirely (a diffuse run of 49-70% windows chained into one ~20s "cluster" plus 12s
of padding), and the last 14s of period 1 played twice because two clusters' padded
windows overlapped. The window scores below are the REAL per-window classifier output
captured from the server for those scenes, so the tests pin the behaviour on real data.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "apps" / "reel_builder"))

from src.assembly.clip_windows import cap_window, merge_windows, shot_windows  # noqa: E402

SC, OT = "scoring_chance", "other"

# (label, confidence) per 4s window at a 2s stride, captured from the server run.
SCENE_052 = [("faceoff_cutscene", .45), ("faceoff_cutscene", .35), (OT, .22), (OT, .48), (SC, .61), (OT, .59), (OT, .59),
             (SC, .63), ("goal", .37), (OT, .33), (SC, .49), (SC, .58), (SC, .49)]
SCENE_053 = [(SC, .38), (OT, .36), (SC, .34), (SC, .68), (SC, .60), (OT, .49), (SC, .65), (SC, .62), (SC, .60),
             (SC, .63), (SC, .62), (SC, .49), (SC, .5995)]   # last window: 0.59945..., just under the extra gate
SCENE_054 = [(OT, .63), (SC, .60), (SC, .66), (OT, .35), (OT, .50), (SC, .68), (SC, .70), (SC, .42), (OT, .58),
             (SC, .31), (SC, .51), (OT, .46), (SC, .45)]
SCENE_055 = [(SC, .71), (SC, .67), (SC, .71), (SC, .56), (OT, .45), (OT, .58), (SC, .41), (SC, .70), (SC, .61),
             (SC, .50), (OT, .35), (OT, .33), ("goal", .45)]


# ── pure helpers ──────────────────────────────────────────────────────────────

def test_cap_window_leaves_short_windows_alone():
    assert cap_window(2.0, 12.0, 7.0, 16.0) == (2.0, 12.0)


def test_cap_window_centres_on_the_peak_and_stays_inside_the_bounds():
    assert cap_window(0.0, 26.1, 16.0, 16.0) == pytest.approx((8.0, 24.0))
    assert cap_window(0.0, 26.1, 1.0, 16.0) == pytest.approx((0.0, 16.0))      # peak near the start
    assert cap_window(0.0, 26.1, 25.0, 16.0) == pytest.approx((10.1, 26.1))    # peak near the end


def test_merge_windows_unions_overlaps_and_keeps_the_primary_flag():
    merged = merge_windows([
        {"trim_start_s": 11.0, "trim_end_s": 27.0, "confidence": .50},
        {"trim_start_s": 0.0, "trim_end_s": 25.0, "confidence": .71, "primary": True},
    ])
    assert len(merged) == 1
    assert (merged[0]["trim_start_s"], merged[0]["trim_end_s"]) == (0.0, 27.0)
    assert merged[0]["confidence"] == .71 and merged[0]["primary"] is True


def test_merge_windows_keeps_genuinely_separate_windows_apart():
    merged = merge_windows([
        {"trim_start_s": 0.0, "trim_end_s": 8.0, "confidence": .7},
        {"trim_start_s": 15.0, "trim_end_s": 23.0, "confidence": .65},
    ])
    assert len(merged) == 2


def test_shot_windows_cut_around_each_shot():
    assert shot_windows([6.5], 26.1, 5.0, 3.0, 20.0) == [(1.5, 9.5)]


def test_shot_windows_clamp_to_the_scene():
    assert shot_windows([2.0], 26.1, 5.0, 3.0, 20.0) == [(0.0, 5.0)]
    assert shot_windows([25.5], 26.1, 5.0, 3.0, 20.0) == [(20.5, 26.1)]


def test_shot_windows_ignore_the_scoreboard_catching_up_at_scene_start():
    # a shot "at 0.07s" is the HUD reading the new count in this scene, not where it happened
    assert shot_windows([0.07], 26.1, 5.0, 3.0, 20.0, ignore_before_s=1.0) == []
    assert shot_windows([0.07, 9.0], 26.1, 5.0, 3.0, 20.0, ignore_before_s=1.0) == [(4.0, 12.0)]


def test_shot_windows_fold_close_shots_and_never_repeat_footage():
    wins = shot_windows([6.0, 10.0], 40.0, 5.0, 3.0, 20.0)
    assert wins == [(1.0, 13.0)]                       # overlapping -> one continuous clip
    far = shot_windows([6.0, 14.0, 22.0, 30.0], 40.0, 5.0, 3.0, 20.0)
    for (a0, a1), (b0, b1) in zip(far, far[1:]):
        assert b0 >= a1, "consecutive shot clips must not overlap"


# ── classifier windowing on the real server data ─────────────────────────────

@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    """A silent 28s clip at 10 fps — only needs enough frames to read; content is irrelevant
    because the classifier's per-window output is scripted."""
    import cv2
    import numpy as np
    path = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(10 * 28):
        writer.write(frame)
    writer.release()
    return path


def _classify(monkeypatch, clip, scripted, duration_s):
    import cv2
    from src.detection.classifier import HighlightClassifier

    clf = HighlightClassifier.__new__(HighlightClassifier)   # skip model load
    clf.num_frames = 16
    queue = list(scripted)

    def fake(self, video_path, frames):
        label, conf = queue.pop(0)
        return {"path": "x", "label": label, "confidence": conf, "scores": {label: conf}}

    monkeypatch.setattr(HighlightClassifier, "_classify_frames", fake)
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    cap.release()
    result = clf._classify_windowed(clip, fps, int(duration_s * fps), duration_s)
    assert not queue, "every scripted window should have been consumed"
    return result


def _length(w):
    return w["trim_end_s"] - w["trim_start_s"]


def test_scene_052_diffuse_run_is_capped_not_kept_whole(monkeypatch, synthetic_clip):
    """Before: trim 1.0-26.1 (25.1s of a 26.1s scene) for a scattered run of 49-63% windows."""
    w = _classify(monkeypatch, synthetic_clip, SCENE_052, 26.1)
    assert w["label"] == SC
    assert _length(w) <= 16.0 + 1e-6
    assert w["trim_start_s"] <= 14.0 and w["trim_end_s"] >= 18.0, "must still contain the strongest (63%) window"
    assert (w["trim_start_s"], w["trim_end_s"]) == pytest.approx((8.0, 24.0))


def test_scene_053_weak_split_off_extra_is_dropped(monkeypatch, synthetic_clip):
    """Before: primary 0-26.1 (the whole scene) PLUS an extra 17.0-26.1 (60%, entirely inside it)."""
    w = _classify(monkeypatch, synthetic_clip, SCENE_053, 26.1)
    assert _length(w) <= 16.0 + 1e-6
    assert not w.get("sc_extra_windows")


def test_scene_054_weak_extra_no_longer_repeats_eight_seconds(monkeypatch, synthetic_clip):
    """Before: 0-21.0 and 13.0-27.5 (51%) -> 8.0s shown twice."""
    w = _classify(monkeypatch, synthetic_clip, SCENE_054, 27.5)
    assert (w["trim_start_s"], w["trim_end_s"]) == pytest.approx((5.0, 21.0))
    assert not w.get("sc_extra_windows")


def test_scene_055_period_end_is_not_shown_twice(monkeypatch, synthetic_clip):
    """Before: 0-25.0 and 11.0-27.0 (50%) -> 14.0s of period 1's end shown twice."""
    w = _classify(monkeypatch, synthetic_clip, SCENE_055, 27.5)
    assert (w["trim_start_s"], w["trim_end_s"]) == pytest.approx((0.0, 16.0))
    assert not w.get("sc_extra_windows")


def test_a_genuinely_strong_second_chance_still_gets_its_own_clip(monkeypatch, synthetic_clip):
    # first chance 0-6s, a second one 20-26s: far enough apart that their padded clips don't touch
    scripted = [(SC, .80), (SC, .78)] + [(OT, .20)] * 8 + [(SC, .75), (SC, .72), (OT, .20)]
    w = _classify(monkeypatch, synthetic_clip, scripted, 26.1)
    extra = w.get("sc_extra_windows")
    assert extra and len(extra) == 1
    assert extra[0]["confidence"] == pytest.approx(.75)
    assert extra[0]["trim_start_s"] > w["trim_end_s"], "a separate chance, not a repeat of the first"


def test_overlapping_strong_extra_is_folded_into_the_primary(monkeypatch, synthetic_clip):
    # two strong runs 6s apart: separate clusters, but their padded clips overlap heavily
    scripted = [(SC, .80), (SC, .78), (OT, .20), (OT, .20), (OT, .20), (OT, .20), (SC, .76), (SC, .74),
                (OT, .20), (OT, .20), (OT, .20), (OT, .20), (OT, .20)]
    w = _classify(monkeypatch, synthetic_clip, scripted, 26.1)
    assert not w.get("sc_extra_windows"), "overlapping clips must be one continuous clip"
    assert w["trim_end_s"] >= 17.0, "...that still covers the second run"


# ── pipeline: anchoring on detected shots ────────────────────────────────────

@pytest.fixture
def pipeline_mod(monkeypatch):
    import importlib
    mod = importlib.import_module("pipeline")
    monkeypatch.setattr(mod, "_get_clip_duration", lambda path: 26.1)
    return mod


def _anchor(mod, results):
    return mod._anchor_chances_on_shots(results, pre_s=5.0, post_s=3.0, max_len_s=20.0, ignore_before_s=1.0)


def test_chance_is_cut_around_the_detected_shot(pipeline_mod):
    seg = {"path": "a.mp4", "label": SC, "confidence": .70, "trim_start_s": 1.0, "trim_end_s": 26.1,
           "shot_times_s": [6.5]}
    _anchor(pipeline_mod, [seg])
    assert (seg["trim_start_s"], seg["trim_end_s"]) == pytest.approx((1.5, 9.5))
    assert seg["trim_anchor"] == "shot"


def test_scene_without_a_detected_shot_keeps_the_classifier_window(pipeline_mod):
    seg = {"path": "a.mp4", "label": SC, "confidence": .60, "trim_start_s": 3.0, "trim_end_s": 19.0, "shot_times_s": []}
    _anchor(pipeline_mod, [seg])
    assert (seg["trim_start_s"], seg["trim_end_s"]) == (3.0, 19.0)
    assert "trim_anchor" not in seg


def test_a_shot_reported_at_scene_start_does_not_move_the_clip(pipeline_mod):
    seg = {"path": "a.mp4", "label": SC, "confidence": .60, "trim_start_s": 3.0, "trim_end_s": 19.0,
           "shot_times_s": [0.07]}
    _anchor(pipeline_mod, [seg])
    assert (seg["trim_start_s"], seg["trim_end_s"]) == (3.0, 19.0)


def test_only_scoring_chances_are_anchored(pipeline_mod):
    seg = {"path": "a.mp4", "label": "goal", "confidence": .9, "shot_times_s": [6.5]}
    _anchor(pipeline_mod, [seg])
    assert "trim_start_s" not in seg


def test_far_apart_shots_become_separate_non_overlapping_clips(pipeline_mod):
    seg = {"path": "a.mp4", "label": SC, "confidence": .70, "shot_times_s": [5.0, 22.0]}
    _anchor(pipeline_mod, [seg])
    assert (seg["trim_start_s"], seg["trim_end_s"]) == pytest.approx((0.0, 8.0))
    extra = seg["sc_extra_windows"]
    assert len(extra) == 1
    assert (extra[0]["trim_start_s"], extra[0]["trim_end_s"]) == pytest.approx((17.0, 25.0))


# ── reel builder: last line of defence against repeated footage ──────────────

def test_reel_builder_never_emits_overlapping_clips():
    from src.assembly.reel_builder import _scoring_chance_leads
    seg = {"path": "055.mp4", "label": SC, "confidence": .71, "trim_start_s": 0.0, "trim_end_s": 25.0,
           "sc_extra_windows": [{"trim_start_s": 11.0, "trim_end_s": 27.0, "confidence": .62}]}
    leads = _scoring_chance_leads(seg, sc_min_confidence=0.45)
    assert len(leads) == 1
    assert (leads[0]["trim_start_s"], leads[0]["trim_end_s"]) == (0.0, 27.0)
    assert leads[0]["confidence"] == .71 and "sc_extra_windows" not in leads[0]


def test_reel_builder_keeps_separate_chances_chronological_and_drops_repeats_of_an_untrimmed_scene():
    from src.assembly.reel_builder import _scoring_chance_leads
    seg = {"path": "x.mp4", "label": SC, "confidence": .70, "trim_start_s": 0.0, "trim_end_s": 8.0,
           "sc_extra_windows": [{"trim_start_s": 17.0, "trim_end_s": 25.0, "confidence": .66}]}
    leads = _scoring_chance_leads(seg, sc_min_confidence=0.45)
    assert [(l["trim_start_s"], l["trim_end_s"]) for l in leads] == [(0.0, 8.0), (17.0, 25.0)]
    assert leads[1]["confidence"] == .66

    untrimmed = {"path": "y.mp4", "label": SC, "confidence": .70,
                 "sc_extra_windows": [{"trim_start_s": 17.0, "trim_end_s": 25.0, "confidence": .66}]}
    assert _scoring_chance_leads(untrimmed, 0.45) == [untrimmed], "the whole scene already plays everything"
