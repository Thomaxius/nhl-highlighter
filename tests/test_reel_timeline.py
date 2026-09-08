"""
tests/test_reel_timeline.py

Unit tests for the reel timeline writer (assembly/reel_builder.py) — the
`<reel>.timeline.txt` / `.timeline.json` sidecar that says which stretch of the
finished reel is a goal, scoring chance, transition, etc. Pure Python, no ffmpeg.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apps.reel_builder.src.assembly.reel_builder import (  # noqa: E402
    _describe_group,
    _scene_label,
    _write_reel_timeline,
)


def _seg(scene, **kw):
    kw.setdefault("path", f"/x/NHL 25_20260903192725_norm-scene-{scene}.mp4")
    kw.setdefault("label", "other")
    kw.setdefault("confidence", 0.9)
    return kw


def test_scene_label_extracts_number():
    assert _scene_label(_seg("082")) == "082"
    assert _scene_label({"path": "/x/weird_name.mp4"}) == "weird_name"


def test_describe_group_kinds():
    assert _describe_group([_seg("082", label="goal", banner_detected=True)]) == ("goal", "banner-confirmed")
    assert _describe_group([_seg("127", label="goal", inferred_goal=True)]) == ("goal", "faceoff-pattern inferred")
    assert _describe_group([_seg("027", label="scoring_chance")])[0] == "scoring_chance"
    assert _describe_group([{"intro": True, "path": ""}])[0] == "intro"
    assert _describe_group([{"period_transition": True, "to_period": 3, "path": ""}]) == ("transition", "3RD PERIOD")
    assert _describe_group([{"period_transition": True, "to_period": 4, "path": ""}]) == ("transition", "OVERTIME")
    assert _describe_group([_seg("192", label="other", game_end=True)])[0] == "game_end"
    assert _describe_group([_seg("177", label="other", regulation_end=True)])[0] == "end_of_regulation"


def test_write_reel_timeline(tmp_path):
    out = tmp_path / "reel_highlights.mp4"
    groups_and_durations = [
        ([{"intro": True, "path": "", "period_number": 1}], 22.8),
        ([_seg("027", label="scoring_chance", period_number=1, confidence=0.81)], 8.0),
        ([{"period_transition": True, "to_period": 2, "path": "", "period_number": 2}], 6.0),
        ([_seg("082", label="goal", banner_detected=True, period_number=2),
          _seg("083", label="goal_replay", period_number=2)], 44.8),
    ]
    _write_reel_timeline(groups_and_durations, out)

    txt = (tmp_path / "reel_highlights.timeline.txt").read_text()
    assert "00:00" in txt and "3RD PERIOD" not in txt
    assert "2ND PERIOD" in txt
    assert "scene 082" in txt or "scenes 082" in txt

    data = json.loads((tmp_path / "reel_highlights.timeline.json").read_text())
    assert data["total_s"] == 81.6
    segs = data["segments"]
    assert segs[0]["kind"] == "intro" and segs[0]["start_tc"] == "00:00"
    assert segs[1]["kind"] == "scoring_chance" and segs[1]["confidence"] == 0.81
    assert segs[2]["kind"] == "transition" and segs[2]["detail"] == "2ND PERIOD"
    # goal group starts right after the 6s transition: 22.8 + 8.0 + 6.0 = 36.8
    assert segs[3]["kind"] == "goal" and segs[3]["start_s"] == 36.8
    assert segs[3]["scenes"] == ["082", "083"]
