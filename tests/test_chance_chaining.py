"""
tests/test_chance_chaining.py

Two things the "two highlights in one segment" conversation surfaced:

1. pipeline._chain_adjacent_scoring_chances — when PySceneDetect cuts mid-
   action (e.g. an "OFFSIDE" banner triggers a scene change right as a chance
   develops), the build-up lands in one scoring_chance segment and the payoff
   in the very next one. Regression case: game 20260903192725, scene 124
   (defensemen passing, kept portion runs to its own scene's end) into scene
   126 (the interception/breakaway, kept portion starts at its own scene's
   beginning) — currently two separate clips with a hard cut and a dropped
   ~0.7s "OFFSIDE WARNING" scene between them, instead of one continuous play.

2. assembly.reel_builder.build_reel's scoring_chance branch, which
   pipeline.py now actually exercises for the first time via chain_sc_idx:
   it referenced an undefined `lead_src` (always dead before — chain_sc_idx
   was never set anywhere) and played every chained follow-up in full, which
   is right for a plain replay but wrong for a chained scoring-chance scene
   that carries its own trim_start_s/trim_end_s.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "apps" / "reel_builder"))

LEAD = REPO_ROOT / "tests" / "fixtures" / "sc_chain" / "lead.mp4"      # 6.0s
FOLLOW = REPO_ROOT / "tests" / "fixtures" / "sc_chain" / "follow.mp4"  # 4.0s

pytestmark = pytest.mark.skipif(
    not (LEAD.exists() and FOLLOW.exists()), reason="sc_chain fixture clips missing"
)


def _seg(path, **kw):
    kw.setdefault("label", "scoring_chance")
    kw.setdefault("confidence", 0.8)
    return {"path": str(path), **kw}


# ── Chaining decision (pure logic, no ffmpeg) ─────────────────────────────────

def test_chains_when_cut_lands_mid_action(monkeypatch):
    import pipeline

    monkeypatch.setattr(pipeline, "_get_clip_duration", lambda p: {
        str(LEAD): 12.6, str(FOLLOW): 32.2,
    }.get(str(p), 0.7))  # anything else (the unselected filler scene) is short

    lead = _seg(LEAD, trim_end_s=12.6)     # ran to its own scene's end
    filler = _seg(REPO_ROOT / "filler.mp4", label="other", confidence=0.1)
    follow = _seg(FOLLOW, trim_start_s=0.0, trim_end_s=11.0)  # starts at its own beginning

    results = [lead, filler, follow]
    pipeline._chain_adjacent_scoring_chances(results)

    assert follow["chain_sc_idx"] == 0


def test_no_chain_when_lead_ends_early(monkeypatch):
    """The lead's trim stopped well short of its own scene's end — it wasn't
    cut off by the scene boundary, the model just chose to end it there."""
    import pipeline

    monkeypatch.setattr(pipeline, "_get_clip_duration", lambda p: {
        str(LEAD): 12.6, str(FOLLOW): 32.2,
    }.get(str(p), 0.7))

    lead = _seg(LEAD, trim_end_s=8.0)  # nowhere near the 12.6s scene end
    follow = _seg(FOLLOW, trim_start_s=0.0, trim_end_s=11.0)

    pipeline._chain_adjacent_scoring_chances([lead, follow])
    assert "chain_sc_idx" not in follow


def test_no_chain_when_follow_starts_late(monkeypatch):
    import pipeline

    monkeypatch.setattr(pipeline, "_get_clip_duration", lambda p: {
        str(LEAD): 12.6, str(FOLLOW): 32.2,
    }.get(str(p), 0.7))

    lead = _seg(LEAD, trim_end_s=12.6)
    follow = _seg(FOLLOW, trim_start_s=4.0, trim_end_s=11.0)  # own action starts 4s in

    pipeline._chain_adjacent_scoring_chances([lead, follow])
    assert "chain_sc_idx" not in follow


def test_no_chain_across_a_real_gap(monkeypatch):
    """A long unselected stretch between the two — different, unrelated plays."""
    import pipeline

    monkeypatch.setattr(pipeline, "_get_clip_duration", lambda p: {
        str(LEAD): 12.6, str(FOLLOW): 32.2,
    }.get(str(p), 25.0))  # the "in-between" scene is a real 25s of other play

    lead = _seg(LEAD, trim_end_s=12.6)
    filler = _seg(REPO_ROOT / "filler.mp4", label="other", confidence=0.1)
    follow = _seg(FOLLOW, trim_start_s=0.0, trim_end_s=11.0)

    pipeline._chain_adjacent_scoring_chances([lead, filler, follow])
    assert "chain_sc_idx" not in follow


def test_no_chain_across_a_structural_boundary(monkeypatch):
    """Even with matching edges, never bridge over a period/game boundary."""
    import pipeline

    monkeypatch.setattr(pipeline, "_get_clip_duration", lambda p: {
        str(LEAD): 12.6, str(FOLLOW): 32.2,
    }.get(str(p), 0.7))

    lead = _seg(LEAD, trim_end_s=12.6)
    follow = _seg(FOLLOW, trim_start_s=0.0, trim_end_s=11.0, period_start=True)

    pipeline._chain_adjacent_scoring_chances([lead, follow])
    assert "chain_sc_idx" not in follow


# ── build_reel actually assembling a chained pair (real ffmpeg, tiny clips) ──

def test_build_reel_merges_a_chained_pair(tmp_path):
    from src.assembly.reel_builder import build_reel

    lead = _seg(LEAD, trim_start_s=1.0, trim_end_s=6.0)       # 5.0s kept
    follow = _seg(FOLLOW, trim_start_s=0.0, trim_end_s=3.0, chain_sc_idx=0)  # 3.0s kept

    out = tmp_path / "reel.mp4"
    build_reel(
        segments=[lead, follow],
        output_path=out,
        add_overlays=False,
        min_confidence=0.1,
        sc_min_confidence=0.1,
    )

    assert out.exists()
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True,
    )
    dur = float(probe.stdout.strip())
    # One continuous clip covering both trims (5.0 + 3.0 = 8.0s), not just the lead.
    assert dur > 7.0, f"expected ~8s merged clip, got {dur:.1f}s — follow-up got dropped?"
