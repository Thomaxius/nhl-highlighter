"""
tests/test_audio_analyzer.py

Unit tests for audio spike detection logic (no GPU / model required).
"""

import numpy as np
import pytest


def test_score_segments_no_spikes():
    from src.preprocessing.audio_analyzer import score_segments_by_audio

    segments = [{"start_s": 0.0, "end_s": 5.0, "score": 0.8}]
    result = score_segments_by_audio(segments, spikes=[])
    assert result[0]["score"] == pytest.approx(0.8)


def test_score_segments_with_overlap():
    from src.preprocessing.audio_analyzer import score_segments_by_audio

    segments = [{"start_s": 10.0, "end_s": 15.0, "score": 0.5}]
    spikes = [{"start_s": 12.0, "end_s": 14.0, "energy": 0.2}]
    result = score_segments_by_audio(segments, spikes, window_s=1.0)
    # Score should be boosted
    assert result[0]["score"] > 0.5


def test_score_segments_no_overlap():
    from src.preprocessing.audio_analyzer import score_segments_by_audio

    segments = [{"start_s": 0.0, "end_s": 5.0, "score": 0.5}]
    spikes = [{"start_s": 100.0, "end_s": 102.0, "energy": 0.9}]
    result = score_segments_by_audio(segments, spikes, window_s=1.0)
    assert result[0]["score"] == pytest.approx(0.5)
