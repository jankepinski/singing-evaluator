import numpy as np
from analyzer import detect_pitch_yin, detect_onsets


def test_detect_pitch_yin():
    """Test pitch detection with 440Hz sine wave (A4)."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    pitches = detect_pitch_yin(audio, sr)

    assert len(pitches) > 0
    # Should detect approximately 440Hz
    assert abs(np.mean([p["freq"] for p in pitches if p["freq"] > 0]) - 440) < 10


def test_detect_onsets():
    """Test onset detection with square wave bursts."""
    sr = 22050
    audio = np.zeros(sr * 2)
    # Add silence at start so onset can be detected, then sound at 0.5s and 1.5s
    audio[sr//2:sr//2+sr//4] = 0.5  # sound at 0.5s
    audio[sr+sr//2:sr+sr//2+sr//4] = 0.5  # another sound at 1.5s

    onsets = detect_onsets(audio, sr)

    assert len(onsets) >= 1
    # Should detect around 0.5s (within tolerance)
    assert any(abs(o - 0.5) < 0.2 for o in onsets)


def test_score_pitch_cents():
    from analyzer import score_pitch_cents

    ref_pitches = [
        {"time": 0.0, "freq": 440.0},
        {"time": 0.5, "freq": 493.88},  # B4
    ]
    user_pitches = [
        {"time": 0.0, "freq": 445.0},   # slightly sharp A4 (~19.6 cents)
        {"time": 0.5, "freq": 493.88},  # perfect B4
    ]

    score = score_pitch_cents(ref_pitches, user_pitches, offset_ms=0)

    assert 0 <= score <= 100
    # Mean error: (19.6 + 0) / 2 = 9.8 cents
    # Score: 100 - (9.8/0.5) = ~80.4
    assert score > 75 and score < 85


def test_score_rhythm_tolerance():
    from analyzer import score_rhythm_tolerance

    ref_onsets = [0.0, 0.5, 1.0, 1.5]
    user_onsets = [0.02, 0.52, 1.05, 2.0]  # first 3 within tolerance, last missed

    score = score_rhythm_tolerance(ref_onsets, user_onsets, offset_ms=0)

    assert score == 75.0  # 3/4 matched
