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
