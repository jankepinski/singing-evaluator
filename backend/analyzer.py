import math

import librosa
import numpy as np


def detect_pitch_yin(audio: np.ndarray, sr: int, fmin=50, fmax=2000):
    """Detect pitch using YIN algorithm via librosa."""
    pitches = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=sr)
    # Filter out unvoiced (0 values)
    times = librosa.times_like(pitches, sr=sr)
    return [
        {"time": float(t), "freq": float(f)}
        for t, f in zip(times, pitches) if f > 0
    ]


def detect_onsets(audio: np.ndarray, sr: int):
    """Detect note onsets using librosa."""
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr, wait=3, pre_avg=3, post_avg=3, pre_max=3, post_max=3
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return [float(t) for t in onset_times]


def score_pitch_cents(ref_pitches, user_pitches, offset_ms=0):
    """
    Score pitch accuracy in cents.
    50 cents (half semitone) = 0 points.
    """
    if not ref_pitches or not user_pitches:
        return 0.0

    offset = offset_ms / 1000.0
    errors = []

    for ref in ref_pitches:
        ref_time = ref["time"] + offset
        ref_freq = ref["freq"]

        # Find closest user pitch in time (within 100ms window)
        closest = None
        min_time_diff = float('inf')

        for user in user_pitches:
            time_diff = abs(user["time"] - ref_time)
            if time_diff < min_time_diff and time_diff < 0.1:  # within 100ms
                min_time_diff = time_diff
                closest = user

        if closest:
            # Calculate error in cents: 1200 * log2(f2/f1)
            error_cents = 1200 * abs(math.log2(closest["freq"] / ref_freq))
            errors.append(error_cents)

    if not errors:
        return 0.0

    mean_error = sum(errors) / len(errors)
    # 50 cents = 0 score
    score = max(0, 100 - (mean_error / 0.5))
    return score


TOLERANCE_MS = 75  # 75ms tolerance for rhythm matching


def score_rhythm_tolerance(ref_onsets, user_onsets, offset_ms=0):
    """
    Score rhythm by counting matched onsets within tolerance.
    """
    if not ref_onsets:
        return 0.0

    tolerance = TOLERANCE_MS / 1000.0
    offset = offset_ms / 1000.0

    matched = 0
    for ref in ref_onsets:
        ref_time = ref + offset
        # Check if any user onset is within tolerance
        if any(abs(ref_time - user) < tolerance for user in user_onsets):
            matched += 1

    return (matched / len(ref_onsets)) * 100
