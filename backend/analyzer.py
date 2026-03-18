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
