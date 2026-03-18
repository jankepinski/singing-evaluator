import io
import wave
import numpy as np
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def create_wav(freq=440, duration=1.0):
    """Create a simple WAV file with a sine wave at specified frequency."""
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(audio.tobytes())
    buf.seek(0)
    return buf


def test_analyze_endpoint():
    ref_file = create_wav(440)
    user_file = create_wav(445)  # slightly sharp

    response = client.post(
        "/api/analyze",
        files={
            "reference_audio": ("ref.wav", ref_file, "audio/wav"),
            "user_audio": ("user.wav", user_file, "audio/wav"),
        },
        data={"offset_ms": "0"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "flow_a" in data
    assert "pitch_score" in data["flow_a"]
    assert "rhythm_score" in data["flow_a"]

    # Validate score ranges (0 <= score <= 100)
    pitch_score = data["flow_a"]["pitch_score"]
    rhythm_score = data["flow_a"]["rhythm_score"]
    assert 0 <= pitch_score <= 100, f"Pitch score {pitch_score} is out of range [0, 100]"
    assert 0 <= rhythm_score <= 100, f"Rhythm score {rhythm_score} is out of range [0, 100]"

    # Check offset_ms is returned correctly
    assert data["offset_ms"] == 0.0

    # Check debug info is present
    assert "debug" in data
    assert "ref_pitches_count" in data["debug"]
    assert "user_onsets_count" in data["debug"]


def test_analyze_endpoint_with_offset():
    ref_file = create_wav(440)
    user_file = create_wav(440)

    response = client.post(
        "/api/analyze",
        files={
            "reference_audio": ("ref.wav", ref_file, "audio/wav"),
            "user_audio": ("user.wav", user_file, "audio/wav"),
        },
        data={"offset_ms": "500"}  # 500ms offset
    )

    assert response.status_code == 200
    data = response.json()
    assert data["offset_ms"] == 500.0
    assert "flow_a" in data
    assert "pitch_score" in data["flow_a"]
    assert "rhythm_score" in data["flow_a"]

    # Validate score ranges
    pitch_score = data["flow_a"]["pitch_score"]
    rhythm_score = data["flow_a"]["rhythm_score"]
    assert 0 <= pitch_score <= 100, f"Pitch score {pitch_score} is out of range [0, 100]"
    assert 0 <= rhythm_score <= 100, f"Rhythm score {rhythm_score} is out of range [0, 100]"


def test_analyze_endpoint_empty_file():
    empty_file = io.BytesIO(b"")
    ref_file = create_wav(440)

    response = client.post(
        "/api/analyze",
        files={
            "reference_audio": ("ref.wav", ref_file, "audio/wav"),
            "user_audio": ("empty.wav", empty_file, "audio/wav"),
        },
        data={"offset_ms": "0"}
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "empty" in data["detail"].lower()


def test_analyze_endpoint_missing_file():
    ref_file = create_wav(440)

    response = client.post(
        "/api/analyze",
        files={
            "reference_audio": ("ref.wav", ref_file, "audio/wav"),
        },
        data={"offset_ms": "0"}
    )

    assert response.status_code == 422


def test_analyze_endpoint_identical_audio():
    ref_file = create_wav(440)
    user_file = create_wav(440)  # Same frequency - should give high scores

    response = client.post(
        "/api/analyze",
        files={
            "reference_audio": ("ref.wav", ref_file, "audio/wav"),
            "user_audio": ("user.wav", user_file, "audio/wav"),
        },
        data={"offset_ms": "0"}
    )

    assert response.status_code == 200
    data = response.json()
    pitch_score = data["flow_a"]["pitch_score"]
    # Identical audio should have high pitch score
    assert pitch_score >= 80, f"Expected high pitch score for identical audio, got {pitch_score}"
