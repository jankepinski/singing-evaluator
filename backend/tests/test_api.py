def test_analyze_endpoint():
    import io
    import wave
    import numpy as np
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # Create a simple WAV file with 440Hz sine wave
    def create_wav(freq=440, duration=1.0):
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
