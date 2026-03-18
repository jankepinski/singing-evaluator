import io
import time

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from analyzer import detect_onsets, detect_pitch_yin, score_pitch_cents, score_rhythm_tolerance

app = FastAPI(title="Singing Evaluator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    reference_audio: UploadFile = File(...),
    user_audio: UploadFile = File(...),
    offset_ms: float = Form(0.0)
):
    try:
        # Read reference audio
        ref_bytes = await reference_audio.read()
        ref_audio, ref_sr = sf.read(io.BytesIO(ref_bytes))
        if ref_audio.ndim > 1:
            ref_audio = ref_audio.mean(axis=1)  # Convert to mono

        # Read user audio
        user_bytes = await user_audio.read()
        user_audio_data, user_sr = sf.read(io.BytesIO(user_bytes))
        if user_audio_data.ndim > 1:
            user_audio_data = user_audio_data.mean(axis=1)

        # Resample to common rate if needed
        target_sr = 22050
        if ref_sr != target_sr:
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=target_sr)
        if user_sr != target_sr:
            user_audio_data = librosa.resample(user_audio_data, orig_sr=user_sr, target_sr=target_sr)

        # Flow A - Deterministic
        start_time = time.time()
        ref_pitches_a = detect_pitch_yin(ref_audio, target_sr)
        user_pitches_a = detect_pitch_yin(user_audio_data, target_sr)
        ref_onsets_a = detect_onsets(ref_audio, target_sr)
        user_onsets_a = detect_onsets(user_audio_data, target_sr)

        pitch_score_a = score_pitch_cents(ref_pitches_a, user_pitches_a, offset_ms)
        rhythm_score_a = score_rhythm_tolerance(ref_onsets_a, user_onsets_a, offset_ms)
        flow_a_time = (time.time() - start_time) * 1000

        return {
            "offset_ms": offset_ms,
            "flow_a": {
                "pitch_score": round(pitch_score_a, 1),
                "rhythm_score": round(rhythm_score_a, 1),
                "pitch_curve": user_pitches_a[:100],  # Limit for response size
                "onsets": user_onsets_a,
                "processing_time_ms": round(flow_a_time, 1),
            },
            "flow_b": {
                "pitch_score": 0,
                "rhythm_score": 0,
                "pitch_curve": [],
                "beats": [],
                "processing_time_ms": 0,
                "status": "not_implemented"
            },
            "debug": {
                "ref_pitches_count": len(ref_pitches_a),
                "ref_onsets_count": len(ref_onsets_a),
                "user_pitches_count": len(user_pitches_a),
                "user_onsets_count": len(user_onsets_a),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
