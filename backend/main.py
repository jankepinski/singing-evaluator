import io
import time

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from analyzer import (
    detect_beats_madmom,
    detect_onsets,
    detect_pitch_crepe,
    detect_pitch_yin,
    score_pitch_cents,
    score_rhythm_tolerance,
)

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
    # Input validation - check files are present
    if not reference_audio or not reference_audio.filename:
        raise HTTPException(status_code=422, detail="Missing required file: reference_audio")
    if not user_audio or not user_audio.filename:
        raise HTTPException(status_code=422, detail="Missing required file: user_audio")

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    try:
        # Read reference audio
        ref_bytes = await reference_audio.read()
        if len(ref_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Reference audio file too large (max 10MB)")
        if len(ref_bytes) == 0:
            raise HTTPException(status_code=400, detail="Reference audio file is empty")

        ref_audio, ref_sr = sf.read(io.BytesIO(ref_bytes))
        if ref_audio.ndim > 1:
            ref_audio = ref_audio.mean(axis=1)  # Convert to mono

        # Read user audio
        user_bytes = await user_audio.read()
        if len(user_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="User audio file too large (max 10MB)")
        if len(user_bytes) == 0:
            raise HTTPException(status_code=400, detail="User audio file is empty")

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

        # Flow B - AI (CREPE + madmom)
        start_time = time.time()
        ref_pitches_b = detect_pitch_crepe(ref_audio, target_sr)
        user_pitches_b = detect_pitch_crepe(user_audio_data, target_sr)
        ref_beats_b = detect_beats_madmom(ref_audio, target_sr)
        user_beats_b = detect_beats_madmom(user_audio_data, target_sr)

        pitch_score_b = score_pitch_cents(ref_pitches_b, user_pitches_b, offset_ms)
        rhythm_score_b = score_rhythm_tolerance(ref_beats_b, user_beats_b, offset_ms)
        flow_b_time = (time.time() - start_time) * 1000

        return {
            "offset_ms": offset_ms,
            "reference": {
                "pitch_curve": ref_pitches_a[:100],  # Reference pitch curve for Flow A
                "onsets": ref_onsets_a,              # Reference onsets
            },
            "flow_a": {
                "pitch_score": round(pitch_score_a, 1),
                "rhythm_score": round(rhythm_score_a, 1),
                "pitch_curve": user_pitches_a[:100],  # Limit for response size
                "onsets": user_onsets_a,
                "processing_time_ms": round(flow_a_time, 1),
            },
            "flow_b": {
                "pitch_score": round(pitch_score_b, 1),
                "rhythm_score": round(rhythm_score_b, 1),
                "pitch_curve": user_pitches_b[:100],
                "beats": user_beats_b,
                "processing_time_ms": round(flow_b_time, 1),
            },
            "debug": {
                "ref_pitches_count": len(ref_pitches_a),
                "ref_onsets_count": len(ref_onsets_a),
                "user_pitches_count": len(user_pitches_a),
                "user_onsets_count": len(user_onsets_a),
            }
        }
    except sf.LibsndfileError as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
