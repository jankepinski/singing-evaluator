# Singing Evaluator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a web app that compares deterministic vs AI-based pitch and rhythm analysis for singing evaluation.

**Architecture:** Next.js frontend for UI and recording, FastAPI backend for audio analysis. Two analysis flows: Flow A (librosa.yin + onset_detect) vs Flow B (CREPE + madmom). Pitch scored in cents, rhythm by onset tolerance matching.

**Tech Stack:** Next.js 15, React, Tailwind, Recharts, FastAPI, Python 3.11, librosa, CREPE, madmom

---

## Prerequisites

- Python 3.11+ installed
- Node.js 20+ installed
- Git initialized in project

---

## Phase 1: Backend Setup

### Task 1: Create FastAPI project structure

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/main.py`
- Create: `backend/analyzer.py`

**Step 1: Write requirements.txt**

```txt
fastapi==0.115.0
uvicorn==0.32.0
python-multipart==0.0.17
librosa==0.10.2
madmom==0.17.1
crepe==0.0.15
numpy==1.26.4
soundfile==0.12.1
```

**Step 2: Create basic main.py**

```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

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
```

**Step 3: Test backend starts**

Run: `cd backend && pip install -r requirements.txt && uvicorn main:app --reload`
Expected: Server starts on http://localhost:8000

**Step 4: Commit**

```bash
git add backend/
git commit -m "feat: initial FastAPI backend setup"
```

---

### Task 2: Implement Flow A (Deterministic) analyzer

**Files:**
- Modify: `backend/analyzer.py`
- Create: `backend/tests/test_analyzer.py`

**Step 1: Write failing test**

```python
def test_detect_pitch_yin():
    import numpy as np
    from analyzer import detect_pitch_yin
    
    # Create 440Hz sine wave (A4)
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    pitches = detect_pitch_yin(audio, sr)
    
    assert len(pitches) > 0
    # Should detect approximately 440Hz
    assert abs(np.mean([p for p in pitches if p > 0]) - 440) < 10
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_analyzer.py -v`
Expected: FAIL - "module analyzer not found"

**Step 3: Implement detect_pitch_yin**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_analyzer.py::test_detect_pitch_yin -v`
Expected: PASS

**Step 5: Write test for onset detection**

```python
def test_detect_onsets():
    import numpy as np
    from analyzer import detect_onsets
    
    # Create audio with clear onsets (square wave bursts)
    sr = 22050
    audio = np.zeros(sr * 2)
    audio[:sr//4] = 0.5  # first quarter second sound
    audio[sr:sr+sr//4] = 0.5  # another sound at 1s
    
    onsets = detect_onsets(audio, sr)
    
    assert len(onsets) >= 1
    # Should detect around 0s and 1s
    assert any(abs(o - 0.0) < 0.1 for o in onsets)
```

**Step 6: Implement detect_onsets**

```python
def detect_onsets(audio: np.ndarray, sr: int):
    """Detect note onsets using librosa."""
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr, wait=3, pre_avg=3, post_avg=3, pre_max=3, post_max=3
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return [float(t) for t in onset_times]
```

**Step 7: Run tests**

Run: `cd backend && python -m pytest tests/test_analyzer.py -v`
Expected: Both tests PASS

**Step 8: Commit**

```bash
git add backend/
git commit -m "feat: implement Flow A pitch and onset detection"
```

---

### Task 3: Implement scoring algorithms

**Files:**
- Modify: `backend/analyzer.py`
- Modify: `backend/tests/test_analyzer.py`

**Step 1: Write failing test for pitch scoring**

```python
def test_score_pitch_cents():
    from analyzer import score_pitch_cents
    
    ref_pitches = [
        {"time": 0.0, "freq": 440.0},
        {"time": 0.5, "freq": 493.88},  # B4
    ]
    user_pitches = [
        {"time": 0.0, "freq": 445.0},   # slightly sharp A4
        {"time": 0.5, "freq": 493.88},  # perfect B4
    ]
    
    score = score_pitch_cents(ref_pitches, user_pitches, offset_ms=0)
    
    assert 0 <= score <= 100
    # 445 vs 440 = ~19.6 cents, so score should be around 100 - (19.6/0.5) = ~61
    assert score > 50 and score < 70
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_analyzer.py::test_score_pitch_cents -v`
Expected: FAIL

**Step 3: Implement score_pitch_cents**

```python
import math

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
        
        # Find closest user pitch in time
        closest = None
        min_time_diff = float('inf')
        
        for user in user_pitches:
            time_diff = abs(user["time"] - ref_time)
            if time_diff < min_time_diff and time_diff < 0.1:  # within 100ms
                min_time_diff = time_diff
                closest = user
        
        if closest:
            # Calculate error in cents
            error_cents = 1200 * abs(math.log2(closest["freq"] / ref_freq))
            errors.append(error_cents)
    
    if not errors:
        return 0.0
    
    mean_error = sum(errors) / len(errors)
    # 50 cents = 0 score
    score = max(0, 100 - (mean_error / 0.5))
    return score
```

**Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_analyzer.py::test_score_pitch_cents -v`
Expected: PASS

**Step 5: Write test for rhythm scoring**

```python
def test_score_rhythm_tolerance():
    from analyzer import score_rhythm_tolerance
    
    ref_onsets = [0.0, 0.5, 1.0, 1.5]
    user_onsets = [0.02, 0.52, 1.05, 2.0]  # first 3 within tolerance, last missed
    
    score = score_rhythm_tolerance(ref_onsets, user_onsets, offset_ms=0)
    
    assert score == 75.0  # 3/4 matched
```

**Step 6: Implement score_rhythm_tolerance**

```python
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
```

**Step 7: Run all tests**

Run: `cd backend && python -m pytest tests/test_analyzer.py -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add backend/
git commit -m "feat: implement pitch (cents) and rhythm (tolerance) scoring"
```

---

### Task 4: Implement Flow A complete endpoint

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/analyzer.py`
- Create: `backend/tests/test_api.py`

**Step 1: Write test for analyze endpoint**

```python
def test_analyze_endpoint():
    import io
    import wave
    import struct
    from fastapi.testclient import TestClient
    from main import app
    
    client = TestClient(app)
    
    # Create a simple WAV file
    def create_wav(freq=440, duration=1.0):
        import numpy as np
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
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_api.py::test_analyze_endpoint -v`
Expected: FAIL - endpoint not implemented

**Step 3: Implement analyze endpoint**

```python
import io
import soundfile as sf
import numpy as np
from fastapi import HTTPException

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
        
        from analyzer import detect_pitch_yin, detect_onsets, score_pitch_cents, score_rhythm_tolerance
        import time
        
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
```

**Step 4: Run test**

Run: `cd backend && python -m pytest tests/test_api.py::test_analyze_endpoint -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/
git commit -m "feat: implement Flow A analyze endpoint"
```

---

### Task 5: Implement Flow B (AI) with CREPE and madmom

**Files:**
- Modify: `backend/analyzer.py`
- Modify: `backend/tests/test_analyzer.py`

**Step 1: Write failing test for CREPE pitch detection**

```python
def test_detect_pitch_crepe():
    import numpy as np
    from analyzer import detect_pitch_crepe
    
    # Create 440Hz sine wave
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    pitches = detect_pitch_crepe(audio, sr)
    
    assert len(pitches) > 0
    # CREPE should detect approximately 440Hz
    avg_pitch = np.mean([p["freq"] for p in pitches])
    assert abs(avg_pitch - 440) < 5  # CREPE is more accurate than YIN
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_analyzer.py::test_detect_pitch_crepe -v`
Expected: FAIL

**Step 3: Implement detect_pitch_crepe**

```python
import crepe

def detect_pitch_crepe(audio: np.ndarray, sr: int):
    """Detect pitch using CREPE deep learning model."""
    time_stamps, frequencies, confidence, activation = crepe.predict(
        audio, sr, viterbi=True
    )
    
    # Filter low confidence predictions
    results = []
    for t, f, c in zip(time_stamps, frequencies, confidence):
        if c > 0.5:  # Confidence threshold
            results.append({"time": float(t), "freq": float(f)})
    
    return results
```

**Step 4: Run test (will take longer due to model loading)**

Run: `cd backend && python -m pytest tests/test_analyzer.py::test_detect_pitch_crepe -v -s`
Expected: PASS (may take 10-30s first time for model download)

**Step 5: Write test for madmom beat detection**

```python
def test_detect_beats_madmom():
    import numpy as np
    from analyzer import detect_beats_madmom
    
    # Create audio with clear beats
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # Create impulse train at 0.5s and 1.0s
    audio = np.zeros_like(t)
    audio[int(0.5*sr):int(0.5*sr)+1000] = 0.5
    audio[int(1.0*sr):int(1.0*sr)+1000] = 0.5
    
    beats = detect_beats_madmom(audio, sr)
    
    assert len(beats) >= 1
```

**Step 6: Implement detect_beats_madmom**

```python
from madmom.features.beats import DBNDownBeatTrackingProcessor
from madmom.features.onsets import CNNOnsetProcessor

def detect_beats_madmom(audio: np.ndarray, sr: int):
    """Detect beats using madmom DBN downbeat tracking."""
    # For simplicity, use onset detection first
    onset_proc = CNNOnsetProcessor()
    
    # madmom expects file path or different format, so we use simpler approach
    # Actually use RNNOnsetProcessor which works with audio directly
    from madmom.features.onsets import RNNOnsetProcessor
    
    act = RNNOnsetProcessor()(audio, sr)
    from madmom.features.onsets import PeakPickingProcessor
    
    peak_picking = PeakPickingProcessor(fps=100, threshold=0.5)
    beats = peak_picking(act)
    
    return [float(b) for b in beats]
```

**Step 7: Run test**

Run: `cd backend && python -m pytest tests/test_analyzer.py::test_detect_beats_madmom -v`
Expected: PASS

**Step 8: Update analyze endpoint to include Flow B**

Modify `/api/analyze` endpoint in `main.py` to also run Flow B analysis and return both results.

**Step 9: Run all tests**

Run: `cd backend && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 10: Commit**

```bash
git add backend/
git commit -m "feat: implement Flow B with CREPE and madmom"
```

---

## Phase 2: Frontend Setup

### Task 6: Initialize Next.js project

**Files:**
- Shell commands to run in project root

**Step 1: Create Next.js app**

Run:
```bash
echo "my-app" | npx shadcn@latest init --yes --template next --base-color zinc
```
Expected: Next.js project created in `my-app/`

**Step 2: Install additional dependencies**

Run:
```bash
cd my-app && npm install recharts
```

**Step 3: Test dev server starts**

Run: `cd my-app && npm run dev`
Expected: Server starts on http://localhost:3000

**Step 4: Commit**

```bash
git add my-app/
git commit -m "feat: initialize Next.js project with shadcn"
```

---

### Task 7: Create AudioUploader component

**Files:**
- Create: `my-app/components/audio-uploader.tsx`
- Create: `my-app/app/page.tsx` (modify)

**Step 1: Write AudioUploader component**

```tsx
"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

interface AudioUploaderProps {
  onUpload: (file: File) => void;
}

export function AudioUploader({ onUpload }: AudioUploaderProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0]);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.flac', '.ogg']
    },
    maxFiles: 1
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
        isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"
      }`}
    >
      <input {...getInputProps()} />
      {isDragActive ? (
        <p>Drop the audio file here...</p>
      ) : (
        <div>
          <p className="text-lg font-medium">Upload reference audio</p>
          <p className="text-sm text-gray-500 mt-2">
            Drag & drop or click to select (WAV, MP3, FLAC)
          </p>
        </div>
      )}
    </div>
  );
}
```

**Step 2: Install react-dropzone**

Run: `cd my-app && npm install react-dropzone`

**Step 3: Update page.tsx to test component**

```tsx
import { AudioUploader } from "@/components/audio-uploader";

export default function Home() {
  return (
    <main className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-8">Singing Evaluator</h1>
      <AudioUploader onUpload={(file) => console.log("Uploaded:", file.name)} />
    </main>
  );
}
```

**Step 4: Test in browser**

Open http://localhost:3000
Expected: See upload box, can drag/drop or click

**Step 5: Commit**

```bash
git add my-app/
git commit -m "feat: add AudioUploader component"
```

---

### Task 8: Create Recorder component with offset tracking

**Files:**
- Create: `my-app/components/recorder.tsx`
- Create: `my-app/hooks/use-audio-context.ts`

**Step 1: Write useAudioContext hook**

```tsx
"use client";

import { useState, useCallback, useRef } from "react";

interface RecordingResult {
  blob: Blob;
  offsetMs: number;
}

export function useRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playbackStartTimeRef = useRef<number>(0);
  const recordingStartTimeRef = useRef<number>(0);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async (onComplete: (result: RecordingResult) => void) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      audioContextRef.current = new AudioContext();
      playbackStartTimeRef.current = audioContextRef.current.currentTime;
      
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };
      
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        const offsetMs = (recordingStartTimeRef.current - playbackStartTimeRef.current) * 1000;
        onComplete({ blob, offsetMs });
        stream.getTracks().forEach(track => track.stop());
      };
      
      // Mark when user actually starts recording
      recordingStartTimeRef.current = audioContextRef.current.currentTime;
      
      mediaRecorderRef.current.start(100); // Collect every 100ms
      setIsRecording(true);
      
      // Update recording time
      const interval = setInterval(() => {
        if (audioContextRef.current) {
          const elapsed = audioContextRef.current.currentTime - recordingStartTimeRef.current;
          setRecordingTime(elapsed);
        }
      }, 100);
      
      // Store interval for cleanup
      (mediaRecorderRef.current as any).intervalId = interval;
      
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone. Please check permissions.");
    }
  }, []);
  
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      const interval = (mediaRecorderRef.current as any).intervalId;
      if (interval) clearInterval(interval);
      
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setRecordingTime(0);
    }
  }, []);

  return {
    isRecording,
    recordingTime,
    startRecording,
    stopRecording,
    playbackStartTime: playbackStartTimeRef.current,
    recordingStartTime: recordingStartTimeRef.current,
  };
}
```

**Step 2: Write Recorder component**

```tsx
"use client";

import { useRecorder } from "@/hooks/use-recorder";
import { Button } from "@/components/ui/button";

interface RecorderProps {
  referenceAudioUrl: string | null;
  onRecordingComplete: (blob: Blob, offsetMs: number) => void;
}

export function Recorder({ referenceAudioUrl, onRecordingComplete }: RecorderProps) {
  const { isRecording, recordingTime, startRecording, stopRecording } = useRecorder();

  const handleStart = async () => {
    if (!referenceAudioUrl) {
      alert("Please upload reference audio first");
      return;
    }
    
    // Start playback
    const audio = new Audio(referenceAudioUrl);
    await audio.play();
    
    // Start recording after a small delay to sync
    await startRecording(({ blob, offsetMs }) => {
      audio.pause();
      onRecordingComplete(blob, offsetMs);
    });
  };

  const handleStop = () => {
    stopRecording();
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="border rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Record Your Singing</h2>
      
      {!referenceAudioUrl && (
        <p className="text-gray-500">Upload reference audio first to enable recording</p>
      )}
      
      {referenceAudioUrl && (
        <div className="flex items-center gap-4">
          {!isRecording ? (
            <Button onClick={handleStart} size="lg" className="bg-red-500 hover:bg-red-600">
              ● Start Recording
            </Button>
          ) : (
            <Button onClick={handleStop} size="lg" variant="outline">
              ⏹ Stop ({formatTime(recordingTime)})
            </Button>
          )}
          
          {isRecording && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
              <span className="text-red-500">Recording...</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

**Step 3: Update page.tsx to integrate**

```tsx
"use client";

import { useState } from "react";
import { AudioUploader } from "@/components/audio-uploader";
import { Recorder } from "@/components/recorder";

export default function Home() {
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [referenceUrl, setReferenceUrl] = useState<string | null>(null);

  const handleUpload = (file: File) => {
    setReferenceFile(file);
    setReferenceUrl(URL.createObjectURL(file));
  };

  const handleRecordingComplete = (blob: Blob, offsetMs: number) => {
    console.log("Recording complete:", { blob, offsetMs });
    // TODO: Send to backend
  };

  return (
    <main className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-8">Singing Evaluator</h1>
      
      <div className="space-y-6">
        <AudioUploader onUpload={handleUpload} />
        
        {referenceFile && (
          <p className="text-sm text-green-600">
            ✓ Loaded: {referenceFile.name}
          </p>
        )}
        
        <Recorder
          referenceAudioUrl={referenceUrl}
          onRecordingComplete={handleRecordingComplete}
        />
      </div>
    </main>
  );
}
```

**Step 4: Test in browser**

Open http://localhost:3000
Expected: Can upload audio, then see recorder. Clicking record should start playback and recording.

**Step 5: Commit**

```bash
git add my-app/
git commit -m "feat: add Recorder component with offset tracking"
```

---

### Task 9: Create API integration and results display

**Files:**
- Create: `my-app/lib/api.ts`
- Create: `my-app/components/results-display.tsx`
- Modify: `my-app/app/page.tsx`

**Step 1: Write API client**

```ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface AnalysisResult {
  offset_ms: number;
  flow_a: {
    pitch_score: number;
    rhythm_score: number;
    pitch_curve: { time: number; freq: number }[];
    onsets: number[];
    processing_time_ms: number;
  };
  flow_b: {
    pitch_score: number;
    rhythm_score: number;
    pitch_curve: { time: number; freq: number }[];
    beats: number[];
    processing_time_ms: number;
    status?: string;
  };
  debug: {
    ref_pitches_count: number;
    ref_onsets_count: number;
    user_pitches_count: number;
    user_onsets_count: number;
  };
}

export async function analyzeAudio(
  referenceFile: File,
  userBlob: Blob,
  offsetMs: number
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("reference_audio", referenceFile);
  formData.append("user_audio", userBlob, "recording.wav");
  formData.append("offset_ms", offsetMs.toString());

  const response = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Analysis failed: ${error}`);
  }

  return response.json();
}
```

**Step 2: Write ResultsDisplay component**

```tsx
"use client";

import { AnalysisResult } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface ResultsDisplayProps {
  result: AnalysisResult;
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Flow A - Deterministic */}
      <Card>
        <CardHeader>
          <CardTitle>Flow A: Deterministic</CardTitle>
          <p className="text-sm text-gray-500">librosa.yin + onset_detect</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex justify-between mb-2">
              <span>Pitch Accuracy</span>
              <span className="font-bold">{result.flow_a.pitch_score}%</span>
            </div>
            <Progress value={result.flow_a.pitch_score} />
          </div>
          
          <div>
            <div className="flex justify-between mb-2">
              <span>Rhythm Accuracy</span>
              <span className="font-bold">{result.flow_a.rhythm_score}%</span>
            </div>
            <Progress value={result.flow_a.rhythm_score} />
          </div>
          
          <p className="text-xs text-gray-400">
            Processed in {result.flow_a.processing_time_ms}ms
          </p>
        </CardContent>
      </Card>

      {/* Flow B - AI */}
      <Card>
        <CardHeader>
          <CardTitle>Flow B: AI-Powered</CardTitle>
          <p className="text-sm text-gray-500">CREPE + madmom</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex justify-between mb-2">
              <span>Pitch Accuracy</span>
              <span className="font-bold">{result.flow_b.pitch_score}%</span>
            </div>
            <Progress value={result.flow_b.pitch_score} />
          </div>
          
          <div>
            <div className="flex justify-between mb-2">
              <span>Rhythm Accuracy</span>
              <span className="font-bold">{result.flow_b.rhythm_score}%</span>
            </div>
            <Progress value={result.flow_b.rhythm_score} />
          </div>
          
          <p className="text-xs text-gray-400">
            Processed in {result.flow_b.processing_time_ms}ms
          </p>
          
          {result.flow_b.status === "not_implemented" && (
            <p className="text-xs text-amber-600">⚠ AI flow not yet implemented</p>
          )}
        </CardContent>
      </Card>

      {/* Debug Info */}
      <Card className="md:col-span-2">
        <CardHeader>
          <CardTitle>Debug Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm font-mono space-y-1">
            <p>Offset: {result.offset_ms.toFixed(1)}ms</p>
            <p>Reference pitches: {result.debug.ref_pitches_count}</p>
            <p>Reference onsets: {result.debug.ref_onsets_count}</p>
            <p>User pitches: {result.debug.user_pitches_count}</p>
            <p>User onsets: {result.debug.user_onsets_count}</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
```

**Step 3: Update page.tsx to integrate full flow**

```tsx
"use client";

import { useState } from "react";
import { AudioUploader } from "@/components/audio-uploader";
import { Recorder } from "@/components/recorder";
import { ResultsDisplay } from "@/components/results-display";
import { analyzeAudio, AnalysisResult } from "@/lib/api";
import { Button } from "@/components/ui/button";

export default function Home() {
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [referenceUrl, setReferenceUrl] = useState<string | null>(null);
  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);
  const [offsetMs, setOffsetMs] = useState<number>(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = (file: File) => {
    setReferenceFile(file);
    setReferenceUrl(URL.createObjectURL(file));
    setResult(null);
  };

  const handleRecordingComplete = (blob: Blob, offset: number) => {
    setRecordingBlob(blob);
    setOffsetMs(offset);
  };

  const handleAnalyze = async () => {
    if (!referenceFile || !recordingBlob) return;
    
    setLoading(true);
    try {
      const analysis = await analyzeAudio(referenceFile, recordingBlob, offsetMs);
      setResult(analysis);
    } catch (err) {
      console.error("Analysis error:", err);
      alert("Analysis failed. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-8">Singing Evaluator</h1>
      
      <div className="space-y-6">
        <AudioUploader onUpload={handleUpload} />
        
        {referenceFile && (
          <p className="text-sm text-green-600">
            ✓ Reference: {referenceFile.name}
          </p>
        )}
        
        <Recorder
          referenceAudioUrl={referenceUrl}
          onRecordingComplete={handleRecordingComplete}
        />
        
        {recordingBlob && (
          <div className="flex items-center gap-4">
            <p className="text-sm text-green-600">
              ✓ Recording captured ({(recordingBlob.size / 1024).toFixed(1)} KB, offset: {offsetMs.toFixed(0)}ms)
            </p>
            <Button 
              onClick={handleAnalyze} 
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze"}
            </Button>
          </div>
        )}
        
        {result && <ResultsDisplay result={result} />}
      </div>
    </main>
  );
}
```

**Step 4: Add Progress component**

Run: `cd my-app && npx shadcn add progress`

**Step 5: Test full flow**

1. Start backend: `cd backend && uvicorn main:app --reload`
2. Start frontend: `cd my-app && npm run dev`
3. Upload reference audio
4. Record singing
5. Click Analyze
6. Verify results display

**Step 6: Commit**

```bash
git add my-app/
git commit -m "feat: integrate frontend with backend API"
```

---

## Phase 3: Visualization & Polish

### Task 10: Add pitch curve visualization

**Files:**
- Create: `my-app/components/pitch-visualizer.tsx`
- Modify: `my-app/components/results-display.tsx`

**Step 1: Write PitchVisualizer component**

```tsx
"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface PitchPoint {
  time: number;
  freq: number;
}

interface PitchVisualizerProps {
  referenceCurve: PitchPoint[];
  userCurve: PitchPoint[];
  height?: number;
}

export function PitchVisualizer({ referenceCurve, userCurve, height = 200 }: PitchVisualizerProps) {
  // Combine data for chart
  const data = referenceCurve.map((p, i) => ({
    time: p.time,
    reference: p.freq,
    user: userCurve[i]?.freq || null,
  }));

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="time" 
            label={{ value: "Time (s)", position: "insideBottom", offset: -5 }}
          />
          <YAxis 
            label={{ value: "Frequency (Hz)", angle: -90, position: "insideLeft" }}
          />
          <Tooltip />
          <Line 
            type="monotone" 
            dataKey="reference" 
            stroke="#8884d8" 
            dot={false} 
            name="Reference"
          />
          <Line 
            type="monotone" 
            dataKey="user" 
            stroke="#82ca9d" 
            dot={false}
            name="Your Singing"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

**Step 2: Update ResultsDisplay to include visualizations**

Add to ResultsDisplay component inside each Flow card.

**Step 3: Commit**

```bash
git add my-app/
git commit -m "feat: add pitch curve visualization"
```

---

## Summary

After completing all tasks:

1. ✅ Backend: FastAPI with Flow A (librosa) and Flow B (CREPE + madmom)
2. ✅ Frontend: Next.js with upload, recording, and results display
3. ✅ Scoring: Pitch in cents, rhythm with tolerance matching
4. ✅ Synchronization: Offset tracking between playback and recording
5. ✅ Visualization: Pitch curves and debug info

**To run:**
```bash
# Terminal 1
cd backend && uvicorn main:app --reload

# Terminal 2
cd my-app && npm run dev
```

Open http://localhost:3000
