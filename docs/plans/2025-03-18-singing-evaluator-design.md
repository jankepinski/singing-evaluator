# Singing Evaluator - Design Document

## Overview

Aplikacja webowa do porównania dwóch podejść (deterministycznego vs AI) w ocenie czystości śpiewu - pitch accuracy i rhythm accuracy.

## Architecture

```
Next.js (Frontend) <──HTTP──> FastAPI (Backend)
    │                           │
    │ Upload + Record           │ Flow A: librosa.yin + onset_detect
    │                           │ Flow B: CREPE + madmom
    │                           │ Scoring: cents (pitch), tolerance matching (rhythm)
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 15, React, Tailwind, Recharts |
| Backend | FastAPI, Python 3.11 |
| Audio Analysis | librosa, CREPE, madmom |
| Pitch Scoring | Cents (1200 * log2(freq_ratio)) |
| Rhythm Scoring | Onset matching with 75ms tolerance |

## User Flow

1. Upload reference audio (WAV/MP3/FLAC)
2. Record singing while playback plays
3. View side-by-side comparison:
   - Flow A (deterministic): librosa.yin + onset_detect
   - Flow B (AI): CREPE + madmom
4. Scores + visualizations + debug data

## API

```
POST /api/analyze
Content-Type: multipart/form-data

Fields:
- reference_audio: File
- user_audio: File
- offset_ms: float  # recording_start - playback_start

Response:
{
  offset_ms: float,
  flow_a: {
    pitch_score: float,      # 0-100
    rhythm_score: float,     # 0-100
    pitch_curve: [{time, freq_cents}],
    onsets: [time]
  },
  flow_b: {
    pitch_score: float,
    rhythm_score: float,
    pitch_curve: [{time, freq_cents}],
    beats: [time]
  },
  debug: {
    processing_time_ms: {flow_a, flow_b},
    raw_detections: {...}
  }
}
```

## Scoring Algorithms

### Pitch Score (Cents)
```python
error_cents = 1200 * abs(log2(pitch_user / pitch_ref))
score = max(0, 100 - (mean_error_cents / 0.5))
# 50 cents (half semitone) = 0 points
```

### Rhythm Score (Tolerance Matching)
```python
TOLERANCE_MS = 75
matched = sum(1 for ref in ref_onsets
              if any(abs(ref - user) < 0.075 for user in user_onsets))
score = (matched / len(ref_onsets)) * 100
```

## UI Components

1. **AudioUploader** - Drag & drop reference file
2. **Recorder** - Record with playback, track offset
3. **WaveformViewer** - Recharts visualization
4. **ComparisonResults** - Side-by-side A/B cards
5. **DebugPanel** - Raw data tables

## Synchronization

Critical for accurate comparison: user hears playback, reacts, starts singing.

```javascript
const playbackStart = audioContext.currentTime;
// ...user clicks record...
const recordingStart = audioContext.currentTime;
const offset_ms = (recordingStart - playbackStart) * 1000;
```

Backend uses offset to align user audio with reference.

## Dependencies

```txt
# Backend
fastapi==0.115.0
uvicorn==0.32.0
python-multipart==0.0.17
librosa==0.10.2
madmom==0.17.1
crepe==0.0.15
numpy==1.26.4
soundfile==0.12.1
```
