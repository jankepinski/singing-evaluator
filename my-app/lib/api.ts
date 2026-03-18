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
