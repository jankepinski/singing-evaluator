"use client";

import { useState, useEffect } from "react";
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

  // Cleanup object URL
  useEffect(() => {
    return () => {
      if (referenceUrl) URL.revokeObjectURL(referenceUrl);
    };
  }, [referenceUrl]);

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
