"use client";

import { useRecorder } from "@/hooks/use-recorder";
import { Button } from "@/components/ui/button";
import { useRef } from "react";

interface RecorderProps {
  referenceAudioUrl: string | null;
  onRecordingComplete: (blob: Blob, offsetMs: number) => void;
}

export function Recorder({ referenceAudioUrl, onRecordingComplete }: RecorderProps) {
  const { isRecording, recordingTime, startRecording, stopRecording } = useRecorder();
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handleStart = async () => {
    if (!referenceAudioUrl) {
      alert("Please upload reference audio first");
      return;
    }

    // Create and preload audio
    const audio = new Audio(referenceAudioUrl);
    audio.preload = "auto";
    audioRef.current = audio;

    // Wait for audio to be ready before starting recording
    await new Promise<void>((resolve, reject) => {
      audio.oncanplaythrough = () => resolve();
      audio.onerror = (e) => reject(e);
      audio.load();
    });

    // Start playback
    await audio.play();

    // Start recording
    await startRecording(({ blob, offsetMs }) => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      onRecordingComplete(blob, offsetMs);
    });
  };

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
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
