"use client";

import { useState, useCallback, useRef, useEffect } from "react";

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
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

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
        const blob = new Blob(chunksRef.current, { type: mediaRecorderRef.current?.mimeType || "audio/webm" });
        const offsetMs = (recordingStartTimeRef.current - playbackStartTimeRef.current) * 1000;
        onComplete({ blob, offsetMs });
        stream.getTracks().forEach(track => track.stop());
        if (intervalRef.current) clearInterval(intervalRef.current);
      };
      
      // Mark when user actually starts recording
      recordingStartTimeRef.current = audioContextRef.current.currentTime;
      
      mediaRecorderRef.current.start(100); // Collect every 100ms
      setIsRecording(true);
      
      // Update recording time
      intervalRef.current = setInterval(() => {
        if (audioContextRef.current) {
          const elapsed = audioContextRef.current.currentTime - recordingStartTimeRef.current;
          setRecordingTime(elapsed);
        }
      }, 100);
      
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone. Please check permissions.");
    }
  }, []);
  
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setRecordingTime(0);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (mediaRecorderRef.current?.state !== "inactive") {
        mediaRecorderRef.current?.stop();
      }
      if (audioContextRef.current?.state !== "closed") {
        audioContextRef.current?.close();
      }
    };
  }, []);

  return {
    isRecording,
    recordingTime,
    startRecording,
    stopRecording,
  };
}
