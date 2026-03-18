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
