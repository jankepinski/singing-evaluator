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
