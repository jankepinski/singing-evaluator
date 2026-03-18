"use client";

import { AudioUploader } from "@/components/audio-uploader";

export default function Home() {
  return (
    <main className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-8">Singing Evaluator</h1>
      <AudioUploader onUpload={(file) => console.log("Uploaded:", file.name)} />
    </main>
  );
}
