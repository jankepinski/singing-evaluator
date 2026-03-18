"use client";

import { AnalysisResult } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { PitchVisualizer } from "./pitch-visualizer";

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

          <PitchVisualizer
            referenceCurve={result.flow_a.pitch_curve}
            userCurve={result.flow_a.pitch_curve}
            title="Pitch Comparison (Flow A)"
          />

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

          <PitchVisualizer
            referenceCurve={result.flow_b.pitch_curve}
            userCurve={result.flow_b.pitch_curve}
            title="Pitch Comparison (Flow B)"
          />

          <p className="text-xs text-gray-400">
            Processed in {result.flow_b.processing_time_ms}ms
          </p>
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
