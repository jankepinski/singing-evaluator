"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface PitchPoint {
  time: number;
  freq: number;
}

interface PitchVisualizerProps {
  referenceCurve: PitchPoint[];
  userCurve: PitchPoint[];
  title?: string;
  height?: number;
}

export function PitchVisualizer({ 
  referenceCurve, 
  userCurve, 
  title = "Pitch Curve",
  height = 250 
}: PitchVisualizerProps) {
  // Merge data for chart - align by time
  const maxPoints = Math.max(referenceCurve.length, userCurve.length);
  const data = [];
  
  for (let i = 0; i < maxPoints; i++) {
    const point: { time: number; reference?: number; user?: number } = { time: 0 };
    
    if (referenceCurve[i]) {
      point.time = referenceCurve[i].time;
      point.reference = referenceCurve[i].freq;
    }
    
    if (userCurve[i]) {
      point.time = userCurve[i].time;
      point.user = userCurve[i].freq;
    }
    
    if (point.reference !== undefined || point.user !== undefined) {
      data.push(point);
    }
  }

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">{title}</h4>
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 5, right: 20, bottom: 25, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis 
              dataKey="time" 
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(value) => `${value.toFixed(1)}s`}
              label={{ value: "Time (s)", position: "insideBottom", offset: -15 }}
            />
            <YAxis 
              label={{ value: "Frequency (Hz)", angle: -90, position: "insideLeft" }}
              domain={["auto", "auto"]}
            />
            <Tooltip
              formatter={(value) => [`${typeof value === 'number' ? value.toFixed(1) : value} Hz`, ""]}
              labelFormatter={(label) => `Time: ${typeof label === 'number' ? label.toFixed(2) : label}s`}
            />
            <Legend verticalAlign="top" height={36} />
            <Line 
              type="monotone" 
              dataKey="reference" 
              stroke="#8884d8" 
              strokeWidth={2}
              dot={false} 
              name="Reference"
              connectNulls={false}
            />
            <Line 
              type="monotone" 
              dataKey="user" 
              stroke="#82ca9d" 
              strokeWidth={2}
              dot={false}
              name="Your Singing"
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
