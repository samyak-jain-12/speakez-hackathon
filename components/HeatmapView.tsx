'use client'

import { HeatmapData } from '@/lib/types'
import { useState } from 'react'

interface HeatmapViewProps {
  data: HeatmapData[]
  duration: number // in seconds
}

export default function HeatmapView({ data, duration }: HeatmapViewProps) {
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null)

  const getIntensity = (value: number, max: number) => {
    const percentage = (value / max) * 100
    if (percentage > 80) return 'bg-red-400'
    if (percentage > 60) return 'bg-orange-400'
    if (percentage > 40) return 'bg-yellow-400'
    if (percentage > 20) return 'bg-blue-400'
    return 'bg-green-400'
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${String(secs).padStart(2, '0')}`
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h3 className="text-lg font-semibold mb-4 text-gray-900">Practice Timeline Heatmap</h3>
      <p className="text-sm text-gray-600 mb-4">
        Visual representation of stress, pace, repetition, and volume throughout your session
      </p>
      
      <div className="space-y-4">
        {/* Stress */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Stress Patterns</span>
            <span className="text-xs text-gray-500">Hover to see details</span>
          </div>
          <div className="flex gap-1 h-8">
            {data.map((point, idx) => (
              <div
                key={idx}
                className={`flex-1 rounded-sm transition-all ${getIntensity(point.stress, 100)} ${
                  hoveredPoint === idx ? 'ring-2 ring-primary-500 scale-110' : ''
                }`}
                onMouseEnter={() => setHoveredPoint(idx)}
                onMouseLeave={() => setHoveredPoint(null)}
                title={`${formatTime(point.timestamp)}s - Stress: ${Math.round(point.stress)}%`}
              />
            ))}
          </div>
        </div>

        {/* Pace */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Pace Shifts</span>
          </div>
          <div className="flex gap-1 h-8">
            {data.map((point, idx) => (
              <div
                key={idx}
                className={`flex-1 rounded-sm transition-all ${getIntensity(point.pace, 200)} ${
                  hoveredPoint === idx ? 'ring-2 ring-primary-500 scale-110' : ''
                }`}
                onMouseEnter={() => setHoveredPoint(idx)}
                onMouseLeave={() => setHoveredPoint(null)}
                title={`${formatTime(point.timestamp)}s - Pace: ${Math.round(point.pace)} WPM`}
              />
            ))}
          </div>
        </div>

        {/* Repetition */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Repetition Points</span>
          </div>
          <div className="flex gap-1 h-8">
            {data.map((point, idx) => (
              <div
                key={idx}
                className={`flex-1 rounded-sm transition-all ${getIntensity(point.repetition, 50)} ${
                  hoveredPoint === idx ? 'ring-2 ring-primary-500 scale-110' : ''
                }`}
                onMouseEnter={() => setHoveredPoint(idx)}
                onMouseLeave={() => setHoveredPoint(null)}
                title={`${formatTime(point.timestamp)}s - Repetition: ${Math.round(point.repetition)}%`}
              />
            ))}
          </div>
        </div>

        {/* Volume */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Volume Consistency</span>
          </div>
          <div className="flex gap-1 h-8">
            {data.map((point, idx) => (
              <div
                key={idx}
                className={`flex-1 rounded-sm transition-all ${getIntensity(point.volume, 100)} ${
                  hoveredPoint === idx ? 'ring-2 ring-primary-500 scale-110' : ''
                }`}
                onMouseEnter={() => setHoveredPoint(idx)}
                onMouseLeave={() => setHoveredPoint(null)}
                title={`${formatTime(point.timestamp)}s - Volume: ${Math.round(point.volume)}%`}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="flex items-center gap-4 text-xs text-gray-600">
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 bg-green-400 rounded"></div>
            <span>Low</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 bg-yellow-400 rounded"></div>
            <span>Medium</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 bg-orange-400 rounded"></div>
            <span>High</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 bg-red-400 rounded"></div>
            <span>Very High</span>
          </div>
        </div>
      </div>
    </div>
  )
}
