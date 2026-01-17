'use client'

import { LiveSignal } from '@/lib/types'
import { Info } from 'lucide-react'
import { useState } from 'react'

interface LiveSignalBarProps {
  signals: LiveSignal[]
}

export default function LiveSignalBar({ signals }: LiveSignalBarProps) {
  const [hoveredSignal, setHoveredSignal] = useState<string | null>(null)

  return (
    <div className="w-full p-4">
      <div className="flex items-center gap-4 overflow-x-auto">
        {signals.map((signal) => {
          const percentage = (signal.value / signal.max) * 100
          const statusColors = {
            normal: 'bg-green-500',
            elevated: 'bg-yellow-500',
            high: 'bg-blue-500',
          }

          return (
            <div
              key={signal.name}
              className="flex-shrink-0 min-w-[120px]"
              onMouseEnter={() => setHoveredSignal(signal.name)}
              onMouseLeave={() => setHoveredSignal(null)}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-gray-600">{signal.name}</span>
                <span className="text-xs text-gray-500">{Math.round(signal.value)}</span>
              </div>
              <div className="w-full bg-gray-100 rounded-full h-2 relative">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${statusColors[signal.status]}`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
              {hoveredSignal === signal.name && (
                <div className="absolute mt-2 p-2 bg-gray-900 text-white text-xs rounded-lg z-50 shadow-lg">
                  <div className="flex items-center gap-1 mb-1">
                    <Info className="w-3 h-3" />
                    <span className="font-medium">{signal.name}</span>
                  </div>
                  <p>{signal.tooltip}</p>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
