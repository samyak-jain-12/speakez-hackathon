'use client'

import { ProgressMetric } from '@/lib/types'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface ProgressBarProps {
  metric: ProgressMetric
}

export default function ProgressBar({ metric }: ProgressBarProps) {
  const percentage = (metric.current / metric.target) * 100
  const trendIcon = {
    up: <TrendingUp className="w-4 h-4 text-green-600" />,
    down: <TrendingDown className="w-4 h-4 text-blue-600" />,
    stable: <Minus className="w-4 h-4 text-gray-500" />,
  }

  return (
    <div className="bg-white p-4 rounded-lg border border-gray-200">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <p className="font-medium text-gray-900">{metric.name}</p>
          {trendIcon[metric.trend]}
        </div>
        <p className="text-sm text-gray-600">
          {Math.round(percentage)}% of target
        </p>
      </div>
      
      <div className="w-full bg-gray-100 rounded-full h-3 mb-2">
        <div
          className="bg-primary-500 h-3 rounded-full transition-all duration-300"
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>Observed across last {metric.sessions.length} sessions</span>
        <span className="cursor-help" title={metric.tooltip}>
          ℹ️
        </span>
      </div>
      
      {/* Mini trend graph */}
      <div className="mt-3 flex items-end gap-1 h-12">
        {metric.sessions.map((value, idx) => (
          <div
            key={idx}
            className="flex-1 bg-primary-100 rounded-t-sm transition-all hover:bg-primary-200"
            style={{ height: `${(value / Math.max(...metric.sessions, 1)) * 100}%` }}
            title={`Session ${idx + 1}: ${value}`}
          />
        ))}
      </div>
    </div>
  )
}
