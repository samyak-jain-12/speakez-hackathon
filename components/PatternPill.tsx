'use client'

import { useState } from 'react'
import { DetectedPattern } from '@/lib/types'
import { ChevronDown, ChevronUp, Info } from 'lucide-react'

interface PatternPillProps {
  pattern: DetectedPattern
}

export default function PatternPill({ pattern }: PatternPillProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const colorClasses = {
    green: 'bg-green-50 border-green-200 text-green-800',
    yellow: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    blue: 'bg-blue-50 border-blue-200 text-blue-800',
  }

  const severityColors = {
    low: 'bg-green-100',
    medium: 'bg-yellow-100',
    high: 'bg-blue-100',
  }

  return (
    <div className={`rounded-lg border-2 p-4 transition-all ${colorClasses[pattern.color]}`}>
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${severityColors[pattern.severity]}`} />
          <div>
            <p className="font-semibold text-gray-900">{pattern.name}</p>
            <p className="text-sm text-gray-600">
              Observed {pattern.occurrences} time{pattern.occurrences !== 1 ? 's' : ''}
            </p>
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-500" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-500" />
        )}
      </div>

      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-gray-200 space-y-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Info className="w-4 h-4 text-gray-500" />
              <p className="text-sm font-medium text-gray-700">When it occurs:</p>
            </div>
            <p className="text-sm text-gray-600 ml-6">{pattern.whenItOccurs}</p>
          </div>
          
          <div>
            <p className="text-sm font-medium text-gray-700 mb-1">Why it happens:</p>
            <p className="text-sm text-gray-600">{pattern.whyItHappens}</p>
          </div>
          
          <div className="bg-white/60 p-3 rounded-lg">
            <p className="text-sm font-medium text-gray-700 mb-1">Suggested next practice:</p>
            <p className="text-sm text-gray-600">{pattern.suggestedPractice}</p>
          </div>
        </div>
      )}
    </div>
  )
}
