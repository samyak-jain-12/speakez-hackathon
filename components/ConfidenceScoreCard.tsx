'use client'

import { ConfidenceData } from '@/lib/types'
import { TrendingUp, Info } from 'lucide-react'

interface ConfidenceScoreCardProps {
  data: ConfidenceData
}

export default function ConfidenceScoreCard({ data }: ConfidenceScoreCardProps) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 60) return 'text-blue-600'
    return 'text-yellow-600'
  }

  const getScoreBg = (score: number) => {
    if (score >= 80) return 'bg-green-50 border-green-200'
    if (score >= 60) return 'bg-blue-50 border-blue-200'
    return 'bg-yellow-50 border-yellow-200'
  }

  return (
    <div className={`${getScoreBg(data.score)} border-2 rounded-lg p-6`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Confidence Score</h3>
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-gray-500" />
          <span className="text-sm text-gray-600">Trending up</span>
        </div>
      </div>

      <div className="text-center mb-6">
        <div className={`text-6xl font-bold ${getScoreColor(data.score)} mb-2`}>
          {data.score}
        </div>
        <p className="text-gray-600">out of 100</p>
      </div>

      <div className="space-y-3 mb-4">
        <div>
          <p className="text-sm font-medium text-gray-700 mb-2">What influenced this score:</p>
          <div className="space-y-2">
            {data.factors.map((factor, idx) => {
              // Handle impact as 0-1 or 0-100
              const impactVal = factor.impact > 1 ? factor.impact : factor.impact * 100

              return (
                <div key={idx} className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">{factor.name}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary-500 h-2 rounded-full"
                        style={{ width: `${impactVal}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-500 w-8 text-right">
                      {Math.round(impactVal)}%
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div className="bg-white/60 p-3 rounded-lg">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-gray-500 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-gray-700 mb-1">Dominant focus area:</p>
            <p className="text-sm text-gray-600">{data.dominantWeakness}</p>
          </div>
        </div>
      </div>

      {/* Trend graph */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-600 mb-2">Confidence trend over sessions:</p>
        <div className="flex items-end gap-1 h-16">
          {data.trend.map((value, idx) => (
            <div
              key={idx}
              className="flex-1 bg-primary-200 rounded-t-sm hover:bg-primary-300 transition-colors"
              style={{ height: `${(value / 100) * 100}%` }}
              title={`Session ${idx + 1}: ${value}`}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
