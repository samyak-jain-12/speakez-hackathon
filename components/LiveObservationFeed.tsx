'use client'

import { LiveObservation } from '@/lib/types'
import { Clock, Lightbulb } from 'lucide-react'

interface LiveObservationFeedProps {
  observations: LiveObservation[]
}

export default function LiveObservationFeed({ observations }: LiveObservationFeedProps) {
  const getTypeColor = (type: LiveObservation['type']) => {
    switch (type) {
      case 'pace':
        return 'bg-blue-50 border-blue-200 text-blue-800'
      case 'stress':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800'
      case 'repetition':
        return 'bg-purple-50 border-purple-200 text-purple-800'
      case 'volume':
        return 'bg-green-50 border-green-200 text-green-800'
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800'
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 h-full overflow-y-auto">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 flex items-center gap-2">
        <Lightbulb className="w-5 h-5 text-primary-500" />
        What We are Noticing
      </h3>
      
      <div className="space-y-3">
        {observations.length === 0 ? (
          <p className="text-sm text-gray-500 text-center py-8">
            Start practicing to see live observations
          </p>
        ) : (
          observations.map((obs) => (
            <div
              key={obs.id}
              className={`p-3 rounded-lg border ${getTypeColor(obs.type)} transition-all hover:shadow-sm`}
            >
              <div className="flex items-start gap-2">
                <Clock className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-medium">{obs.message}</p>
                  {obs.suggestion && (
                    <p className="text-xs mt-1 opacity-80">
                      💡 {obs.suggestion}
                    </p>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
