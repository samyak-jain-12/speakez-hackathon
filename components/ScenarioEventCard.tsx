'use client'

import { AlertCircle, Clock, Users, MessageSquare } from 'lucide-react'

interface ScenarioEvent {
  id: string
  type: 'interruption' | 'time-pressure' | 'role-prompt' | 'question'
  message: string
  expectedResponse?: string
  timestamp: number
}

interface ScenarioEventCardProps {
  event: ScenarioEvent
  onDismiss?: () => void
}

export default function ScenarioEventCard({ event, onDismiss }: ScenarioEventCardProps) {
  const getIcon = () => {
    switch (event.type) {
      case 'interruption':
        return <AlertCircle className="w-5 h-5 text-yellow-600" />
      case 'time-pressure':
        return <Clock className="w-5 h-5 text-orange-600" />
      case 'role-prompt':
        return <Users className="w-5 h-5 text-blue-600" />
      case 'question':
        return <MessageSquare className="w-5 h-5 text-purple-600" />
      default:
        return <AlertCircle className="w-5 h-5 text-gray-600" />
    }
  }

  const getColorClasses = () => {
    switch (event.type) {
      case 'interruption':
        return 'bg-yellow-50 border-yellow-200'
      case 'time-pressure':
        return 'bg-orange-50 border-orange-200'
      case 'role-prompt':
        return 'bg-blue-50 border-blue-200'
      case 'question':
        return 'bg-purple-50 border-purple-200'
      default:
        return 'bg-gray-50 border-gray-200'
    }
  }

  return (
    <div className={`${getColorClasses()} border-2 rounded-lg p-4 animate-slide-in`}>
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 mt-0.5">
          {getIcon()}
        </div>
        <div className="flex-1">
          <p className="font-medium text-gray-900 mb-1">{event.message}</p>
          {event.expectedResponse && (
            <div className="mt-2 pt-2 border-t border-gray-200">
              <p className="text-xs text-gray-600 mb-1">Expected response type:</p>
              <p className="text-sm text-gray-700">{event.expectedResponse}</p>
            </div>
          )}
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="text-gray-400 hover:text-gray-600"
          >
            ×
          </button>
        )}
      </div>
    </div>
  )
}
