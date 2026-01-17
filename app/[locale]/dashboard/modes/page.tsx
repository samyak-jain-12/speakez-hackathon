'use client'

import { useState } from 'react'
import { Target, TrendingUp, Clock, Volume2 } from 'lucide-react'

const modes = [
  {
    id: 'pitch',
    name: 'Pitch Mode',
    description: 'Optimized for startup pitches and presentations',
    idealPace: '140-160 WPM',
    characteristics: [
      'Higher energy and enthusiasm',
      'More gesture tolerance',
      'Clear value proposition delivery',
      'Engaging storytelling',
    ],
  },
  {
    id: 'interview',
    name: 'Interview Mode',
    description: 'Professional communication for job interviews',
    idealPace: '120-150 WPM',
    characteristics: [
      'Professional tone',
      'Structured responses',
      'Moderate gesture usage',
      'Confident but not overbearing',
    ],
  },
  {
    id: 'professional',
    name: 'Professional / Office Mode',
    description: 'Business meetings and workplace communication',
    idealPace: '130-160 WPM',
    characteristics: [
      'Calm and measured delivery',
      'Minimal gestures',
      'Clear and concise',
      'Respectful tone',
    ],
  },
  {
    id: 'debate',
    name: 'Debate Mode',
    description: 'Assertive communication for debates and discussions',
    idealPace: '150-180 WPM',
    characteristics: [
      'Confident and assertive',
      'Strong gestures',
      'Quick thinking',
      'Clear argumentation',
    ],
  },
  {
    id: 'conversation',
    name: 'Daily Conversation Mode',
    description: 'Natural, everyday speaking',
    idealPace: '120-180 WPM',
    characteristics: [
      'Natural and relaxed',
      'Flexible pace',
      'Casual tone',
      'Comfortable flow',
    ],
  },
]

export default function ModesPage() {
  const [selectedMode, setSelectedMode] = useState<string | null>(null)

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-4 text-gray-900">Purpose-Based Speech Modes</h1>
      <p className="text-gray-600 mb-8">
        Each mode adapts feedback and expectations to match your speaking context. Select a mode when starting a practice session.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {modes.map((mode) => (
          <div
            key={mode.id}
            className={`bg-white p-6 rounded-lg shadow-sm border-2 transition-all cursor-pointer ${
              selectedMode === mode.id
                ? 'border-primary-500 shadow-md'
                : 'border-gray-200 hover:border-gray-300'
            }`}
            onClick={() => setSelectedMode(selectedMode === mode.id ? null : mode.id)}
          >
            <div className="flex items-center gap-3 mb-3">
              <Target className="w-6 h-6 text-primary-600" />
              <h3 className="text-xl font-semibold text-gray-900">{mode.name}</h3>
            </div>
            
            <p className="text-gray-600 mb-4 text-sm">{mode.description}</p>

            {selectedMode === mode.id && (
              <div className="mt-4 pt-4 border-t border-gray-200 space-y-3">
                <div className="flex items-center gap-2 text-sm">
                  <Clock className="w-4 h-4 text-gray-500" />
                  <span className="text-gray-700">
                    <strong>Ideal Pace:</strong> {mode.idealPace}
                  </span>
                </div>
                
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Key Characteristics:</p>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                    {mode.characteristics.map((char, idx) => (
                      <li key={idx}>{char}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-8 bg-blue-50 border border-blue-200 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-2 text-blue-900 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          How Modes Work
        </h3>
        <p className="text-blue-800 text-sm">
          When you select a mode, SPEAKEZ adjusts its analysis to match that context. For example, 
          Pitch Mode expects higher energy and faster pace, while Interview Mode focuses on professional 
          tone and structured delivery. The system adapts to help you practice effectively for each scenario.
        </p>
      </div>
    </div>
  )
}
