'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useStore } from '@/lib/store'
import { 
  Briefcase, 
  Users, 
  Phone, 
  Utensils, 
  Lightbulb,
  ArrowRight 
} from 'lucide-react'

interface ScenarioEvent {
  id: string
  type: 'interruption' | 'time-pressure' | 'role-prompt' | 'question'
  message: string
  expectedResponse?: string
  timestamp: number
}

const scenarios = [
  {
    id: 'startup-pitch',
    title: 'Startup Pitch',
    icon: Lightbulb,
    description: 'Practice pitching your idea to investors',
    prompt: 'Introduce your startup idea in 2 minutes. Explain the problem, your solution, and why it matters.',
    mode: 'pitch',
  },
  {
    id: 'job-interview',
    title: 'Job Interview',
    icon: Briefcase,
    description: 'Prepare for job interviews',
    prompt: 'Tell me about yourself. Walk through your background, experience, and why you\'re interested in this role.',
    mode: 'interview',
  },
  {
    id: 'team-meeting',
    title: 'Team Meeting',
    icon: Users,
    description: 'Practice presenting in team meetings',
    prompt: 'Present a project update to your team. Share progress, challenges, and next steps.',
    mode: 'professional',
  },
  {
    id: 'client-call',
    title: 'Client Call',
    icon: Phone,
    description: 'Practice client communication',
    prompt: 'Explain a complex concept to a client in simple terms. Make sure they understand the value proposition.',
    mode: 'professional',
  },
  {
    id: 'ordering-food',
    title: 'Ordering Food',
    icon: Utensils,
    description: 'Practice everyday conversations',
    prompt: 'Order food at a restaurant. Include dietary preferences and ask questions about the menu.',
    mode: 'conversation',
  },
]

export default function ScenariosPage() {
  const router = useRouter()
  const { startSession } = useStore()
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null)

  const handleStartScenario = (scenario: typeof scenarios[0]) => {
    startSession(scenario.mode, scenario.title)
    // Pass scenario context for live events
    router.push(`/dashboard/practice?scenario=${scenario.id}&prompt=${encodeURIComponent(scenario.prompt)}&mode=${scenario.mode}`)
  }

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-4 text-gray-900">Scenario Simulator</h1>
      <p className="text-gray-600 mb-8">
        Practice real-world speaking scenarios. Select a scenario to begin practicing with context-specific guidance.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {scenarios.map((scenario) => {
          const Icon = scenario.icon
          return (
            <div
              key={scenario.id}
              className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 bg-primary-100 rounded-lg">
                  <Icon className="w-6 h-6 text-primary-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900">{scenario.title}</h3>
              </div>
              
              <p className="text-gray-600 mb-4 text-sm">{scenario.description}</p>
              
              <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-700 font-medium mb-1">Practice Prompt:</p>
                <p className="text-sm text-gray-600">{scenario.prompt}</p>
              </div>

              <button
                onClick={() => handleStartScenario(scenario)}
                className="w-full bg-primary-600 text-white py-2 rounded-lg font-semibold hover:bg-primary-700 flex items-center justify-center gap-2 transition-colors"
              >
                Start Scenario
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
