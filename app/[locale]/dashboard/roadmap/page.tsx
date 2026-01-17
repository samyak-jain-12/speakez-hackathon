'use client'

import { useState, useEffect } from 'react'
import { useStore } from '@/lib/store'
import { OrchestratorAgent } from '@/lib/agents/orchestrator'
import { Map, Play, Pause, Volume2, BookOpen, ExternalLink } from 'lucide-react'

export default function RoadmapPage() {
  const { practiceHistory, user } = useStore()
  const [roadmap, setRoadmap] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [synth, setSynth] = useState<SpeechSynthesis | null>(null)

  useEffect(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      setSynth(window.speechSynthesis)
    }
  }, [])

  const generateRoadmap = async () => {
    if (practiceHistory.length === 0) {
      alert('Please complete at least one practice session to generate a roadmap.')
      return
    }

    setLoading(true)
    const lastSession = practiceHistory[practiceHistory.length - 1]

    try {
      // Try API first, fallback to local generation
      const response = await fetch('/api/roadmap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionData: lastSession,
          userProfile: user,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setRoadmap(data.roadmap)
      } else {
        // Fallback to local generation
        const orchestrator = new OrchestratorAgent()
        const generated = await orchestrator.generateRoadmap(lastSession, user)
        setRoadmap(generated)
      }
    } catch (error) {
      console.error('Error generating roadmap:', error)
      // Fallback to local generation
      try {
        const orchestrator = new OrchestratorAgent()
        const generated = await orchestrator.generateRoadmap(lastSession, user)
        setRoadmap(generated)
      } catch (fallbackError) {
        alert('Error generating roadmap. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  const playRoadmap = () => {
    if (!synth || !roadmap) return

    if (isPlaying) {
      synth.cancel()
      setIsPlaying(false)
    } else {
      const utterance = new SpeechSynthesisUtterance(roadmap)
      utterance.rate = 0.9
      utterance.pitch = 1
      utterance.volume = 1
      utterance.onend = () => setIsPlaying(false)
      synth.speak(utterance)
      setIsPlaying(true)
    }
  }

  useEffect(() => {
    // Auto-generate roadmap if there's practice history
    if (practiceHistory.length > 0 && !roadmap) {
      generateRoadmap()
    }
  }, [practiceHistory])

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2 text-gray-900 flex items-center gap-3">
            <Map className="w-8 h-8 text-primary-600" />
            Improvement Roadmap
          </h1>
          <p className="text-gray-600">
            Your personalized, step-by-step practice plan based on your sessions
          </p>
        </div>
        {roadmap && synth && (
          <button
            onClick={playRoadmap}
            className="flex items-center gap-2 bg-primary-600 text-white px-4 py-2 rounded-lg hover:bg-primary-700 transition-colors"
          >
            {isPlaying ? (
              <>
                <Pause className="w-5 h-5" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Play Audio
              </>
            )}
          </button>
        )}
      </div>

      {loading ? (
        <div className="bg-white p-12 rounded-lg shadow-sm border border-gray-200 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Generating your personalized roadmap...</p>
        </div>
      ) : roadmap ? (
        <div className="bg-white p-8 rounded-lg shadow-sm border border-gray-200">
          <div className="prose max-w-none">
            {roadmap.split('\n').map((line, idx) => {
              if (line.startsWith('# ')) {
                return (
                  <h2 key={idx} className="text-2xl font-bold mt-6 mb-4 text-gray-900">
                    {line.substring(2)}
                  </h2>
                )
              } else if (line.startsWith('## ')) {
                return (
                  <h3 key={idx} className="text-xl font-semibold mt-4 mb-3 text-gray-800">
                    {line.substring(3)}
                  </h3>
                )
              } else if (line.startsWith('- ')) {
                return (
                  <li key={idx} className="ml-6 mb-2 text-gray-700">
                    {line.substring(2)}
                  </li>
                )
              } else if (line.trim() === '') {
                return <br key={idx} />
              } else {
                return (
                  <p key={idx} className="mb-3 text-gray-700">
                    {line}
                  </p>
                )
              }
            })}
          </div>
        </div>
      ) : (
        <div className="bg-white p-12 rounded-lg shadow-sm border border-gray-200 text-center">
          <Map className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2 text-gray-900">No Roadmap Yet</h3>
          <p className="text-gray-600 mb-6">
            Complete a practice session to generate your personalized improvement roadmap.
          </p>
          <button
            onClick={generateRoadmap}
            className="bg-primary-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors"
          >
            Generate Roadmap
          </button>
        </div>
      )}

      <div className="mt-8 bg-green-50 border border-green-200 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-2 text-green-900">About Your Roadmap</h3>
        <p className="text-green-800 text-sm">
          This roadmap is generated specifically for you based on your practice sessions. It adapts to your
          personal baseline and provides supportive, actionable steps for improvement. Remember, there's no
          universal standard—your journey is unique.
        </p>
      </div>

      <div className="mt-8 bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-xl font-semibold mb-4 text-gray-900 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-primary-600" />
          Recommended Learning Resources
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <a href="#" className="block p-4 border border-gray-200 rounded-lg hover:border-primary-500 hover:shadow-md transition-all">
            <h4 className="font-semibold text-gray-900 mb-1 flex items-center justify-between">
              Mastering Public Speaking
              <ExternalLink className="w-4 h-4 text-gray-400" />
            </h4>
            <p className="text-sm text-gray-600">Comprehensive guide to overcoming stage fright and delivering impact.</p>
          </a>
          <a href="#" className="block p-4 border border-gray-200 rounded-lg hover:border-primary-500 hover:shadow-md transition-all">
            <h4 className="font-semibold text-gray-900 mb-1 flex items-center justify-between">
              Effective Storytelling
              <ExternalLink className="w-4 h-4 text-gray-400" />
            </h4>
            <p className="text-sm text-gray-600">Learn how to weave compelling narratives into your presentations.</p>
          </a>
          <a href="#" className="block p-4 border border-gray-200 rounded-lg hover:border-primary-500 hover:shadow-md transition-all">
            <h4 className="font-semibold text-gray-900 mb-1 flex items-center justify-between">
              Voice Modulation Techniques
              <ExternalLink className="w-4 h-4 text-gray-400" />
            </h4>
            <p className="text-sm text-gray-600">Exercises to improve your pitch, tone, and volume control.</p>
          </a>
        </div>
      </div>
    </div>
  )
}
