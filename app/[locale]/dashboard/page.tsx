'use client'

import { useEffect, useState } from 'react'
import { useStore } from '@/lib/store'
import Link from 'next/link'
import { TrendingUp, Target, Calendar, Activity, Clock, Sparkles } from 'lucide-react'
import InteractiveCard from '@/components/InteractiveCard'
import PatternPill from '@/components/PatternPill'
import ProgressBar from '@/components/ProgressBar'
import {
  generateMockPatterns,
  generateMockProgressMetrics,
  generateMockConfidenceData,
  generateMockHeatmapData,
} from '@/lib/mockData'
import { DetectedPattern, ProgressMetric, ConfidenceData } from '@/lib/types'
import HeatmapView from '@/components/HeatmapView'

export default function DashboardPage() {
  const { user, practiceHistory } = useStore()
  const [streak, setStreak] = useState(0)
  const [patterns, setPatterns] = useState<DetectedPattern[]>([])
  const [progressMetrics, setProgressMetrics] = useState<ProgressMetric[]>([])
  const [confidenceData, setConfidenceData] = useState<ConfidenceData | null>(null)
  const [selectedCard, setSelectedCard] = useState<string | null>(null)

  useEffect(() => {
    // Calculate practice streak
    if (practiceHistory.length > 0) {
      const sorted = [...practiceHistory].sort((a, b) =>
        new Date(b.startTime).getTime() - new Date(a.startTime).getTime()
      )
      let currentStreak = 0
      const today = new Date()
      today.setHours(0, 0, 0, 0)

      for (const session of sorted) {
        const sessionDate = new Date(session.startTime)
        sessionDate.setHours(0, 0, 0, 0)
        const daysDiff = Math.floor((today.getTime() - sessionDate.getTime()) / (1000 * 60 * 60 * 24))

        if (daysDiff === currentStreak) {
          currentStreak++
        } else {
          break
        }
      }
      setStreak(currentStreak)
    }

    // Load mock data (in production, this comes from backend)
    setPatterns(generateMockPatterns())
    setProgressMetrics(generateMockProgressMetrics())
    setConfidenceData(generateMockConfidenceData())
  }, [practiceHistory])

  const lastSession = practiceHistory.length > 0 ? practiceHistory[practiceHistory.length - 1] : null
  const lastSessionDuration = lastSession && lastSession.endTime
    ? Math.round((new Date(lastSession.endTime).getTime() - new Date(lastSession.startTime).getTime()) / 1000 / 60)
    : null

  return (
    <div className="space-y-8">
      {/* Greeting + Action */}
      <div className="bg-gradient-to-r from-primary-50 to-accent-50 rounded-lg p-6 border border-primary-100">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2 text-gray-900">
              Welcome back, {user?.name || 'there'}!
            </h1>
            {user?.goal && (
              <p className="text-lg text-gray-700">
                Ready to work on your <span className="font-semibold capitalize">{user.goal}</span> goal?
              </p>
            )}
          </div>
          <Link
            href="/dashboard/practice"
            className="bg-primary-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors shadow-sm"
          >
            Start Practice
          </Link>
        </div>
      </div>

      {/* Interactive Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <InteractiveCard
          title="Practice Streak"
          value={streak}
          subtitle="days"
          icon={Calendar}
          iconColor="text-primary-500"
          onClick={() => setSelectedCard(selectedCard === 'streak' ? null : 'streak')}
          trend={[1, 2, 3, streak]}
        >
          {selectedCard === 'streak' && (
            <div className="mt-4 pt-4 border-t border-gray-200 text-sm text-gray-600">
              Keep practicing daily to maintain your streak!
            </div>
          )}
        </InteractiveCard>

        <InteractiveCard
          title="Last Session"
          value={lastSessionDuration ? `${lastSessionDuration}m` : '—'}
          subtitle={lastSession ? new Date(lastSession.startTime).toLocaleDateString() : 'No sessions yet'}
          icon={Clock}
          iconColor="text-accent-500"
          onClick={() => setSelectedCard(selectedCard === 'last' ? null : 'last')}
        >
          {selectedCard === 'last' && lastSession && (
            <div className="mt-4 pt-4 border-t border-gray-200 text-sm text-gray-600">
              <p className="mb-2">Mode: <span className="font-medium capitalize">{lastSession.mode}</span></p>
              {lastSession.scenario && (
                <p>Scenario: <span className="font-medium">{lastSession.scenario}</span></p>
              )}
            </div>
          )}
        </InteractiveCard>

        {confidenceData && (
          <InteractiveCard
            title="Confidence Trend"
            value={confidenceData.score}
            subtitle="out of 100"
            icon={TrendingUp}
            iconColor="text-green-500"
            onClick={() => setSelectedCard(selectedCard === 'confidence' ? null : 'confidence')}
            trend={confidenceData.trend}
          >
            {selectedCard === 'confidence' && (
              <div className="mt-4 pt-4 border-t border-gray-200 text-sm text-gray-600">
                <p className="mb-2">Dominant focus: {confidenceData.dominantWeakness}</p>
                <p className="text-xs">Based on recent practice patterns</p>
              </div>
            )}
          </InteractiveCard>
        )}

        <InteractiveCard
          title="Current Focus"
          value={patterns.length > 0 ? patterns[0].name : 'Getting Started'}
          subtitle="Area to practice"
          icon={Target}
          iconColor="text-blue-500"
          onClick={() => setSelectedCard(selectedCard === 'focus' ? null : 'focus')}
        >
          {selectedCard === 'focus' && patterns.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200 text-sm text-gray-600">
              <p>{patterns[0].suggestedPractice}</p>
            </div>
          )}
        </InteractiveCard>
      </div>

      {/* Analytics & Heatmap Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary-500" />
              Performance Heatmap
            </h2>
            <select className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg block p-2.5">
              <option>Last 7 Days</option>
              <option>Last 30 Days</option>
            </select>
          </div>
          <HeatmapView data={generateMockHeatmapData()} duration={300} />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 text-gray-900">Key Metrics</h2>
          <div className="space-y-6">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Average Pace</span>
                <span className="font-medium text-gray-900">145 WPM</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '75%' }}></div>
              </div>
              <p className="text-xs text-gray-500 mt-1">Optimal range: 130-160 WPM</p>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Filler Word Usage</span>
                <span className="font-medium text-gray-900">Low (2%)</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-500 h-2 rounded-full" style={{ width: '90%' }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Clarity Score</span>
                <span className="font-medium text-gray-900">8.5/10</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-purple-500 h-2 rounded-full" style={{ width: '85%' }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Detected Patterns Panel (Expandable) */}
      {patterns.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-primary-500" />
                Detected Patterns
              </h2>
              <p className="text-sm text-gray-600 mt-1">
                Observations from your practice sessions. Click to learn more.
              </p>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {patterns.map((pattern) => (
              <PatternPill key={pattern.id} pattern={pattern} />
            ))}
          </div>
        </div>
      )}

      {/* Progress Overview (Interactive Bars) */}
      {progressMetrics.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 text-gray-900">Progress Overview</h2>
          <p className="text-sm text-gray-600 mb-6">
            Track your improvement across key areas. Each metric shows trends from recent sessions.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {progressMetrics.map((metric) => (
              <ProgressBar key={metric.name} metric={metric} />
            ))}
          </div>
        </div>
      )}

      {/* Practice History Timeline */}
      {practiceHistory.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 text-gray-900">Practice History</h2>
          <div className="space-y-4">
            {practiceHistory.slice(-5).reverse().map((session) => {
              const duration = session.endTime
                ? Math.round((new Date(session.endTime).getTime() - new Date(session.startTime).getTime()) / 1000 / 60)
                : null

              return (
                <Link
                  key={session.id}
                  href={`/dashboard/insights?session=${session.id}`}
                  className="block p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-sm transition-all"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className="font-semibold text-gray-900 capitalize">{session.mode}</span>
                        {session.scenario && (
                          <span className="text-sm text-gray-600">• {session.scenario}</span>
                        )}
                      </div>
                      <p className="text-sm text-gray-600">
                        {new Date(session.startTime).toLocaleDateString()}
                        {duration && ` • ${duration} minutes`}
                      </p>
                      {session.improvementHighlight && (
                        <p className="text-sm text-primary-600 mt-2">
                          ✨ {session.improvementHighlight}
                        </p>
                      )}
                    </div>
                    {session.confidenceScore && (
                      <div className="text-right">
                        <p className="text-2xl font-bold text-gray-900">{session.confidenceScore}</p>
                        <p className="text-xs text-gray-500">confidence</p>
                      </div>
                    )}
                  </div>
                </Link>
              )
            })}
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link
          href="/dashboard/practice"
          className="bg-primary-600 text-white p-6 rounded-lg shadow-sm hover:bg-primary-700 transition-colors"
        >
          <h3 className="text-xl font-semibold mb-2">Start Practice Session</h3>
          <p className="text-primary-100">Begin a new practice session with real-time feedback</p>
        </Link>
        <Link
          href="/dashboard/scenarios"
          className="bg-accent-600 text-white p-6 rounded-lg shadow-sm hover:bg-accent-700 transition-colors"
        >
          <h3 className="text-xl font-semibold mb-2">Try Scenario Simulator</h3>
          <p className="text-accent-100">Practice real-world speaking scenarios</p>
        </Link>
      </div>
    </div>
  )
}
