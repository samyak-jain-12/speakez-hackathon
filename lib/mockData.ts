// Mock data generators for UI components - all data-driven

import {
  DetectedPattern,
  ProgressMetric,
  LiveSignal,
  LiveObservation,
  ConfidenceData,
  HeatmapData,
  SessionData,
} from './types'

export function generateMockPatterns(): DetectedPattern[] {
  return [
    {
      id: '1',
      name: 'Volume Consistency',
      severity: 'medium',
      color: 'green',
      occurrences: 3,
      whenItOccurs: 'During transitions between topics',
      whyItHappens: 'Natural variation when shifting focus or gathering thoughts',
      suggestedPractice: 'Practice maintaining steady volume during topic transitions',
    },
    {
      id: '2',
      name: 'Rapid Pacing',
      severity: 'low',
      color: 'yellow',
      occurrences: 2,
      whenItOccurs: 'When explaining complex ideas',
      whyItHappens: 'Excitement or eagerness to convey information quickly',
      suggestedPractice: 'Try pausing after key points to let ideas settle',
    },
    {
      id: '3',
      name: 'Repetition During Transitions',
      severity: 'low',
      color: 'blue',
      occurrences: 1,
      whenItOccurs: 'When moving between main points',
      whyItHappens: 'Common way to maintain flow and ensure clarity',
      suggestedPractice: 'Practice smooth transitions with brief pauses instead',
    },
    {
      id: '4',
      name: 'Stress Spikes',
      severity: 'medium',
      color: 'yellow',
      occurrences: 4,
      whenItOccurs: 'At the beginning of new sections',
      whyItHappens: 'Natural response when starting new topics or sections',
      suggestedPractice: 'Begin with a deep breath before each new section',
    },
  ]
}

export function generateMockProgressMetrics(): ProgressMetric[] {
  return [
    {
      name: 'Volume Control',
      current: 75,
      target: 80,
      trend: 'up',
      sessions: [65, 70, 72, 75],
      tooltip: 'Observed across last 4 sessions',
    },
    {
      name: 'Pace Stability',
      current: 68,
      target: 70,
      trend: 'stable',
      sessions: [65, 68, 67, 68],
      tooltip: 'Observed across last 4 sessions',
    },
    {
      name: 'Filler Word Reduction',
      current: 82,
      target: 85,
      trend: 'up',
      sessions: [70, 75, 78, 82],
      tooltip: 'Observed across last 4 sessions',
    },
    {
      name: 'Eye Contact Consistency',
      current: 60,
      target: 75,
      trend: 'up',
      sessions: [50, 55, 58, 60],
      tooltip: 'Observed across last 4 sessions',
    },
  ]
}

export function generateMockLiveSignals(): LiveSignal[] {
  return [
    {
      name: 'Stress',
      value: 45,
      max: 100,
      color: 'yellow',
      tooltip: 'Current stress level based on voice patterns',
      status: 'normal',
    },
    {
      name: 'Pace',
      value: 145,
      max: 200,
      color: 'blue',
      tooltip: 'Words per minute - ideal range: 120-160',
      status: 'normal',
    },
    {
      name: 'Pitch',
      value: 65,
      max: 100,
      color: 'purple',
      tooltip: 'Voice pitch variation for engagement',
      status: 'normal',
    },
    {
      name: 'Volume',
      value: 70,
      max: 100,
      color: 'green',
      tooltip: 'Speaking volume consistency',
      status: 'normal',
    },
    {
      name: 'Fillers',
      value: 3,
      max: 10,
      color: 'orange',
      tooltip: 'Filler words detected (um, uh, like)',
      status: 'normal',
    },
    {
      name: 'Eye Contact',
      value: 75,
      max: 100,
      color: 'blue',
      tooltip: 'Eye contact with camera',
      status: 'normal',
    },
    {
      name: 'Posture',
      value: 80,
      max: 100,
      color: 'green',
      tooltip: 'Posture quality',
      status: 'normal',
    },
    {
      name: 'Gestures',
      value: 5,
      max: 10,
      color: 'purple',
      tooltip: 'Natural gesture activity',
      status: 'normal',
    },
  ]
}

export function generateMockObservations(): LiveObservation[] {
  return [
    {
      id: '1',
      timestamp: Date.now() - 30000,
      message: 'Pace increasing here',
      type: 'pace',
      suggestion: 'Try pausing after this point',
    },
    {
      id: '2',
      timestamp: Date.now() - 20000,
      message: 'Stress spike detected',
      type: 'stress',
      suggestion: 'Take a breath - you\'re doing great',
    },
    {
      id: '3',
      timestamp: Date.now() - 10000,
      message: 'Volume consistency observed',
      type: 'volume',
    },
  ]
}

export function generateMockConfidenceData(): ConfidenceData {
  return {
    score: 72,
    factors: [
      { name: 'Pace Control', impact: 0.25 },
      { name: 'Volume Consistency', impact: 0.20 },
      { name: 'Eye Contact', impact: 0.15 },
      { name: 'Stress Management', impact: 0.20 },
      { name: 'Clarity', impact: 0.20 },
    ],
    dominantWeakness: 'Stress spikes during transitions',
    trend: [65, 68, 70, 72],
  }
}

export function generateMockHeatmapData(): HeatmapData[] {
  const data: HeatmapData[] = []
  for (let i = 0; i < 60; i++) {
    data.push({
      timestamp: i * 5,
      stress: Math.random() * 100,
      pace: 120 + Math.random() * 60,
      repetition: Math.random() * 50,
      volume: 60 + Math.random() * 40,
    })
  }
  return data
}
