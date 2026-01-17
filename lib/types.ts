// Data structures for UI components - all data-driven

export interface PracticeMetrics {
  stress: number[]
  stuttering: number[]
  repetition: number[]
  pitch: number[]
  volume: number[]
  pace: number[]
  fillerWords: number[]
  posture?: number[]
  eyeContact?: number[]
  gestures?: number[]
}

export interface SessionData {
  id: string
  mode: string
  scenario?: string
  startTime: Date
  endTime?: Date
  metrics: PracticeMetrics
  insights?: string[]
  confidenceScore?: number
  dominantWeakness?: string
  improvementHighlight?: string

  videoMetrics?: VideoMetrics
  confidenceMetrics?: ConfidenceMetric
  alignmentMetrics?: AlignmentMetric
  drillMetrics?: any
}

export interface VideoMetrics {
  eye_contact_score: number
  posture_alert: boolean
  gesture_intensity: 'low' | 'medium' | 'high'
}

export interface ConfidenceMetric {
  confidence_score: number
  dominant_weakness: string
  factors: {
    name: string
    impact: number
    score: number
  }[]
  key_observations: string[]
  summary_text: string
}

export interface AlignmentMetric {
  tone_match: boolean
  suggestion: string
}

export interface DetectedPattern {
  id: string
  name: string
  severity: 'low' | 'medium' | 'high'
  color: 'green' | 'yellow' | 'blue'
  occurrences: number
  whenItOccurs: string
  whyItHappens: string
  suggestedPractice: string
}

export interface ProgressMetric {
  name: string
  current: number
  target: number
  trend: 'up' | 'down' | 'stable'
  sessions: number[]
  tooltip: string
}

export interface LiveSignal {
  name: string
  value: number
  max: number
  color: string
  tooltip: string
  status: 'normal' | 'elevated' | 'high'
}

export interface LiveObservation {
  id: string
  timestamp: number
  message: string
  type: 'pace' | 'stress' | 'repetition' | 'volume' | 'general'
  suggestion?: string
}

export interface ConfidenceData {
  score: number
  factors: {
    name: string
    impact: number
    score?: number
  }[]
  dominantWeakness: string
  trend: number[]
}

export interface HeatmapData {
  timestamp: number
  stress: number
  pace: number
  repetition: number
  volume: number
}
