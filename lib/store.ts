import { create } from 'zustand'

export interface User {
  name: string
  email: string
  hasSpeechDisorder?: boolean
  goal?: string
  currentAccent?: string
  targetAccent?: string
}

export interface PracticeSession {
  id: string
  mode: string
  scenario?: string
  startTime: Date
  endTime?: Date
  metrics: {
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
  videoMetrics?: {
    eye_contact_score: number
    posture_alert: boolean
    gesture_intensity: 'low' | 'medium' | 'high'
  }
  confidenceMetrics?: {
    confidence_score: number
    dominant_weakness: string
    factors?: {
      name: string
      impact: number
      score?: number
    }[]
    key_observations?: string[]
    summary_text?: string
  }
  alignmentMetrics?: {
    tone_match: boolean
    suggestion: string
  }
  drillMetrics?: {
    task: string
    feedback: string
  }
}

interface AppState {
  user: User | null
  currentSession: PracticeSession | null
  practiceHistory: PracticeSession[]
  setUser: (user: User | null) => void
  startSession: (mode: string, scenario?: string) => void
  endSession: () => PracticeSession | null
  updateSessionMetrics: (metrics: Partial<PracticeSession['metrics']>) => void
  addSession: (session: PracticeSession) => void
  updateHistorySession: (id: string, data: Partial<PracticeSession>) => void
}

export const useStore = create<AppState>((set) => ({
  user: null,
  currentSession: null,
  practiceHistory: [],
  setUser: (user) => set({ user }),
  startSession: (mode, scenario) => set({
    currentSession: {
      id: Date.now().toString(),
      mode,
      scenario,
      startTime: new Date(),
      metrics: {
        stress: [],
        stuttering: [],
        repetition: [],
        pitch: [],
        volume: [],
        pace: [],
        fillerWords: [],
      },
    },
  }),
  endSession: () => {
    let endedSession: PracticeSession | null = null
    set((state) => {
      if (state.currentSession) {
        endedSession = {
          ...state.currentSession,
          endTime: new Date(),
        }
        return {
          currentSession: null,
          practiceHistory: [...state.practiceHistory, endedSession!],
        }
      }
      return { currentSession: null }
    })
    return endedSession
  },
  updateSessionMetrics: (newMetrics) => set((state) => {
    if (state.currentSession) {
      const updatedMetrics = { ...state.currentSession.metrics }

      // Iterate through new keys and append to existing arrays
      Object.keys(newMetrics).forEach((key) => {
        const value = (newMetrics as any)[key]
        if (Array.isArray(value)) {
          // @ts-ignore
          updatedMetrics[key] = [...(updatedMetrics[key] || []), ...value]
        }
      })

      return {
        currentSession: {
          ...state.currentSession,
          metrics: updatedMetrics,
        },
      }
    }
    return state
  }),
  addSession: (session) => set((state) => ({
    practiceHistory: [...state.practiceHistory, session],
  })),
  updateHistorySession: (id, data) => set((state) => ({
    practiceHistory: state.practiceHistory.map(session =>
      session.id === id ? { ...session, ...data } : session
    )
  })),
}))
