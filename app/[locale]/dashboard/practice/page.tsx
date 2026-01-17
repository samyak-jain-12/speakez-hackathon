'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useStore } from '@/lib/store'
import { OrchestratorAgent } from '@/lib/agents/orchestrator'
import {
  Mic,
  Video,
  VideoOff,
  Play,
  Pause,
  Square,
  MessageSquare,
  Target,
  Clock,
  Briefcase,
  Users,
  Phone,
  Utensils,
  Lightbulb,
  ArrowRight,
  Sparkles
} from 'lucide-react'
import LiveSignalBar from '@/components/LiveSignalBar'
import LiveObservationFeed from '@/components/LiveObservationFeed'
import SpeechPatternAwareness from '@/components/SpeechPatternAwareness'
import ScenarioEventCard from '@/components/ScenarioEventCard'
import {
  generateMockLiveSignals,
} from '@/lib/mockData'
import { LiveSignal, LiveObservation } from '@/lib/types'

type LiveTranscriptItem = {
  timestamp: string
  text: string
}

// --- Data from Modes Page ---
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

// --- Data from Scenarios Page ---
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
  {
    id: 'debate',
    title: 'Debate',
    icon: MessageSquare,
    description: 'Practice argumentation and counterpoints',
    prompt: 'Argue for or against a topic. Be prepared for counterarguments.',
    mode: 'debate',
  },
  {
    id: 'sales-call',
    title: 'Sales Call',
    icon: Phone,
    description: 'Practice objection handling',
    prompt: 'Sell a product to a skeptical lead. Handle their objections gracefully.',
    mode: 'professional',
  },
]

export default function PracticeSessionPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { currentSession, startSession, endSession, updateSessionMetrics, updateHistorySession, user } = useStore()

  // States
  const [setupStep, setSetupStep] = useState<'selection' | 'preview'>('selection')
  const [activeTab, setActiveTab] = useState<'modes' | 'scenarios'>('modes')
  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [cameraEnabled, setCameraEnabled] = useState(true)

  // Session Configuration
  const [selectedMode, setSelectedMode] = useState<string | null>(null)
  const [selectedScenario, setSelectedScenario] = useState<typeof scenarios[0] | null>(null)
  const [selectedLanguage, setSelectedLanguage] = useState<string>('en-US')

  // Real-time Data
  const [currentMetrics, setCurrentMetrics] = useState<any>(null)
  const [reassurance, setReassurance] = useState<string>('')
  const [liveSignals, setLiveSignals] = useState<LiveSignal[]>([])
  const [liveObservations, setLiveObservations] = useState<LiveObservation[]>([])
  const [liveTranscriptItems, setLiveTranscriptItems] = useState<LiveTranscriptItem[]>([])
  const liveTranscriptRef = useRef<LiveTranscriptItem[]>([]) // Ref for real-time access
  const [sessionStartTime, setSessionStartTime] = useState<number>(Date.now())
  const [interimTranscript, setInterimTranscript] = useState<string>('') // For real-time feedback
  const [detectedPatterns, setDetectedPatterns] = useState<string[]>([])
  const [scenarioEvents, setScenarioEvents] = useState<any[]>([])

  // Refs
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const orchestratorRef = useRef<OrchestratorAgent | null>(null)
  const observationIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const analysisIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const scenarioIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Speech Recognition Refs
  const recognitionRef = useRef<any>(null)
  const isRecordingRef = useRef(false)
  const transcriptRef = useRef<HTMLParagraphElement | null>(null)
  const eyeContactRef = useRef(100) // Simulated Eye Contact State

  // Sync ref with state for event listeners
  useEffect(() => {
    isRecordingRef.current = isRecording
  }, [isRecording])

  useEffect(() => {
    orchestratorRef.current = new OrchestratorAgent()
    setLiveSignals(generateMockLiveSignals()) // Keep initial mock signals for UI shell

    // Check URL params for quick start
    const urlScenario = searchParams.get('scenario')
    const urlMode = searchParams.get('mode')

    if (urlScenario) {
      const found = scenarios.find(s => s.id === urlScenario)
      if (found) {
        handleScenarioSelect(found)
      }
    } else if (urlMode) {
      setSelectedMode(urlMode)
      setSetupStep('preview')
    }

    return () => cleanup()
  }, [])

  const cleanup = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
    }
    if (observationIntervalRef.current) {
      clearInterval(observationIntervalRef.current)
    }
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current)
    }
    if (scenarioIntervalRef.current) {
      clearInterval(scenarioIntervalRef.current)
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop() // Ensure zombie instances are killed
    }
  }

  // Effect to attach stream to video element when it becomes available
  useEffect(() => {
    if (isRecording && cameraEnabled && videoRef.current && mediaStreamRef.current) {
      videoRef.current.srcObject = mediaStreamRef.current
    }
  }, [isRecording, cameraEnabled])

  // --- Logic Handlers ---

  const handleModeSelect = (modeId: string) => {
    setSelectedMode(modeId)
    setSelectedScenario(null)
    setSetupStep('preview')
  }

  const handleScenarioSelect = (scenario: typeof scenarios[0]) => {
    setSelectedScenario(scenario)
    setSelectedMode(scenario.mode) // Scenario implies a mode
    setSetupStep('preview')
  }

  // Helper to safely start recognition
  const startSpeechRecognition = () => {
    if (typeof window === 'undefined') return

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      alert("Speech Recognition not supported")
      return
    }

    // Cleanup old instance if exists
    if (recognitionRef.current) {
      try {
        recognitionRef.current.onend = null // prevent recursive restart during manual cleanup
        recognitionRef.current.abort()
      } catch (e) { console.error(e) }
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = selectedLanguage

    recognition.onstart = () => {
      console.log("SpeechRecognition started (Fresh Instance)")
    }

    recognition.onresult = (event: any) => {
      let interimTranscriptChunk = ''
      let finalTranscriptChunk = ''

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscriptChunk += event.results[i][0].transcript + ' '
        } else {
          interimTranscriptChunk += event.results[i][0].transcript
        }
      }

      setInterimTranscript(interimTranscriptChunk)

      if (finalTranscriptChunk) {
        // Calculate relative time (MM:SS)
        const elapsedSeconds = Math.floor((Date.now() - sessionStartTime) / 1000)
        const minutes = Math.floor(elapsedSeconds / 60).toString().padStart(2, '0')
        const seconds = (elapsedSeconds % 60).toString().padStart(2, '0')
        const timestamp = `${minutes}:${seconds}`

        setLiveTranscriptItems(prev => {
          const newState = [...prev, { timestamp, text: finalTranscriptChunk }]
          liveTranscriptRef.current = newState // Update Ref
          return newState
        })
        setInterimTranscript('')
      }
    }

    recognition.onerror = (event: any) => {
      console.error("Speech Recognition Error", event.error)
      if (event.error === 'not-allowed') {
        setIsRecording(false) // Stop if permission denied
        alert("Microphone denied.")
      }
    }

    recognition.onend = () => {
      console.log("SpeechRecognition ended")
      if (isRecordingRef.current) {
        console.log("Auto-restarting with new instance in 200ms...")
        setTimeout(() => {
          if (isRecordingRef.current) {
            startSpeechRecognition() // Recursive restart with FRESH instance
          }
        }, 200)
      }
    }

    try {
      recognition.start()
      recognitionRef.current = recognition
    } catch (e) {
      console.error("Failed to start new recognition instance", e)
    }
  }

  // Effect to update language if active
  useEffect(() => {
    if (isRecording && recognitionRef.current && recognitionRef.current.lang !== selectedLanguage) {
      console.log("Language changed, restarting recognition...")
      startSpeechRecognition()
    }
  }, [selectedLanguage])

  // Update ref when isRecording changes so onend knows whether to restart
  useEffect(() => {
    isRecordingRef.current = isRecording
  }, [isRecording])

  const startRecording = async () => {
    console.log("Start Session Block Triggered")
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: cameraEnabled,
      })

      mediaStreamRef.current = stream
      chunksRef.current = []

      if (cameraEnabled && videoRef.current) {
        videoRef.current.srcObject = stream
      }

      // Audio Setup for visualization (optional)
      audioContextRef.current = new AudioContext()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 2048
      source.connect(analyserRef.current)

      // MediaRecorder Setup for Agent Pipeline
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' })
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      // Start recording and firing dataavailable event every 2 seconds
      mediaRecorder.start(2000)

      // Start Speech Recognition
      setLiveTranscriptItems([])
      liveTranscriptRef.current = [] // Reset Ref
      setInterimTranscript('')
      setSessionStartTime(Date.now())
      startSpeechRecognition()

      // Start Session in Store
      startSession(selectedMode || 'conversation', selectedScenario?.title)

      setIsRecording(true)
      setIsPaused(false)
      setLiveObservations([])
      setDetectedPatterns([])

      // Start Analysis Loop
      analyzeAudio()

      // Start Simulators
      startSimulators()

    } catch (error) {
      console.error('Error accessing media devices:', error)
      alert('Please allow microphone access to start practicing.')
    }
  }

  const analyzeAudio = () => {
    // Legacy function, logic relocated to useEffect for polling
  }

  // Use Effect to drive the analysis loop
  useEffect(() => {
    if (isRecording && !isPaused) {
      analysisIntervalRef.current = setInterval(async () => {
        if (!analyserRef.current) return
        const bufferLength = analyserRef.current.fftSize
        const dataArray = new Float32Array(bufferLength)
        analyserRef.current.getFloatTimeDomainData(dataArray)

        // Convert to simple array for JSON
        const audioData = Array.from(dataArray)

        try {
          const dspResponse = await fetch('/api/dsp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              audio_data: audioData,
              sample_rate: audioContextRef.current?.sampleRate || 16000
            })
          })

          if (dspResponse.ok) {
            const metrics = await dspResponse.json()

            // --- Calculate Real-Time NLP Metrics ---
            const fullText = liveTranscriptRef.current.map(i => i.text).join(' ').toLowerCase()
            const fillerRegex = /\b(um|uh|like|you know|basically|actually|literally)\b/g
            const fillerCount = (fullText.match(fillerRegex) || []).length

            // Simple Clarity Heuristic
            const wordCount = fullText.split(' ').length || 1
            const fillerDensity = (fillerCount / wordCount) * 100
            const clarityScore = Math.max(0, Math.min(100, 100 - (fillerDensity * 5)))

            // Update State with basic DSP + NLP
            setCurrentMetrics({
              stress: metrics.stress_index,
              pitch: metrics.pitch,
              volume: metrics.volume,
              pace: metrics.speaking_pace,
              fillerWords: fillerCount // Store cumulative count
            })

            // PERSIST METRICS TO STORE (Fixes 0 WPM issue)
            updateSessionMetrics({
              stress: [metrics.stress_index],
              pitch: [metrics.pitch],
              volume: [metrics.volume],
              pace: [metrics.speaking_pace],
              fillerWords: [fillerCount]
            })

            // Update Live Signal Bar
            setLiveSignals([
              {
                name: 'Pace',
                value: metrics.speaking_pace,
                max: 200,
                color: metrics.speaking_pace > 160 ? '#EF4444' : '#10B981',
                tooltip: 'Words per minute',
                status: metrics.speaking_pace > 160 ? 'high' : 'normal'
              },
              {
                name: 'Filler Words',
                value: fillerCount * 10, // Scale for visibility (max 10 fillers = 100%)
                max: 100,
                color: fillerCount > 5 ? '#EF4444' : '#F59E0B',
                tooltip: `${fillerCount} detected`,
                status: fillerCount > 5 ? 'elevated' : 'normal'
              },
              {
                name: 'Clarity',
                value: clarityScore,
                max: 100,
                color: clarityScore < 70 ? '#EF4444' : '#10B981',
                tooltip: 'Speech clarity score',
                status: clarityScore < 70 ? 'elevated' : 'normal'
              },
              {
                name: 'Stress',
                value: metrics.stress_index * 100,
                max: 100,
                color: metrics.stress_index > 0.7 ? '#EF4444' : '#10B981',
                tooltip: 'Voice stress analysis',
                status: metrics.stress_index > 0.7 ? 'elevated' : 'normal'
              },
              {
                name: 'Volume',
                value: metrics.volume * 100,
                max: 100,
                color: metrics.volume < 0.2 ? '#F59E0B' : '#10B981',
                tooltip: 'Speaking volume',
                status: 'normal'
              },
              // Simulated Agent 4 Live Metrics
              {
                name: 'Eye Contact',
                value: eyeContactRef.current,
                max: 100,
                color: eyeContactRef.current > 50 ? '#3B82F6' : '#9CA3AF',
                tooltip: 'Visual engagement score',
                status: eyeContactRef.current < 30 ? 'elevated' : 'normal'
              },
              {
                name: 'Posture',
                value: cameraEnabled ? 90 : 0, // Simulate stable posture
                max: 100,
                color: cameraEnabled ? '#10B981' : '#9CA3AF',
                tooltip: 'Body language stability',
                status: 'normal'
              }
            ])
          }
        } catch (e) {
          console.error("Realtime DSP Error", e)
        }
      }, 1000) // Run every 1000ms for slower, cleaner updates
    } else {
      if (analysisIntervalRef.current) clearInterval(analysisIntervalRef.current)
    }

    return () => {
      if (analysisIntervalRef.current) clearInterval(analysisIntervalRef.current)
    }
  }, [isRecording, isPaused])

  const startSimulators = () => {
    // 1. Observations Simulator
    observationIntervalRef.current = setInterval(() => {
      const newObservation: LiveObservation = {
        id: Date.now().toString(),
        timestamp: Date.now(),
        message: ['Pace increasing', 'Stress spike detected', 'Good clarity', 'Volume consistent'][Math.floor(Math.random() * 4)],
        type: ['pace', 'stress', 'general', 'volume'][Math.floor(Math.random() * 4)] as any,
        suggestion: Math.random() > 0.7 ? 'Keep it up!' : undefined,
      }
      setLiveObservations(prev => [newObservation, ...prev].slice(0, 5))
    }, 5000)

    // 2. Scenario Event Simulator (if scenario active)
    if (selectedScenario) {
      // Run Agent 7 every 15 seconds to simulate pressure events
      const eventInterval = setInterval(async () => {
        // Get larger context window (last 15 items or ~30-60s)
        const recentSpeech = liveTranscriptItems.length > 0
          ? liveTranscriptItems.slice(-15).map(i => i.text).join(' ')
          : " "

        try {
          const res = await fetch('/api/agent/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              scenario: selectedScenario.title,
              current_speech: recentSpeech,
              live_stats: {
                confidence_score: currentMetrics?.stress ? 100 - (currentMetrics.stress * 100) : 70, // derive approx confidence
                pace: currentMetrics?.pace || 140
              },
              timestamp: Date.now()
            })
          })

          if (res.ok) {
            const eventData = await res.json()
            // Expect { scenario_event: "...", expected_response_type: "..." }
            if (eventData && eventData.scenario_event) {
              const newEvent: any = {
                id: Date.now().toString(),
                title: "Scenario Event",
                description: eventData.scenario_event,
                type: 'intervention',
                timestamp: Date.now(),
                advice: eventData.expected_response_type ? `Try to be ${eventData.expected_response_type.replace('_', ' ')}` : undefined
              }
              setScenarioEvents(prev => [newEvent, ...prev].slice(0, 3)) // Keep last 3 events
            }
          }
        } catch (e) {
          console.error("Agent 7 Simulation Failed", e)
        }

      }, 15000) // 15 seconds loop

      scenarioIntervalRef.current = eventInterval
    }
  }

  const stopRecording = async () => {
    isRecordingRef.current = false // PREVENT auto-restart in onend
    if (recognitionRef.current) {
      try {
        recognitionRef.current.onend = null // Prevent restart
        recognitionRef.current.abort()
      } catch (e) { console.error(e) }
      recognitionRef.current = null
    }

    cleanup()
    setIsRecording(false)
    setIsPaused(false)
    const endedSession = endSession()

    // 1. Stop Recorder and get Blob
    const videoBlob = new Blob(chunksRef.current, { type: 'video/webm' })
    console.log(`Video Blob Size: ${videoBlob.size} bytes`)

    // 2. Upload Video to Agent 4 (Parallel with other mock setups)
    let agent4Result: any = null

    try {
      if (cameraEnabled && videoBlob.size > 0) {
        console.log("Uploading video to Agent 4...")
        const formData = new FormData()
        formData.append('file', videoBlob)
        formData.append('fileName', `session_${endedSession?.id || Date.now()}.webm`)

        // Note: We don't have a sessionId from the backend yet for *Agent 4's* chat, 
        // so the API route will create one.

        const videoRes = await fetch('/api/agent/video', {
          method: 'POST',
          body: formData
        })

        if (videoRes.ok) {
          const videoData = await videoRes.json()
          // Agent 4 returns { data: { answer: "JSON String" } }
          if (videoData.data?.answer) {
            try {
              const cleanJson = videoData.data.answer.replace(/```json/g, '').replace(/```/g, '').trim()
              agent4Result = JSON.parse(cleanJson)
            } catch (e) { console.error("Agent 4 JSON Parse error", e) }
          }
        } else {
          console.error("Video Upload Failed", await videoRes.text())
        }
      }
    } catch (e) {
      console.error("Agent 4 Flow Failed", e)
    }

    // 3. Fallback Mocking if Agent 4 failed or camera disabled
    if (!agent4Result) {
      console.log("Using Mock Agent 4 Data (Fallback)")
      // Logic matched from Analyze Agent Mocking
      const stress = currentMetrics?.stress || 0
      const pace = currentMetrics?.pace || 130
      agent4Result = {
        eye_contact_score: stress < 0.3 ? 0.85 : 0.45,
        posture_alert: stress > 0.6,
        gesture_intensity: pace > 160 ? 'high' : 'medium'
      }
    }

    // 4. Construct Agent 2 & 3 inputs
    // Calculate Averages from the full session for Agent 5
    const metrics = endedSession?.metrics
    const avg = (arr: number[]) => arr && arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0

    const agent2Data = {
      pitch: avg(metrics?.pitch || []),
      volume: avg(metrics?.volume || []),
      pace: avg(metrics?.pace || []),
      stress_index: avg(metrics?.stress || []),
      repetition_count: 0
    }

    const agent3Data = {
      filler_words: currentMetrics?.fillerWords || 0,
      clarity_score: 0.85,
      key_phrases_repeated: []
    }

    // 5. Call Analysis Endpoint with REAL (or confidently mocked) Agent 4 Data
    try {
      const fullTranscript = liveTranscriptItems.map(item => item.text).join(' ')
      const endedSessionId = endedSession?.id

      if (!endedSessionId) return

      // Show analyzing state (optional, or just wait)

      const res = await fetch('/api/agent/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent2: agent2Data,
          agent3: agent3Data, // Now ignored by backend in favor of real Agent 3 but kept for structure
          agent4: agent4Result,
          mode: selectedMode,
          transcript: fullTranscript
        })
      })

      if (res.ok) {
        const data = await res.json()
        console.log("Analysis Result:", data)

        if (data.confidenceMetrics || data.alignmentMetrics) {
          updateHistorySession(endedSessionId, {
            confidenceMetrics: data.confidenceMetrics,
            alignmentMetrics: data.alignmentMetrics,
            videoMetrics: agent4Result, // SAVING REAL VIDEO METRICS
            drillMetrics: data.drillMetrics
          })
          console.log("Session updated with new metrics")
        }
      } else {
        console.error("Analysis API failed", await res.text())
      }

      router.push(`/dashboard/insights?session=${endedSessionId}`)

    } catch (e) {
      console.error("Error triggering analysis", e)
      // Fallback redirect even if analysis fails
      if (endedSession?.id) router.push(`/dashboard/insights?session=${endedSession.id}`)
    }
  }

  // --- Render ---

  // 1. Selection Screen
  if (setupStep === 'selection') {
    return (
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-gray-900">Start a Live Session</h1>

        {/* Tabs */}
        <div className="flex border-b border-gray-200 mb-8">
          <button
            onClick={() => setActiveTab('modes')}
            className={`pb-4 px-6 text-lg font-medium transition-colors ${activeTab === 'modes'
              ? 'border-b-2 border-primary-600 text-primary-600'
              : 'text-gray-600 hover:text-gray-900'
              }`}
          >
            Purpose Modes
          </button>
          <button
            onClick={() => setActiveTab('scenarios')}
            className={`pb-4 px-6 text-lg font-medium transition-colors ${activeTab === 'scenarios'
              ? 'border-b-2 border-primary-600 text-primary-600'
              : 'text-gray-600 hover:text-gray-900'
              }`}
          >
            Scenario Simulator
          </button>
        </div>

        {/* Modes Tab */}
        {activeTab === 'modes' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {modes.map((mode) => (
              <button
                key={mode.id}
                onClick={() => handleModeSelect(mode.id)}
                className="text-left bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:border-primary-500 hover:shadow-md transition-all"
              >
                <div className="flex items-center gap-3 mb-3">
                  <Target className="w-6 h-6 text-primary-600" />
                  <h3 className="text-xl font-semibold text-gray-900">{mode.name}</h3>
                </div>
                <p className="text-gray-600 mb-4 text-sm">{mode.description}</p>
                <div className="flex items-center gap-2 text-sm text-gray-500 bg-gray-50 p-2 rounded">
                  <Clock className="w-4 h-4" />
                  <span>Ideal Pace: {mode.idealPace}</span>
                </div>
              </button>
            ))}
          </div>
        )}

        {/* Scenarios Tab */}
        {activeTab === 'scenarios' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {scenarios.map((scenario) => {
              const Icon = scenario.icon
              return (
                <button
                  key={scenario.id}
                  onClick={() => handleScenarioSelect(scenario)}
                  className="text-left bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:border-accent-500 hover:shadow-md transition-all"
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 bg-primary-100 rounded-lg">
                      <Icon className="w-6 h-6 text-primary-600" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900">{scenario.title}</h3>
                  </div>
                  <p className="text-gray-600 mb-4 text-sm line-clamp-2">{scenario.description}</p>
                  <div className="text-sm bg-gray-50 p-3 rounded text-gray-700">
                    <span className="font-medium">Prompt:</span> {scenario.prompt}
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>
    )
  }

  // 2. Preview / Setup Screen
  if (setupStep === 'preview' && !isRecording) {
    const activeMode = modes.find(m => m.id === selectedMode)

    return (
      <div className="max-w-4xl mx-auto">
        <button
          onClick={() => setSetupStep('selection')}
          className="mb-6 text-gray-600 hover:text-gray-900 flex items-center gap-2"
        >
          ← Back to Selection
        </button>

        <h1 className="text-3xl font-bold mb-2 text-gray-900">Ready to Practice?</h1>
        <p className="text-gray-600 mb-8">
          You chose <strong>{selectedScenario ? selectedScenario.title : activeMode?.name}</strong>.
          Adjust your settings below before starting.
        </p>

        {selectedScenario && (
          <div className="bg-blue-50 border border-blue-200 p-6 rounded-lg mb-6">
            <h2 className="text-lg font-semibold mb-2 text-blue-900 flex items-center gap-2">
              <Sparkles className="w-5 h-5" />
              Scenario Prompt
            </h2>
            <p className="text-blue-800 text-lg">{selectedScenario.prompt}</p>
          </div>
        )}

        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-6">
          <div className="flex items-center justify-between mb-4">
            <label className="text-lg font-medium text-gray-700 flex items-center gap-2">
              <Video className="w-5 h-5" />
              Camera Feed
            </label>
            <button
              onClick={() => setCameraEnabled(!cameraEnabled)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${cameraEnabled ? 'bg-primary-100 text-primary-700' : 'bg-gray-100 text-gray-600'
                }`}
            >
              {cameraEnabled ? 'Enabled' : 'Disabled'}
            </button>
          </div>

          <div className="bg-black rounded-lg overflow-hidden aspect-video relative flex items-center justify-center">
            {cameraEnabled ? (
              <div className="w-full h-full bg-gray-900 flex items-center justify-center">
                <p className="text-gray-400 text-sm">Camera will start when session begins</p>
              </div>
            ) : (
              <div className="text-white text-center opacity-50">
                <VideoOff className="w-16 h-16 mx-auto mb-2" />
                <p>Camera Disabled</p>
              </div>
            )}
          </div>
        </div>

        {/* Language Selector */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-6">
          <h3 className="text-lg font-medium text-gray-700 mb-3 flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Spoken Language
          </h3>
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg bg-gray-50 text-gray-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all"
          >
            <option value="en-US">English (US)</option>
            <option value="en-IN">English (India)</option>
            <option value="es-ES">Spanish (Spain)</option>
            <option value="fr-FR">French</option>
            <option value="de-DE">German</option>
            <option value="hi-IN">Hindi</option>
            <option value="zh-CN">Chinese (Simplified)</option>
          </select>
        </div>

        <button
          onClick={startRecording}
          className="w-full bg-primary-600 text-white py-4 rounded-lg text-lg font-semibold hover:bg-primary-700 transition-all flex items-center justify-center gap-2 shadow-lg"
        >
          <Mic className="w-6 h-6" />
          Start Session
        </button>
      </div>
    )
  }

  // 3. Active Recording Interface - Thrilling HerTech UI
  return (
    <div className="w-full h-[calc(100vh-64px)] overflow-hidden bg-[radial-gradient(ellipse_at_top_left,_var(--tw-gradient-stops))] from-cyan-50 via-white to-blue-50 flex flex-col relative">
      {/* Animated Background Elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] bg-purple-200/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-[-10%] left-[-10%] w-[600px] h-[600px] bg-blue-200/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 w-full h-full flex flex-col z-10">
        {/* Header with Pulse & Mode */}
        <div className="flex items-center justify-between mb-6 shrink-0">
          <div className="flex items-center gap-5">
            <div className="relative">
              <div className="w-4 h-4 bg-red-500 rounded-full animate-ping absolute inset-0 opacity-75"></div>
              <div className="w-4 h-4 bg-gradient-to-br from-red-500 to-red-600 rounded-full relative shadow-[0_0_15px_rgba(239,68,68,0.5)]"></div>
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-3xl font-black text-gray-900 tracking-tight leading-none">Live Session</h1>
                <span className="px-3 py-1 bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-[10px] font-bold uppercase tracking-widest rounded-full shadow-lg">Team HerTech</span>
              </div>
              <p className="text-sm font-medium text-gray-500 mt-1 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
                {selectedScenario ? 'Scenario Simulation' : 'Free Practice'} Mode
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="px-3 py-1.5 bg-gray-100 rounded-md text-xs font-mono font-medium text-gray-600">
              {new Date(Date.now() - sessionStartTime).toISOString().substr(14, 5)}
            </span>
            <button
              onClick={stopRecording}
              className="px-4 py-1.5 bg-red-50 text-red-600 border border-red-100 rounded-lg text-sm font-semibold hover:bg-red-100 transition-colors"
            >
              End Session
            </button>
          </div>
        </div>

        {/* Main Grid Layout */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-0">

          {/* LEFT COLUMN: Video & Transcript (8 cols) */}
          <div className="lg:col-span-8 flex flex-col gap-4 min-h-0">

            {/* Top Stats Bar (Glass Card) */}
            <div className="bg-white rounded-2xl p-4 shadow-sm border border-gray-200">
              <LiveSignalBar signals={liveSignals} />
            </div>

            {/* Video Feed (Cinematic & Glowing) */}
            <div className="flex-1 bg-gray-900 rounded-3xl overflow-hidden relative shadow-[0_20px_50px_-12px_rgba(0,0,0,0.5)] ring-1 ring-white/10 group min-h-[400px] border border-white/20">
              {cameraEnabled ? (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover transition-transform duration-[2s] hover:scale-105"
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center bg-gray-900 text-white/40">
                  <Mic className="w-20 h-20 mb-4 stroke-1" />
                  <p className="font-light tracking-wide">Audio Only Mode</p>
                </div>
              )}

              {/* Floating Controls */}
              <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 z-20">
                <button
                  onClick={() => setIsPaused(!isPaused)}
                  className="w-12 h-12 flex items-center justify-center bg-white/10 backdrop-blur-md hover:bg-white/90 hover:text-gray-900 text-white rounded-full border border-white/20 transition-all shadow-lg text-white"
                  title={isPaused ? "Resume" : "Pause"}
                >
                  {isPaused ? <Play className="w-5 h-5 fill-current" /> : <Pause className="w-5 h-5 fill-current" />}
                </button>
              </div>
            </div>

            {/* Transcript Box (Enlarged & Animated) */}
            <div className="bg-white/80 backdrop-blur-xl p-6 rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-white/50 h-80 overflow-y-auto flex flex-col-reverse relative group transition-all hover:shadow-[0_8px_30px_rgb(0,0,0,0.08)]">
              <div className="absolute top-4 right-4 text-[10px] font-black text-blue-400 uppercase tracking-[0.2em] bg-blue-50 px-3 py-1.5 rounded-lg border border-blue-100/50">Live Transcript</div>
              <div ref={transcriptRef} className="space-y-4 pr-2">
                {liveTranscriptItems.map((item, idx) => (
                  <div key={idx} className="flex gap-4 group/item transition-opacity duration-500">
                    <span className="text-gray-300 font-mono text-xs shrink-0 select-none pt-2 font-medium opacity-0 group-hover/item:opacity-100 transition-opacity">{item.timestamp}</span>
                    <p className="text-gray-800 text-lg leading-relaxed font-medium tracking-wide">
                      {item.text}
                    </p>
                  </div>
                ))}
                {interimTranscript && (
                  <div className="flex gap-4 animate-pulse ml-[52px]">
                    <span className="text-gray-500 text-lg italic leading-relaxed">{interimTranscript}</span>
                  </div>
                )}
                {!liveTranscriptItems.length && !interimTranscript && (
                  <div className="h-full flex flex-col items-center justify-center text-center opacity-40">
                    <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mb-4 text-gray-300">
                      <MessageSquare className="w-8 h-8" />
                    </div>
                    <p className="text-gray-500 text-lg font-medium">Start speaking to see your words...</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: Sidebar Insights (4 cols) */}
          <div className="lg:col-span-4 flex flex-col gap-4 min-h-0 overflow-hidden">

            {/* AI Agent Status */}
            <div className="bg-gradient-to-br from-indigo-50 to-white p-5 rounded-2xl border border-indigo-100 flex items-start gap-4 shadow-sm">
              <div className="p-3 bg-white rounded-xl shadow-sm border border-indigo-50 text-indigo-600">
                <Sparkles className="w-6 h-6" />
              </div>
              <div>
                <h3 className="font-bold text-gray-900 text-sm mb-1">AI Coach Active</h3>
                <p className="text-xs text-gray-500 leading-relaxed">
                  Analyzing <strong>Eye Contact</strong>, <strong>Posture</strong>, and <strong>Speech Patterns</strong> in real-time.
                </p>
              </div>
            </div>

            {/* Live Observations Feed */}
            <div className="flex-1 bg-white rounded-2xl shadow-sm border border-gray-200 flex flex-col overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
                <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">Live Suggestions</span>
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              </div>
              <div className="flex-1 overflow-y-auto p-2">
                <LiveObservationFeed observations={liveObservations} />
              </div>
            </div>

            {/* Scenario Events */}
            {selectedScenario && (
              <div className="bg-orange-50 rounded-2xl border border-orange-100 overflow-hidden flex flex-col max-h-[250px]">
                <div className="px-5 py-3 border-b border-orange-100 bg-orange-50/80">
                  <span className="text-xs font-bold text-orange-600 uppercase tracking-wider flex items-center gap-2">
                    <Briefcase className="w-3 h-3" />
                    Scenario Events
                  </span>
                </div>
                <div className="bg-orange-100/50 px-5 py-3 border-b border-orange-100 text-sm text-orange-900 italic">
                  &quot;{selectedScenario.prompt}&quot;
                </div>
                <div className="overflow-y-auto p-3 space-y-2">
                  {scenarioEvents.length > 0 ? (
                    scenarioEvents.map(event => (
                      <div key={event.id} className="bg-white p-3 rounded-xl border border-orange-100 shadow-sm text-sm">
                        <p className="font-medium text-gray-800 mb-1">{event.description}</p>
                        {event.advice && <p className="text-xs text-gray-500 italic">{event.advice}</p>}
                      </div>
                    ))
                  ) : (
                    <div className="p-4 text-center text-gray-400 text-xs italic">
                      Waiting for simulation events...
                    </div>
                  )}
                </div>
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  )
}
