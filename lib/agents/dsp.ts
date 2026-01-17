import { v4 as uuidv4 } from 'uuid'
// @ts-ignore
import fft from 'fft-js'

const BASE_URL = "https://api.on-demand.io/chat/v1"
const API_KEY = "pGdYFEt0VkaxknRL2H6F2ljfs7SsMRMw"
const AGENT_IDS = ["696a1b83ac9b040cc2f753e3"] // Agent 2 ID
const ENDPOINT_ID = "predefined-xai-grok4.1-fast"

const FULFILLMENT_PROMPT = `You are a speech feature orchestration agent.

You MUST extract speech features by calling the DSPFeatureService tool.

Instructions:
1. When an audio frame is received, ALWAYS call the DSPFeatureService tool.
2. Pass the audio_frame and sample_rate exactly as received.
3. Take the JSON response returned by the tool.
4. Return that JSON as your final output with NO modifications.
5. Do NOT generate values yourself.
6. Do NOT return placeholder values.
7. Do NOT explain anything.

Your final response must be ONLY the JSON returned by the tool.
`

export interface DSPResult {
    pitch: number
    volume: number
    speaking_pace: number
    stress_index: number
    repetition_count: number
}

// --- LOCAL HEURISTICS (For Live Feedback Speed) ---

function decodeAudio(base64Audio: string): Float32Array {
    try {
        const binaryString = atob(base64Audio)
        const len = binaryString.length
        const adjustedLen = len % 2 === 0 ? len : len - 1
        const buffer = new Int16Array(adjustedLen / 2)

        for (let i = 0; i < adjustedLen; i += 2) {
            const low = binaryString.charCodeAt(i)
            const high = binaryString.charCodeAt(i + 1)
            const sample = (high << 8) | low
            const int16Sample = sample >= 32768 ? sample - 65536 : sample
            buffer[i / 2] = int16Sample
        }

        const float32 = new Float32Array(buffer.length)
        for (let i = 0; i < buffer.length; i++) {
            float32[i] = buffer[i] / 32768.0
        }

        return float32
    } catch (e) {
        console.error("DSP Decode Error", e)
        return new Float32Array(512)
    }
}

function computeRMS(audio: Float32Array): number {
    if (audio.length === 0) return 0
    let sumSquares = 0
    for (let i = 0; i < audio.length; i++) {
        sumSquares += audio[i] * audio[i]
    }
    return Math.sqrt(sumSquares / audio.length)
}

// ... (skipping computePitch which was fine)
function computePitch(audio: Float32Array, sampleRate: number): number {
    try {
        let n = 1
        while (n < audio.length) n *= 2

        const padded = new Array(n).fill(0)
        for (let i = 0; i < audio.length; i++) padded[i] = audio[i]

        const phasors = fft.fft(padded)
        const frequencies = fft.util.fftFreq(phasors, sampleRate)
        const magnitudes = fft.util.fftMag(phasors)

        let maxMag = 0
        let peakFreq = 0
        for (let i = 0; i < frequencies.length; i++) {
            const freq = frequencies[i]
            if (freq > 85 && freq < 400) {
                if (magnitudes[i] > maxMag) {
                    maxMag = magnitudes[i]
                    peakFreq = freq
                }
            }
        }
        return peakFreq
    } catch (e) {
        return 0
    }
}

export function extractDSPFeatures(input: string | number[], sampleRate = 16000): DSPResult {
    let audio: Float32Array

    if (Array.isArray(input)) {
        audio = new Float32Array(input)
    } else {
        audio = decodeAudio(input)
    }

    const rms = computeRMS(audio)
    const volume = Math.min(rms / 0.1, 1.0)
    const pitch = computePitch(audio, sampleRate)

    let activeSamples = 0
    for (let i = 0; i < audio.length; i++) {
        if (Math.abs(audio[i]) > 0.05) activeSamples++
    }
    const pace = Math.round((activeSamples / Math.max(audio.length, 1)) * 160 * 60)

    let mean = 0
    for (let i = 0; i < audio.length; i++) {
        mean += audio[i]
    }
    mean /= audio.length

    let variance = 0
    for (let i = 0; i < audio.length; i++) {
        variance += (audio[i] - mean) ** 2
    }
    variance /= audio.length

    const stress = Math.min(
        0.5 * variance +
        0.5 * Math.abs(pitch - 150.0) / 150.0,
        1.0
    )

    return {
        pitch: Math.round(pitch * 100) / 100,
        volume: Math.round(volume * 1000) / 1000,
        speaking_pace: pace,
        stress_index: Math.round(stress * 1000) / 1000,
        repetition_count: 0
    }
}

// --- CLOUD AGENT 2 (For Final Analysis Connection) ---

export async function processAgent2(audioData: number[], sampleRate: number) {
    const sessionId = await createChatSession()

    const inputPayload = JSON.stringify({
        audio_frame: audioData,
        sample_rate: sampleRate
    })

    return await submitQuery(sessionId, inputPayload)
}

async function createChatSession(): Promise<string> {
    const url = `${BASE_URL}/sessions`
    const body = {
        agentIds: AGENT_IDS,
        externalUserId: uuidv4(),
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Agent 2 Session Failed: ${await response.text()}`)
    const data = await response.json()
    return data.data.id
}

async function submitQuery(sessionId: string, queryInput: string) {
    const url = `${BASE_URL}/sessions/${sessionId}/query`
    const body = {
        endpointId: ENDPOINT_ID,
        query: queryInput,
        agentIds: AGENT_IDS,
        responseMode: "sync",
        reasoningMode: "grok-4-fast",
        modelConfigs: {
            fulfillmentPrompt: FULFILLMENT_PROMPT,
            temperature: 0,
            topP: 1,
        },
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Agent 2 Query Failed: ${await response.text()}`)

    const result = await response.json()

    try {
        if (result.data && result.data.answer) {
            const cleanAnswer = result.data.answer.replace(/```json/g, '').replace(/```/g, '').trim()
            return JSON.parse(cleanAnswer)
        }
    } catch (e) {
        console.warn("Could not parse Agent 2 answer as JSON, returning raw:", e)
    }

    return result.data
}
