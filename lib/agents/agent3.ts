import { v4 as uuidv4 } from 'uuid'

const BASE_URL = "https://api.on-demand.io/chat/v1"
const API_KEY = "pGdYFEt0VkaxknRL2H6F2ljfs7SsMRMw" // Using the key from agent5.ts

const AGENT_IDS = ["696a9ed4ac9b040cc2f75590"] // Agent 3 ID
const ENDPOINT_ID = "predefined-xai-grok4.1-fast"
const REASONING_MODE = "grok-4-fast"
const RESPONSE_MODE = "sync"

const FULFILLMENT_PROMPT = `You are AGENT 3 — NLP & Filler Word Intelligence Agent.

Input:
• Transcribed text (mandatory if available)
• Timing info (optional)

Steps:
1. If transcript is missing, use the Whisper tool to generate it.
2. Use the Chat API tool to analyze the transcript for:
   - filler words
   - clarity score
   - sentence completeness
   - phrase repetition
   - vocabulary richness
When generating the output:
• clarity_score MUST be a decimal value between 0 and 1 (example: 0.78)
• Do NOT use a 1–10 or 1–100 scale
3. Output ONLY valid JSON:

{
  "filler_words": number,
  "clarity_score": number,
  "sentence_completeness": number, 
  "vocabulary_richness": number,
  "key_phrases_repeated": []
}

IMPORTANT: PARSE input JSON. Do NOT echo it.
PARSE this JSON input:
<AGENT_DATA>
{{input}}
</AGENT_DATA>`

export interface Agent3Input {
    transcript?: string
    audioUrl?: string
}

export async function processAgent3(input: Agent3Input) {
    const sessionId = await createChatSession()

    // Prepare input JSON
    const inputString = JSON.stringify(input)

    // Inject input into prompt
    const finalPrompt = FULFILLMENT_PROMPT.replace('{{input}}', inputString)

    return await submitQuery(sessionId, finalPrompt)
}

async function createChatSession(): Promise<string> {
    const url = `${BASE_URL}/sessions`
    const body = {
        agentIds: AGENT_IDS,
        externalUserId: uuidv4(),
        contextMetadata: [
            { key: "type", value: "nlp-analysis" }
        ]
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Failed to create session Agent 3: ${response.status}`)
    const data = await response.json()
    return data.data.id
}

async function submitQuery(sessionId: string, prompt: string) {
    const url = `${BASE_URL}/sessions/${sessionId}/query`

    const body = {
        endpointId: ENDPOINT_ID,
        query: "analyze_transcript", // Static trigger
        agentIds: AGENT_IDS,
        responseMode: RESPONSE_MODE,
        reasoningMode: REASONING_MODE,
        modelConfigs: {
            fulfillmentPrompt: prompt,
            temperature: 0.2,
            topP: 0.9,
            maxTokens: 800,
            frequencyPenalty: 0.2
        },
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Query failed Agent 3: ${await response.text()}`)
    return await response.json()
}
