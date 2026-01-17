
import { v4 as uuidv4 } from 'uuid'

const BASE_URL = "https://api.on-demand.io/chat/v1"
const API_KEY = "pGdYFEt0VkaxknRL2H6F2ljfs7SsMRMw"

// Agent 9: Progress, Heatmap & Support Agent
const RESPONSE_MODE = "sync"
const AGENT_IDS = ["696a41eeac9b040cc2f7548c"] // Agent 9 ID
const ENDPOINT_ID = "predefined-xai-grok4.1-fast"
const REASONING_MODE = "grok-4-fast"

const FULFILLMENT_PROMPT = `IMPORTANT: PARSE input JSON. Do NOT echo.

Parse this JSON:
<PROGRESS_DATA>
{{input}}
</PROGRESS_DATA>

You are Agent 9: Progress, Heatmap & Support Agent. Analyze historical sessions for trends/resources.

INPUT FORMAT (array of sessions):
[
  {"session_id": 1, "confidence_score": 68, "pace": 0.55, "fillers": 6, "eye_contact": 0.64, "date": "2026-01-16"},
  ...
]

PROCESSING:
1. Trends: Overall confidence ↑/↓, strongest/weakest metric.
2. Heatmap (text): e.g., "Pace: 🔥🔥🟡 (poor) | Eye: 🟡🟡🔥 (improving)".
3. Resources: 2-3 targeted links/videos (YouTube/TED/Toastmasters).
4. Support: If avg confidence <60, suggest "local Toastmasters".

OUTPUT JSON:
{
  "overall_trend": "+7% confidence (improving)",
  "heatmap": {
    "pace": "🔴🔴🟡 (weakest)",
    "fillers": "🟡🔥🔥",
    "eye_contact": "🔥🔥🟡",
    "avg_confidence": 71.5
  },
  "resources": [
    "https://youtu.be/example-pace-training",
    "TED Talk: Body Language Mastery"
  ],
  "support_tip": "Join Toastmasters for live practice"
}

Use 5 sessions max. Realistic links. Ethical/non-medical.`

export interface Agent9Input {
    sessions: Array<{
        session_id: string
        confidence_score: number
        pace: number
        fillers: number
        eye_contact: number
        date: string
    }>
}

export async function processAgent9(input: Agent9Input) {
    const sessionId = await createChatSession()
    // Take only last 5 sessions to match prompt constraint
    const recentSessions = input.sessions.slice(-5)
    const inputString = JSON.stringify(recentSessions)

    return await submitQuery(sessionId, inputString)
}

async function createChatSession(): Promise<string> {
    const url = `${BASE_URL}/sessions`
    const body = {
        agentIds: AGENT_IDS,
        externalUserId: uuidv4(),
        contextMetadata: [
            { key: "type", value: "progress-analysis" }
        ]
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Failed to create session Agent 9: ${response.status} ${await response.text()}`)
    const data = await response.json()
    return data.data.id
}

async function submitQuery(sessionId: string, queryInput: string) {
    const url = `${BASE_URL}/sessions/${sessionId}/query`

    const finalPrompt = FULFILLMENT_PROMPT.replace('{{input}}', queryInput)

    const body = {
        endpointId: ENDPOINT_ID,
        query: "analyze_progress",
        agentIds: AGENT_IDS,
        responseMode: RESPONSE_MODE,
        reasoningMode: REASONING_MODE,
        modelConfigs: {
            fulfillmentPrompt: finalPrompt,
            temperature: 0.3,
            topP: 0.9,
            maxTokens: 400,
            presencePenalty: 0,
            frequencyPenalty: 0.1
        },
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Query failed Agent 9: ${await response.text()}`)
    return await response.json()
}
