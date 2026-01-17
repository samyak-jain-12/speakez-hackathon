import { v4 as uuidv4 } from 'uuid'

const BASE_URL = "https://api.on-demand.io/chat/v1"
const API_KEY = "pGdYFEt0VkaxknRL2H6F2ljfs7SsMRMw"

// Agent 6: Purpose & Emotion Alignment Agent
const RESPONSE_MODE = "sync"
const AGENT_IDS = ["696a3c5aac9b040cc2f7546d"] // Agent 6 ID
const ENDPOINT_ID = "predefined-xai-grok4.1-fast"
const REASONING_MODE = "grok-4-fast"

const FULFILLMENT_PROMPT = `You are Agent 6: Purpose & Emotion Alignment Agent. Analyze if speech style/emotion matches selected context using NLP/confidence data.

INPUT FORMAT (JSON from Agents 2-5 + mode):
<DATA>
{{input}}
</DATA>

CONTEXT RULES:
• Interview: Low pitch variance, slow pace (<150), low fillers/stress, high eye contact, professional phrases.
• Debate: Higher pace/energy, assertive tone, fewer hesitations.
• Casual: Relaxed pitch/pace, natural fillers OK.
• Presentation: Steady pace, confident posture/voice.

PROCESSING:
1. Score tone_match: true/false (match >80% alignment).
2. Detect mismatches (e.g., high stress in Interview → hesitant).
3. Generate 1 actionable suggestion (10-20 words).

OUTPUT ONLY VALID JSON:
{
  "tone_match": false,
  "suggestion": "Reduce pace and fillers for professional interview tone; maintain eye contact."
}

Be precise: Use semantic analysis on repeats/fillers, map pitch/pace/stress to emotions. No medical claims.`

export interface Agent6Input {
    agent2: any
    agent3: any
    agent4: any
    agent5: any
    mode: string
}

export async function processAgent6(input: Agent6Input) {
    const sessionId = await createChatSession() // Can reuse session if needed, but let's create fresh

    // Prepare input JSON
    const inputString = JSON.stringify(input)

    return await submitQuery(sessionId, inputString)
}

async function createChatSession(): Promise<string> {
    const url = `${BASE_URL}/sessions`
    const body = {
        agentIds: AGENT_IDS,
        externalUserId: uuidv4(),
        contextMetadata: [
            { key: "type", value: "alignment-analysis" }
        ]
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Failed to create session Agent 6: ${response.status} ${await response.text()}`)
    const data = await response.json()
    return data.data.id
}

async function submitQuery(sessionId: string, queryInput: string) {
    const url = `${BASE_URL}/sessions/${sessionId}/query`

    const finalPrompt = FULFILLMENT_PROMPT.replace('{{input}}', queryInput)

    const body = {
        endpointId: ENDPOINT_ID,
        query: "analyze_alignment",
        agentIds: AGENT_IDS,
        responseMode: RESPONSE_MODE,
        reasoningMode: REASONING_MODE,
        modelConfigs: {
            fulfillmentPrompt: finalPrompt,
            temperature: 0.3,
            topP: 0.9,
            maxTokens: 300,
            presencePenalty: 0.1,
            frequencyPenalty: 0.2
        },
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Query failed Agent 6: ${await response.text()}`)
    return await response.json()
}
