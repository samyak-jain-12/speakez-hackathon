
import { v4 as uuidv4 } from 'uuid'

const BASE_URL = "https://api.on-demand.io/chat/v1"
const API_KEY = "pGdYFEt0VkaxknRL2H6F2ljfs7SsMRMw"

// Agent 8: Drill Generator Agent
const RESPONSE_MODE = "sync"
const AGENT_IDS = ["696a40ecb2795e4f7116f6bb"] // Agent 8 ID
const ENDPOINT_ID = "predefined-xai-grok4.1-fast"
const REASONING_MODE = "grok-4-fast"

const FULFILLMENT_PROMPT = `Call drill_generator.generateDrillTask with {{input}}

Expected output: {"task": "...", "feedback": "..."}
`

export interface Agent8Input {
    session_summary: string
    weak_points: string[]
}

export async function processAgent8(input: Agent8Input) {
    const sessionId = await createChatSession()
    const inputString = JSON.stringify(input)
    return await submitQuery(sessionId, inputString)
}

async function createChatSession(): Promise<string> {
    const url = `${BASE_URL}/sessions`
    const body = {
        agentIds: AGENT_IDS,
        externalUserId: uuidv4(),
        contextMetadata: [
            { key: "type", value: "drill-generation" }
        ]
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Failed to create session Agent 8: ${response.status} ${await response.text()}`)
    const data = await response.json()
    return data.data.id
}

async function submitQuery(sessionId: string, queryInput: string) {
    const url = `${BASE_URL}/sessions/${sessionId}/query`

    // Replace {{input}} in the prompt
    // Note: The prompt asks to "Call drill_generator...", implying function calling, 
    // but here we are just sending text to Grok. 
    // We'll rely on Grok to interpret "Call..." as "Generate a JSON with..."
    const finalPrompt = FULFILLMENT_PROMPT.replace('{{input}}', queryInput)

    const body = {
        endpointId: ENDPOINT_ID,
        query: "generate_drill", // Abstract query
        agentIds: AGENT_IDS,
        responseMode: RESPONSE_MODE,
        reasoningMode: REASONING_MODE,
        modelConfigs: {
            fulfillmentPrompt: finalPrompt,
            temperature: 0, // Explicitly 0 as per user snippet
            topP: 0.9,
            maxTokens: 200,
            presencePenalty: 0.1,
            frequencyPenalty: 0.2
        },
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Query failed Agent 8: ${await response.text()}`)
    return await response.json()
}
