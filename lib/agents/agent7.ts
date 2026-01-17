
import { v4 as uuidv4 } from 'uuid'

const BASE_URL = "https://api.on-demand.io/chat/v1"
const API_KEY = "pGdYFEt0VkaxknRL2H6F2ljfs7SsMRMw" // Using consistent API Key from other agents

// Agent 7: Scenario Simulation Agent
const RESPONSE_MODE = "sync"
const AGENT_IDS = ["696a3f93ac9b040cc2f7547c"] // Agent 7 ID provided by user
const ENDPOINT_ID = "predefined-xai-grok4.1-fast"
const REASONING_MODE = "grok-4-fast"

const FULFILLMENT_PROMPT = `IMPORTANT: PARSE input JSON. Do NOT echo it.

Parse this JSON:
<SCENARIO_DATA>
{{input}}
</SCENARIO_DATA>

You are Agent 7: Scenario Simulation Agent. Generate real-world pressure events for public speaking practice.

INPUT FORMAT:
{
  "scenario": "HR Interview",  // "Client Pitch", "Team Meeting", "Debate", "Sales Call"
  "current_speech": "User's recent transcribed text...",
  "live_stats": {"confidence_score": 68, "pace": 145},  // Optional from Agent 5
  "timestamp": 12345
}

SCENARIOS & EVENTS:
• HR Interview: "Sorry, we have another candidate...", time pressure.
• Client Pitch: Technical question, interruption.
• Team Meeting: Colleague disagrees.
• Debate: Counterargument.
• Sales Call: Objection handling.

PROCESSING:
1. Select fitting event based on scenario/progress (e.g., low confidence → tougher interrupt).
2. Generate scenario_event (string: "The interviewer interrupts: 'Sorry, we have another candidate who has more experience...'").
3. expected_response_type: "calm_defensive", "assertive_clarify", etc.

OUTPUT ONLY JSON:
{
  "scenario_event": "HR interrupts: 'Why should we hire you over others?'",
  "expected_response_type": "calm_assertive"
}

Make realistic, progressive (escalate pressure). Use Chat API for natural language.`

export interface Agent7Input {
    scenario: string
    current_speech: string
    live_stats?: {
        confidence_score?: number
        pace?: number
    }
    timestamp: number
}

export async function processAgent7(input: Agent7Input) {
    const sessionId = await createChatSession()

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
            { key: "type", value: "scenario-simulation" }
        ]
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Failed to create session Agent 7: ${response.status} ${await response.text()}`)
    const data = await response.json()
    return data.data.id
}

async function submitQuery(sessionId: string, queryInput: string) {
    const url = `${BASE_URL}/sessions/${sessionId}/query`

    const finalPrompt = FULFILLMENT_PROMPT.replace('{{input}}', queryInput)

    const body = {
        endpointId: ENDPOINT_ID,
        query: "simulate_event",
        agentIds: AGENT_IDS,
        responseMode: RESPONSE_MODE,
        reasoningMode: REASONING_MODE,
        modelConfigs: {
            fulfillmentPrompt: finalPrompt,
            temperature: 0.6,
            topP: 0.95,
            maxTokens: 250,
            presencePenalty: 0.2,
            frequencyPenalty: 0.3
        },
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Query failed Agent 7: ${await response.text()}`)
    const result = await response.json()

    // Parse wrapped JSON if needed
    try {
        if (result.data && result.data.answer) {
            const cleanAnswer = result.data.answer.replace(/```json/g, '').replace(/```/g, '').trim()
            return { ...result, data: { ...result.data, answer: cleanAnswer } }
        }
    } catch (e) {
        console.warn("Failed to clean Agent 7 JSON", e)
    }

    return result
}
