import { v4 as uuidv4 } from 'uuid'
import { AgentResult } from './orchestrator'

const BASE_URL = "https://api.on-demand.io/chat/v1"
const API_KEY = "pGdYFEt0VkaxknRL2H6F2ljfs7SsMRMw"

// Agent 5: Confidence Signal & Score Engine
const QUERY = "calculate_confidence"
const RESPONSE_MODE = "sync"
const AGENT_IDS = ["696a3892c7d6dfdf7e337ca2"] // Agent 5 ID
const ENDPOINT_ID = "predefined-xai-grok4.1-fast"
const REASONING_MODE = "grok-4-fast"

const FULFILLMENT_PROMPT = `IMPORTANT: PARSE input JSON. Do NOT echo it.

Parse this JSON:
<AGENT_DATA>
{{input}}
</AGENT_DATA>

You are Agent 5: Confidence Signal & Score Engine. Your job is to synthesize speech metrics (Agent 2), fluency (Agent 3), and body language (Agent 4) into a cohesive confidence profile.

INPUT FORMAT (from Agents 2, 3, 4):
{
  "agent2": {"pitch": 210, "volume": 0.62, "pace": 145, "stress_index": 0.71, "repetition_count": 3},
  "agent3": {"filler_words": 6, "clarity_score": 0.78, "key_phrases_repeated": ["I think", "basically"]},
  "agent4": {"eye_contact_score": 0.64, "posture_alert": true, "gesture_intensity": "low"}
}

TOOL CALL INSTRUCTION:
1. Analyze the input data to determine confidence levels.
2. Call tool: confidence_calculator (conceptually, or just output the JSON directly if no tool is available, but here we simulate the output).
   
OUTPUT JSON FORMAT (Strictly match this):
{
  "confidence_score": 0-100,
  "dominant_weakness": "one of (Pace, Volume, Clarity, Eye Contact, Stress, Posture, Vocabulary, Completeness)",
  "factors": [
    { "name": "Pace Control", "impact": 0-100, "score": 0-100 },
    { "name": "Volume Consistency", "impact": 0-100, "score": 0-100 },
    { "name": "Eye Contact", "impact": 0-100, "score": 0-100 },
    { "name": "Stress Management", "impact": 0-100, "score": 0-100 },
    { "name": "Clarity & Fillers", "impact": 0-100, "score": 0-100 },
    { "name": "Sentence Completeness", "impact": 0-100, "score": 0-100 },
    { "name": "Vocabulary Richness", "impact": 0-100, "score": 0-100 }
  ],
  "key_observations": [
    "Observation 1",
    "Observation 2",
    "Observation 3"
  ],
  "summary_text": "A short, encouraging summary."
}

CRITICAL RULES:
1. Do NOT copy the example output.
2. CALCULATE scores based on the input data 'agent2', 'agent3', 'agent4'.
3. If valid input is missing, use sensible defaults but do NOT return the specific example text below.
4. 'impact' should sum to roughly 100 across all factors (it represents weight/importance).
5. 'score' is your evaluation of that factor (0-100).

Example Output (DO NOT COPY THIS):
{
  "confidence_score": 50,
  "dominant_weakness": "Clarity & Fillers",
  "factors": [
    { "name": "Pace Control", "impact": 10, "score": 50 },
    { "name": "Volume Consistency", "impact": 10, "score": 50 },
    { "name": "Eye Contact", "impact": 10, "score": 50 },
    { "name": "Stress Management", "impact": 10, "score": 50 },
    { "name": "Clarity & Fillers", "impact": 30, "score": 40 },
    { "name": "Sentence Completeness", "impact": 15, "score": 60 },
    { "name": "Vocabulary Richness", "impact": 15, "score": 55 }
  ],
  "key_observations": [
    "You used 12 filler words which impacted clarity.",
    "Sentence structure was fragmented.",
    "Volume was low."
  ],
  "summary_text": "This is a generic example summary. Do not output this text."
}`

export interface Agent5Input {
    agent2: any // Speech Signal
    agent3: any // Fluency
    agent4: any // Body Language (mocked if deleted)
}

export async function processAgent5(input: Agent5Input) {
    const sessionId = await createChatSession()

    // Prepare input JSON for the prompt
    const inputString = JSON.stringify(input)
    const query = inputString // Pass data as the query or wrapped in format

    // Actually, prompt says "QUERY = <your_query>". We should likely send the data as the query
    // or inject it into the prompt if the API allows variables. 
    // The prompt has `{{input}}`. 
    // Let's assume we pass the input JSON as the query string so the model sees it.

    return await submitQuery(sessionId, inputString)
}

async function createChatSession(): Promise<string> {
    const url = `${BASE_URL}/sessions`
    const body = {
        agentIds: AGENT_IDS,
        externalUserId: uuidv4(),
        contextMetadata: [
            { key: "type", value: "confidence-analysis" }
        ]
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Failed to create session Agent 5: ${response.status} ${await response.text()}`)
    const data = await response.json()
    return data.data.id
}

async function submitQuery(sessionId: string, queryInput: string) {
    const url = `${BASE_URL}/sessions/${sessionId}/query`

    // Construct the prompt by replacing {{input}} manually if needed, or rely on model to see the query.
    // The user's code snippet used `QUERY` as a static string and `FULFILLMENT_PROMPT` had `{{input}}`.
    // We'll replace {{input}} in the prompt with our JSON data and send a generic query or the data itself.
    // Let's inject into prompt for robustness.

    const finalPrompt = FULFILLMENT_PROMPT.replace('{{input}}', queryInput)

    const body = {
        endpointId: ENDPOINT_ID,
        query: "analyze_confidence", // Static query trigger
        agentIds: AGENT_IDS,
        responseMode: RESPONSE_MODE,
        reasoningMode: REASONING_MODE,
        modelConfigs: {
            fulfillmentPrompt: finalPrompt,
            temperature: 0.1,
            topP: 0.9,
            maxTokens: 500
        },
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Query failed Agent 5: ${await response.text()}`)
    return await response.json()
}
