import { v4 as uuidv4 } from 'uuid'

const BASE_URL = "https://api.on-demand.io/chat/v1"
const MEDIA_BASE_URL = "https://api.on-demand.io/media/v1"
const API_KEY = process.env.ON_DEMAND_API_KEY || "<your_api_key>"

// Dynamic Configuration
// Dynamic Configuration
const RESPONSE_MODE = "sync"
const AGENT_IDS = ["agent-1712327325", "agent-1713962163", "agent-1768559587", "agent-1768562436", "agent-1768570503"]
const ENDPOINT_ID = "predefined-xai-grok4.1-fast"
const REASONING_MODE = "grok-4-fast"
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

Your final response must be ONLY the JSON returned by the tool.`
const STOP_SEQUENCES: string[] = []
const FILE_AGENTS = ["agent-1713954536", "agent-1713958591", "agent-1713958830", "agent-1713961903", "agent-1713967141"]

export interface AgentInput {
    query: string
    file?: Blob
    fileName?: string
    sessionId?: string
    externalUserId?: string
}

export async function processAgentInput(input: AgentInput) {
    let { query, file, fileName, sessionId, externalUserId } = input

    if (!API_KEY || API_KEY === "<your_api_key>") {
        throw new Error("API_KEY is not configured")
    }

    if (!externalUserId) {
        externalUserId = uuidv4()
    }

    if (!sessionId) {
        sessionId = await createChatSession(externalUserId)
    }

    // Upload Media if present
    if (file && fileName) {
        await uploadMediaFile(file, fileName, FILE_AGENTS, sessionId)
    }

    // Submit Query
    return await submitQuery(sessionId, query)
}

async function createChatSession(externalUserId: string): Promise<string> {
    const url = `${BASE_URL}/sessions`

    const body = {
        agentIds: AGENT_IDS,
        externalUserId: externalUserId,
        contextMetadata: [
            { key: "userId", value: "1" },
            { key: "name", value: "User" }
        ]
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'apikey': API_KEY,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    })

    if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Failed to create session: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    return data.data.id
}

async function uploadMediaFile(file: Blob, fileName: string, agents: string[], sessionId: string) {
    const url = `${MEDIA_BASE_URL}/public/file/raw`

    const formData = new FormData()
    formData.append('file', file)
    formData.append('sessionId', sessionId)
    formData.append('createdBy', 'AIREV')
    formData.append('updatedBy', 'AIREV')
    formData.append('name', fileName)
    formData.append('responseMode', RESPONSE_MODE)

    agents.forEach(agent => formData.append('agents', agent))

    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'apikey': API_KEY,
        },
        body: formData
    })

    if (!response.ok) {
        const errorText = await response.text()
        console.error(`Upload failed: ${response.status} - ${errorText}`)
        // We continue even if upload fails, but log it
        return null
    }

    return await response.json()
}

async function submitQuery(sessionId: string, query: string) {
    const url = `${BASE_URL}/sessions/${sessionId}/query`

    const body = {
        endpointId: ENDPOINT_ID,
        query: query,
        agentIds: AGENT_IDS,
        responseMode: RESPONSE_MODE,
        reasoningMode: REASONING_MODE,
        modelConfigs: {
            fulfillmentPrompt: FULFILLMENT_PROMPT,
            stopSequences: STOP_SEQUENCES,
            temperature: 0,
            topP: 1,
            maxTokens: 0,
            presencePenalty: 0,
            frequencyPenalty: 0,
        },
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'apikey': API_KEY,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    })

    if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Query failed: ${response.status} - ${errorText}`)
    }

    return await response.json()
}
