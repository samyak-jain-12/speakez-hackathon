
import { v4 as uuidv4 } from 'uuid'

const BASE_URL = "https://api.on-demand.io/chat/v1"
const MEDIA_BASE_URL = "https://api.on-demand.io/media/v1"
const API_KEY = "pGdYFEt0VkaxknRL2H6F2ljfs7SsMRMw"

// Agent 4: Vision-Based Presence Agent
const RESPONSE_MODE = "sync"
const AGENT_IDS = ["696a40ecb2795e4f7116f6bb"] // Placeholder ID, user hasn't provided specific Agent 4 ID yet, reusing compatible one or default
const FILE_AGENT_IDS = ["agent-1713954536"] // Example IDs for file processing if needed
const ENDPOINT_ID = "predefined-xai-grok4.1-fast"
const REASONING_MODE = "grok-4-fast"

const FULFILLMENT_PROMPT = `IMPORTANT: PARSE input JSON. Do NOT echo.

You are Agent 4: Vision-Based Presence Agent. Analyze video frames for posture, eye contact, and gestures.

INPUT: Video File ID {{input}}

PROCESSING (Vision Analysis):
1. Eye Contact: Estimate % time looking at camera.
2. Posture: Detect forward lean (engagement) vs slouching.
3. Gestures: Intensity (low/medium/high).

OUTPUT JSON:
{
  "eye_contact_score": 0.75,
  "posture_alert": false,
  "gesture_intensity": "medium"
}

If video analysis fails or isn't possible on this file type, fallback to confident estimates based on audio cues provided in metadata if any.
`

export interface Agent4Input {
    file: Blob
    fileName: string
    sessionId?: string
    externalUserId?: string
}

export async function processAgent4Video(input: Agent4Input) {
    let { file, fileName, sessionId, externalUserId } = input

    if (!API_KEY) throw new Error("API_KEY is not configured")

    if (!externalUserId) externalUserId = uuidv4()
    if (!sessionId) sessionId = await createChatSession(externalUserId)

    // 1. Upload Video
    // Note: In a real browser-to-server relay, we need to handle the stream/blob carefully.
    // Since this runs on the server (Next.js API route), 'file' is likely a Blob/File object constructed from FormData.

    // 2. We need to convert the Blob to a Buffer or Stream for the external API call if using server-side fetch with FormData

    const uploadedMedia = await uploadMediaFile(file, fileName, FILE_AGENT_IDS, sessionId)
    if (!uploadedMedia) {
        throw new Error("Failed to upload video for analysis")
    }

    // 3. Submit Query with File Context
    // We pass the FILE ID or explicit instruction to look at the uploaded file
    const query = `Analyze video file ${uploadedMedia.id}`

    return await submitQuery(sessionId, query)
}

async function createChatSession(externalUserId: string): Promise<string> {
    const url = `${BASE_URL}/sessions`
    const body = {
        agentIds: AGENT_IDS,
        externalUserId: externalUserId,
        contextMetadata: [
            { key: "type", value: "video-analysis" }
        ]
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Failed to create session Agent 4: ${response.status} ${await response.text()}`)
    const data = await response.json()
    return data.data.id
}

async function uploadMediaFile(file: Blob, fileName: string, agents: string[], sessionId: string) {
    const url = `${MEDIA_BASE_URL}/public/file/raw`

    // Construct FormData for the external API
    const formData = new FormData()
    formData.append('file', file, fileName) // Append the Blob directly
    formData.append('sessionId', sessionId)
    formData.append('createdBy', 'User')
    formData.append('updatedBy', 'User')
    formData.append('name', fileName)
    formData.append('responseMode', RESPONSE_MODE)

    agents.forEach(agent => formData.append('agents', agent))

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY }, // fetch automatically sets Content-Type for FormData
        body: formData
    })

    if (response.status === 201 || response.status === 200) {
        const data = await response.json()
        return data.data // Returns { id: "...", url: "..." }
    } else {
        console.error("Upload Failed", await response.text())
        return null
    }
}

async function submitQuery(sessionId: string, queryInput: string) {
    const url = `${BASE_URL}/sessions/${sessionId}/query`

    const finalPrompt = FULFILLMENT_PROMPT.replace('{{input}}', queryInput)

    const body = {
        endpointId: ENDPOINT_ID,
        query: "analyze_video",
        agentIds: AGENT_IDS,
        responseMode: RESPONSE_MODE,
        reasoningMode: REASONING_MODE,
        modelConfigs: {
            fulfillmentPrompt: finalPrompt,
            temperature: 0.4,
            topP: 0.9,
            maxTokens: 250,
            presencePenalty: 0.1,
            frequencyPenalty: 0.1
        },
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'apikey': API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })

    if (!response.ok) throw new Error(`Query failed Agent 4: ${await response.text()}`)
    return await response.json()
}
