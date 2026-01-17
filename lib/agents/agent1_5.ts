import { v4 as uuidv4 } from 'uuid'

const BASE_URL = "https://api.on-demand.io/chat/v1"
const MEDIA_BASE_URL = "https://api.on-demand.io/media/v1"
const API_KEY = "pGdYFEt0VkaxknRL2H6F2ljfs7SsMRMw" // Using the key from agent5.ts

const AGENT_IDS = ["696aaa02c7d6dfdf7e337e18"] // Agent 1.5 ID
const QUERY = "Store this audio and return URL"
const RESPONSE_MODE = "sync"
const ENDPOINT_ID = "predefined-xai-grok4.1-fast"
const REASONING_MODE = "grok-4-fast"

const FULFILLMENT_PROMPT = `You are AGENT 1.5 — Audio Storage & URL Manager.

Your role is to:
1. Receive base64-encoded audio from Agent 1.
2. Store the audio using the storage tool.
3. Return a temporary audio_url and session_id.

Do not perform transcription or analysis.
Output JSON only:
{
  "audio_url": "...",
  "session_id": "..."
}`

const FILE_AGENTS = ["agent-1713954536", "agent-1713958591", "agent-1713958830", "agent-1713961903", "agent-1713967141"]

export interface Agent1_5Input {
    file: Blob
    fileName: string
}

export async function processAgent1_5(input: Agent1_5Input) {
    const sessionId = await createChatSession()

    // Upload the file
    if (input.file) {
        const uploadResult = await uploadMediaFile(input.file, input.fileName, FILE_AGENTS, sessionId)
        if (uploadResult) {
            // Return the upload result directly as it contains the URL usually, 
            // or we can query the agent to confirm. 
            // The prompt says "Return a temporary audio_url".
            // Let's submit the query to let the agent finalize the "Storage" task representation
            // or just return the upload metadata if that's what the User wants.
            // The user script did `submitQuery` after upload.
            return {
                audio_url: uploadResult.url,
                file_id: uploadResult.id,
                session_id: sessionId
            }
        }
    }

    return null
}

async function createChatSession(): Promise<string> {
    const url = `${BASE_URL}/sessions`
    const body = {
        agentIds: AGENT_IDS,
        externalUserId: uuidv4(),
        contextMetadata: [
            { key: "type", value: "audio-storage" }
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

    if (!response.ok) throw new Error(`Failed to create session Agent 1.5: ${response.status}`)
    const data = await response.json()
    return data.data.id
}

async function uploadMediaFile(file: Blob, fileName: string, agents: string[], sessionId: string) {
    const url = `${MEDIA_BASE_URL}/public/file/raw`

    // Convert Blob to File-like object if needed, or append directly
    const formData = new FormData()
    formData.append('file', file)
    formData.append('sessionId', sessionId)
    formData.append('createdBy', 'AIREV')
    formData.append('updatedBy', 'AIREV')
    formData.append('name', fileName)
    formData.append('responseMode', RESPONSE_MODE)

    agents.forEach(agent => {
        formData.append('agents', agent)
    })

    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'apikey': API_KEY
            // FormData headers are handled automatically by fetch in browser/node environment usually
            // but in some node environments we might need headers from formData
        },
        body: formData
    })

    if (!response.ok) {
        console.error(`Agent 1.5 Upload failed: ${await response.text()}`)
        return null
    }

    const mediaResponse = await response.json()
    return mediaResponse.data
}
