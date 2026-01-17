import { NextRequest, NextResponse } from 'next/server'
import { processAgent4Video } from '@/lib/agents/agent4'

const fs = require('fs');
const path = require('path');

function logDebug(message: string) {
    const logPath = path.join(process.cwd(), 'debug_agent4.log');
    fs.appendFileSync(logPath, `[${new Date().toISOString()}] ${message}\n`);
}

export async function POST(req: NextRequest) {
    try {
        logDebug("Received POST request to /api/agent/video");
        const formData = await req.formData()
        const file = formData.get('file') as Blob
        const fileName = formData.get('fileName') as string || `video_${Date.now()}.webm`
        const sessionId = formData.get('sessionId') as string

        logDebug(`File details: name=${fileName}, size=${file?.size}, type=${file?.type}`);
        logDebug(`Session ID: ${sessionId}`);

        if (!file) {
            logDebug("Error: No file provided");
            return NextResponse.json({ error: "No file provided" }, { status: 400 })
        }

        const result = await processAgent4Video({
            file,
            fileName,
            sessionId
        })

        logDebug(`Process result: ${JSON.stringify(result).substring(0, 200)}...`);

        return NextResponse.json(result)
    } catch (error: any) {
        logDebug(`Error: ${error.message}`);
        console.error("Agent 4 Video Error:", error)
        return NextResponse.json({ error: error.message }, { status: 500 })
    }
}
