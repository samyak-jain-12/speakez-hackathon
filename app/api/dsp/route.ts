import { NextRequest, NextResponse } from 'next/server'
import { extractDSPFeatures } from '@/lib/agents/dsp'

export async function POST(req: NextRequest) {
    try {
        const body = await req.json()
        const { audio_frame, audio_data, sample_rate } = body

        // Support both field names for flexibility
        const data = audio_data || audio_frame

        if (!data) {
            return NextResponse.json({ error: "Missing audio_frame or audio_data" }, { status: 400 })
        }

        // Use LOCAL Heuristics for real-time speed (1-second loop)
        // If we used Agent 2 here, the UI would freeze/lag.
        // We can call Agent 2 separately for the summary.
        const features = extractDSPFeatures(data, sample_rate || 16000)

        return NextResponse.json(features)
    } catch (error: any) {
        console.error("DSP Error", error)
        return NextResponse.json({ error: error.message }, { status: 500 })
    }
}
