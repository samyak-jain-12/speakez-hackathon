
import { NextResponse } from 'next/server'
import { processAgent7 } from '@/lib/agents/agent7'

export async function POST(req: Request) {
    try {
        const body = await req.json()
        const { scenario, current_speech, live_stats, timestamp } = body

        if (!scenario || !current_speech) {
            return NextResponse.json(
                { error: 'Missing required fields: scenario, current_speech' },
                { status: 400 }
            )
        }

        const result = await processAgent7({
            scenario,
            current_speech,
            live_stats,
            timestamp: timestamp || Date.now()
        })

        // Agent 7 returns structure like { data: { answer: "JSON String" } }
        // We need to parse the inner JSON if possible, or return as is

        let parsedAnswer = null
        if (result?.data?.answer) {
            try {
                // The AI might wrap it in markdown block ```json ... ```
                const cleanJson = result.data.answer.replace(/```json/g, '').replace(/```/g, '').trim()
                parsedAnswer = JSON.parse(cleanJson)
            } catch (e) {
                console.warn("Agent 7 answer was not valid JSON", result.data.answer)
                parsedAnswer = { raw_answer: result.data.answer }
            }
        }

        return NextResponse.json(parsedAnswer || result)

    } catch (error: any) {
        console.error('Error in Agent 7 Simulation:', error)
        return NextResponse.json(
            { error: error.message || 'Internal Server Error' },
            { status: 500 }
        )
    }
}
