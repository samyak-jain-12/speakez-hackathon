
import { NextResponse } from 'next/server'
import { processAgent8 } from '@/lib/agents/agent8'

export async function POST(req: Request) {
    try {
        const body = await req.json()
        const { session_summary, weak_points } = body

        const result = await processAgent8({
            session_summary,
            weak_points: weak_points || []
        })

        // Clean JSON buffer if needed (Agent 8 prompt says "Expected output: JSON")
        let parsedData = result
        if (result?.data?.answer) {
            try {
                const cleanJson = result.data.answer.replace(/```json/g, '').replace(/```/g, '').trim()
                parsedData = JSON.parse(cleanJson)
            } catch (e) {
                parsedData = { raw_answer: result.data.answer }
            }
        }

        return NextResponse.json(parsedData)
    } catch (error: any) {
        return NextResponse.json({ error: error.message }, { status: 500 })
    }
}
