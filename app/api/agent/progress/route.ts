
import { NextResponse } from 'next/server'
import { processAgent9 } from '@/lib/agents/agent9'

export async function POST(req: Request) {
    try {
        const body = await req.json()
        const { sessions } = body

        if (!sessions || !Array.isArray(sessions)) {
            return NextResponse.json({ error: "Invalid sessions array" }, { status: 400 })
        }

        const result = await processAgent9({ sessions })

        // Clean JSON buffer
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
