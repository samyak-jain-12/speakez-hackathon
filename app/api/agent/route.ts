import { NextRequest, NextResponse } from 'next/server'
import { processAgentInput } from '@/lib/agents/agent1'

export async function POST(req: NextRequest) {
    try {
        const formData = await req.formData()
        const query = formData.get('query') as string
        const file = formData.get('file') as Blob | null
        const fileName = formData.get('fileName') as string | undefined
        const sessionId = formData.get('sessionId') as string | undefined
        const externalUserId = formData.get('externalUserId') as string | undefined

        if (!query) {
            return NextResponse.json({ error: 'Query is required' }, { status: 400 })
        }

        const result = await processAgentInput({
            query,
            file: file || undefined,
            fileName,
            sessionId,
            externalUserId
        })

        return NextResponse.json(result)
    } catch (error: any) {
        console.error('Agent 1 Error:', error)
        return NextResponse.json(
            { error: error.message || 'Internal Server Error' },
            { status: 500 }
        )
    }
}
