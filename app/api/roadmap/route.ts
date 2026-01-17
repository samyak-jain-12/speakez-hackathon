import { NextRequest, NextResponse } from 'next/server'
import OpenAI from 'openai'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || '',
})

export async function POST(request: NextRequest) {
  try {
    const { sessionData, userProfile } = await request.json()

    // Build prompt for GPT
    const prompt = `Generate a personalized, supportive improvement roadmap for speech practice based on the following session data:

Session Metrics:
- Pace: ${sessionData.metrics?.pace || 'N/A'}
- Stress patterns: ${sessionData.metrics?.stress || 'N/A'}
- Filler words: ${sessionData.metrics?.fillerWords || 'N/A'}
- Stuttering: ${sessionData.metrics?.stuttering || 'N/A'}

User Profile:
- Goal: ${userProfile?.goal || 'Not specified'}
- Has speech disorder: ${userProfile?.hasSpeechDisorder ? 'Yes' : 'No'}

Generate a step-by-step, day-by-day practice plan that:
1. Is supportive and non-judgmental
2. Adapts to the user's personal baseline (not universal standards)
3. Provides specific exercises and drills
4. Includes breathing and pacing guidance
5. Offers disorder-aware suggestions if applicable
6. Is encouraging and builds confidence gradually

Format as markdown with clear sections. Use a warm, supportive tone throughout.`

    const completion = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: 'You are a supportive speech practice coach. Generate personalized, encouraging improvement roadmaps. Never use judgmental language. Focus on understanding and growth.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature: 0.7,
    })

    const roadmap = completion.choices[0]?.message?.content || 'Unable to generate roadmap at this time.'

    return NextResponse.json({ roadmap })
  } catch (error) {
    console.error('Error generating roadmap:', error)
    return NextResponse.json(
      { error: 'Failed to generate roadmap' },
      { status: 500 }
    )
  }
}
