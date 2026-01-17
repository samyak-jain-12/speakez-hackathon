import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const query = searchParams.get('q') || 'public speaking'
    const apiKey = process.env.YOUTUBE_API_KEY

    if (!apiKey) {
      // Return mock data if API key is not configured
      return NextResponse.json({
        items: [
          {
            id: { videoId: 'example1' },
            snippet: {
              title: 'Building Confidence in Public Speaking',
              channelTitle: 'Speech Coach',
              thumbnails: {
                medium: { url: 'https://via.placeholder.com/320x180' },
              },
            },
          },
          {
            id: { videoId: 'example2' },
            snippet: {
              title: 'Managing Speech Anxiety',
              channelTitle: 'Communication Skills',
              thumbnails: {
                medium: { url: 'https://via.placeholder.com/320x180' },
              },
            },
          },
        ],
      })
    }

    // In production, use YouTube Data API v3
    const response = await fetch(
      `https://www.googleapis.com/youtube/v3/search?part=snippet&q=${encodeURIComponent(query)}&type=video&maxResults=10&key=${apiKey}`
    )

    if (!response.ok) {
      throw new Error('YouTube API error')
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error fetching YouTube videos:', error)
    return NextResponse.json(
      { error: 'Failed to fetch videos' },
      { status: 500 }
    )
  }
}
