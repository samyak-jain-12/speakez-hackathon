'use client'

import { useState, useEffect } from 'react'
import { useStore } from '@/lib/store'
import { BookOpen, ExternalLink, Loader2 } from 'lucide-react'

export default function ResourcesPage() {
  const { user, practiceHistory } = useStore()
  const [videos, setVideos] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  // Detect patterns from practice history to recommend relevant videos
  const detectPatterns = () => {
    if (practiceHistory.length === 0) return []
    
    const lastSession = practiceHistory[practiceHistory.length - 1]
    const patterns: string[] = []
    
    if (lastSession.metrics?.stuttering?.some((s: number) => s > 0.3)) {
      patterns.push('stammering')
    }
    if (lastSession.metrics?.stress?.some((s: number) => s > 0.6)) {
      patterns.push('confidence')
    }
    if (lastSession.metrics?.pace?.some((p: number) => p > 180)) {
      patterns.push('pitch delivery')
    }
    if (user?.goal === 'accent') {
      patterns.push('accent adaptation')
    }
    
    return patterns.length > 0 ? patterns : ['public speaking']
  }

  const fetchVideos = async () => {
    setLoading(true)
    try {
      const patterns = detectPatterns()
      const searchQuery = patterns.join(' OR ') || 'public speaking'
      
      // In production, use YouTube Data API
      // For now, simulate with mock data
      const mockVideos = [
        {
          id: '1',
          title: 'Building Confidence in Public Speaking',
          channel: 'Speech Coach',
          thumbnail: 'https://via.placeholder.com/320x180',
          url: 'https://youtube.com/watch?v=example1',
        },
        {
          id: '2',
          title: 'Managing Speech Anxiety',
          channel: 'Communication Skills',
          thumbnail: 'https://via.placeholder.com/320x180',
          url: 'https://youtube.com/watch?v=example2',
        },
        {
          id: '3',
          title: 'Effective Pitch Delivery Techniques',
          channel: 'Startup Academy',
          thumbnail: 'https://via.placeholder.com/320x180',
          url: 'https://youtube.com/watch?v=example3',
        },
      ]
      
      setVideos(mockVideos)
    } catch (error) {
      console.error('Error fetching videos:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchVideos()
  }, [practiceHistory, user])

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2 text-gray-900 flex items-center gap-3">
            <BookOpen className="w-8 h-8 text-primary-600" />
            Learning Resources
          </h1>
          <p className="text-gray-600">
            Curated YouTube videos based on your practice patterns and goals
          </p>
        </div>
        <button
          onClick={fetchVideos}
          className="bg-primary-600 text-white px-4 py-2 rounded-lg hover:bg-primary-700 transition-colors"
        >
          Refresh Recommendations
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
        </div>
      ) : videos.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {videos.map((video) => (
            <div
              key={video.id}
              className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow"
            >
              <div className="aspect-video bg-gray-200 relative">
                <img
                  src={video.thumbnail}
                  alt={video.title}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="p-4">
                <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2">
                  {video.title}
                </h3>
                <p className="text-sm text-gray-600 mb-3">{video.channel}</p>
                <a
                  href={video.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-600 hover:text-primary-700 text-sm font-medium flex items-center gap-1"
                >
                  Watch on YouTube
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-white p-12 rounded-lg shadow-sm border border-gray-200 text-center">
          <BookOpen className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2 text-gray-900">No Recommendations Yet</h3>
          <p className="text-gray-600">
            Complete practice sessions to get personalized video recommendations.
          </p>
        </div>
      )}

      <div className="mt-8 bg-blue-50 border border-blue-200 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-2 text-blue-900">About Recommendations</h3>
        <p className="text-blue-800 text-sm">
          Videos are recommended based on patterns detected in your practice sessions and your selected goals. 
          These resources complement your practice and provide additional learning opportunities.
        </p>
      </div>
    </div>
  )
}
