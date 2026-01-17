'use client'

import { PlayCircle, BookOpen } from 'lucide-react'
import Link from 'next/link'

interface SpeechPatternAwarenessProps {
  patterns: string[]
  hasDisorder?: boolean
}

export default function SpeechPatternAwareness({
  patterns,
  hasDisorder = false,
}: SpeechPatternAwarenessProps) {
  if (patterns.length === 0) return null

  return (
    <div className="bg-gradient-to-br from-purple-50 to-blue-50 border-2 border-purple-200 rounded-lg p-6 mt-6">
      <div className="flex items-start gap-4">
        <div className="p-3 bg-purple-100 rounded-full">
          <BookOpen className="w-6 h-6 text-purple-600" />
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Speech Pattern Awareness
          </h3>
          <p className="text-gray-700 mb-4">
            We noticed {patterns.join(', ')} patterns during your practice. These are common and can improve with guided practice.
          </p>
          
          {hasDisorder && (
            <div className="bg-white/60 p-3 rounded-lg mb-4">
              <p className="text-sm text-gray-700">
                <strong>Note:</strong> This is not a medical diagnosis. These observations are meant to support your practice journey.
              </p>
            </div>
          )}
          
          <div className="flex gap-3">
            <Link
              href="/dashboard/resources"
              className="flex items-center gap-2 bg-primary-600 text-white px-4 py-2 rounded-lg hover:bg-primary-700 transition-colors text-sm font-medium"
            >
              <PlayCircle className="w-4 h-4" />
              Watch Recommended Videos
            </Link>
            <Link
              href="/dashboard/roadmap"
              className="flex items-center gap-2 bg-white border-2 border-primary-600 text-primary-600 px-4 py-2 rounded-lg hover:bg-primary-50 transition-colors text-sm font-medium"
            >
              <BookOpen className="w-4 h-4" />
              Explore Supportive Exercises
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
