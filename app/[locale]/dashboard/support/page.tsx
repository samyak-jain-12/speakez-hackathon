'use client'

import { useState } from 'react'
import { Heart, BookOpen, PlayCircle, Shield, Youtube, ExternalLink, Mic } from 'lucide-react'
import Link from 'next/link'
import PatternPill from '@/components/PatternPill'
import { generateMockPatterns } from '@/lib/mockData'

export default function SpeechSupportPage() {
  const [activeTab, setActiveTab] = useState<'general' | 'disorders'>('general')
  const [patterns] = useState(generateMockPatterns().filter(p =>
    p.name.includes('Repetition') || p.name.includes('Stress')
  ))

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-8 border-2 border-purple-200">
        <div className="flex items-start gap-4">
          <div className="p-4 bg-purple-100 rounded-full">
            <Heart className="w-8 h-8 text-purple-600" />
          </div>
          <div className="flex-1">
            <h1 className="text-3xl font-bold mb-4 text-gray-900">Speech Pattern Support</h1>
            <p className="text-lg text-gray-700 mb-4">
              Resources and understanding for your unique communication journey.
            </p>

            {/* Tab Navigation */}
            <div className="flex gap-4 border-b border-purple-200 mt-6">
              <button
                onClick={() => setActiveTab('general')}
                className={`pb-4 px-4 font-medium transition-colors ${activeTab === 'general'
                    ? 'text-purple-700 border-b-2 border-purple-600'
                    : 'text-gray-500 hover:text-gray-700'
                  }`}
              >
                General Patterns
              </button>
              <button
                onClick={() => setActiveTab('disorders')}
                className={`pb-4 px-4 font-medium transition-colors ${activeTab === 'disorders'
                    ? 'text-purple-700 border-b-2 border-purple-600'
                    : 'text-gray-500 hover:text-gray-700'
                  }`}
              >
                Speech Disorders & Stammering
              </button>
            </div>
          </div>
        </div>
      </div>

      {activeTab === 'general' ? (
        <div className="space-y-8 animate-fade-in">
          {/* Gentle Explanation */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">Understanding Speech Patterns</h2>
            <div className="prose max-w-none text-gray-700 space-y-4">
              <p>
                Everyone has unique speech patterns. Some patterns, like repetition, hesitation, or rapid pacing,
                are common and can be part of natural communication. These patterns dont indicate anything wrong
                with how you speak.
              </p>
              <p>
                Our platform observes these patterns to help you practice in ways that feel comfortable and build confidence.
                The goal is understanding and growth, not perfection.
              </p>
            </div>
          </div>

          {/* Patterns Observed */}
          {patterns.length > 0 && (
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h2 className="text-xl font-semibold mb-4 text-gray-900">Patterns Observed in Your Practice</h2>
              <p className="text-sm text-gray-600 mb-6">
                These are observations from your practice sessions. Click on any pattern to learn more.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {patterns.map((pattern) => (
                  <PatternPill key={pattern.id} pattern={pattern} />
                ))}
              </div>
            </div>
          )}

          {/* Practice Suggestions */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h2 className="text-xl font-semibold mb-4 text-gray-900 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-primary-500" />
              Supportive Practice Suggestions
            </h2>
            <div className="space-y-4">
              <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-900 mb-2">Pacing Practice</h3>
                <p className="text-sm text-gray-700 mb-2">
                  Practice speaking at a comfortable pace. There is no right speed—find what feels natural for you.
                </p>
              </div>
              <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-900 mb-2">Repetition & Hesitation</h3>
                <p className="text-sm text-gray-700 mb-2">
                  Repetition is natural. It gives your listeners time to process and you time to formulate thoughts.
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-8 animate-fade-in">
          {/* Supportive Disclaimer */}
          <div className="bg-amber-50 border border-amber-200 p-6 rounded-lg flex gap-4">
            <Shield className="w-6 h-6 text-amber-600 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-amber-900 mb-2">A Supportive Space</h3>
              <p className="text-amber-800 text-sm">
                If you experience stammering, stuttering, or other speech disorders, know that this is a safe space.
                SPEAKEZ is designed to help you build confidence without judgment. These resources are curated to support
                your journey, but always consult with a SLP (Speech-Language Pathologist) for professional guidance.
              </p>
            </div>
          </div>

          {/* YouTube Resources */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h2 className="text-xl font-semibold mb-6 text-gray-900 flex items-center gap-2">
              <Youtube className="w-6 h-6 text-red-600" />
              Learnings & Techniques
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                {
                  title: "Understanding Stuttering",
                  author: "The Stuttering Foundation",
                  url: "https://www.youtube.com/watch?v=Po-WMo8vXRY", // Example
                  desc: "A comprehensive guide to understanding the mechanics and psychology of stuttering."
                },
                {
                  title: "Breathing Techniques for Speech",
                  author: "Speech Therapy 101",
                  url: "https://www.youtube.com/watch?v=example2",
                  desc: "Learn diaphragmatic breathing to support smoother speech flow."
                },
                {
                  title: "Building Confidence with a Stutter",
                  author: "TEDx Talks",
                  url: "https://www.youtube.com/watch?v=example3",
                  desc: "Inspiring stories and practical tips for speaking with confidence."
                }
              ].map((video, idx) => (
                <a
                  key={idx}
                  href={video.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group block border border-gray-200 rounded-xl overflow-hidden hover:shadow-md transition-all"
                >
                  <div className="bg-gray-100 aspect-video flex items-center justify-center group-hover:bg-gray-200 transition-colors relative">
                    <PlayCircle className="w-12 h-12 text-gray-400 group-hover:text-red-600 transition-colors" />
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/5 transition-colors" />
                  </div>
                  <div className="p-4">
                    <h3 className="font-semibold text-gray-900 mb-1 group-hover:text-primary-600 transition-colors">
                      {video.title}
                    </h3>
                    <p className="text-xs text-gray-500 mb-3">{video.author}</p>
                    <p className="text-sm text-gray-600 line-clamp-2">{video.desc}</p>
                    <div className="mt-4 flex items-center gap-1 text-xs font-medium text-primary-600">
                      Watch on YouTube <ExternalLink className="w-3 h-3" />
                    </div>
                  </div>
                </a>
              ))}
            </div>
          </div>

          {/* Techniques Section */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h2 className="text-xl font-semibold mb-4 text-gray-900 flex items-center gap-2">
              <Mic className="w-5 h-5 text-primary-500" />
              Helpful Techniques
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-5 border border-gray-100 rounded-xl bg-gray-50">
                <h3 className="font-semibold text-gray-900 mb-2">Easy Onsets</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Start voicing gently by letting a little air out before the sound begins. This reduces tension in the vocal cords.
                </p>
                <div className="text-xs bg-white px-3 py-2 rounded border border-gray-200 inline-block text-gray-500">
                  Try saying Hhhh-ello instead of a hard Hello
                </div>
              </div>
              <div className="p-5 border border-gray-100 rounded-xl bg-gray-50">
                <h3 className="font-semibold text-gray-900 mb-2">Light Contact</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Touch the articulators (lips, tongue, teeth) together very lightly when producing consonants.
                </p>
                <div className="text-xs bg-white px-3 py-2 rounded border border-gray-200 inline-block text-gray-500">
                  Imagine your lips are made of bubbles you dont want to pop
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
