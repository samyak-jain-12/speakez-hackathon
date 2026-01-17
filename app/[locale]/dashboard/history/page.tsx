'use client'

import { useStore } from '@/lib/store'
import { Calendar, Clock, PlayCircle, BarChart2, Video } from 'lucide-react'
import Link from 'next/link'
import { useState } from 'react'

export default function HistoryPage() {
    const { practiceHistory } = useStore()
    const [selectedVideo, setSelectedVideo] = useState<string | null>(null)

    // Sort by date descending
    const sortedHistory = [...practiceHistory].sort((a, b) =>
        new Date(b.startTime).getTime() - new Date(a.startTime).getTime()
    )

    const handlePlayRecording = (sessionId: string) => {
        setSelectedVideo(sessionId)
    }

    return (
        <div className="max-w-6xl mx-auto space-y-8">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold text-gray-900">Session History</h1>
                <div className="flex items-center gap-2 text-sm text-gray-500 bg-gray-50 px-4 py-2 rounded-lg">
                    <Calendar className="w-4 h-4" />
                    <span>Total Sessions: {practiceHistory.length}</span>
                </div>
            </div>

            {sortedHistory.length === 0 ? (
                <div className="text-center py-16 bg-white rounded-lg border border-dashed border-gray-300">
                    <div className="bg-gray-50 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Video className="w-8 h-8 text-gray-400" />
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Sessions Yet</h3>
                    <p className="text-gray-500 mb-6">Start your first live session to see your history and progress.</p>
                    <Link
                        href="/dashboard/practice"
                        className="inline-flex items-center gap-2 px-6 py-3 bg-primary-600 text-text-light rounded-lg hover:bg-primary-700 transition-colors"
                    >
                        <Video className="w-4 h-4" />
                        Start Session
                    </Link>
                </div>
            ) : (
                <div className="grid grid-cols-1 gap-6">
                    {sortedHistory.map((session) => (
                        <div key={session.id} className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:border-primary-200 transition-all">
                            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                                <div className="flex items-start gap-4">
                                    <div className="w-12 h-12 bg-primary-50 rounded-lg flex items-center justify-center flex-shrink-0">
                                        <Video className="w-6 h-6 text-primary-600" />
                                    </div>
                                    <div>
                                        <h3 className="font-semibold text-lg text-gray-900 mb-1">
                                            {session.scenario ? session.scenario : (session.mode === 'pitch' ? 'Pitch Practice' : 'Conversation Practice')}
                                        </h3>
                                        <div className="flex items-center gap-4 text-sm text-gray-500">
                                            <span className="flex items-center gap-1">
                                                <Calendar className="w-3 h-3" />
                                                {new Date(session.startTime).toLocaleDateString()}
                                            </span>
                                            <span className="flex items-center gap-1">
                                                <Clock className="w-3 h-3" />
                                                {new Date(session.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                {/* Metrics Preview */}
                                <div className="flex items-center gap-6">
                                    <div className="text-center">
                                        <p className="text-xs text-gray-500 uppercase font-medium">Pace</p>
                                        <p className="font-semibold text-gray-900">
                                            {session.metrics.pace.length > 0
                                                ? Math.round(session.metrics.pace.reduce((a, b) => a + b, 0) / session.metrics.pace.length)
                                                : 0} WPM
                                        </p>
                                    </div>
                                    <div className="text-center">
                                        <p className="text-xs text-gray-500 uppercase font-medium">Confidence</p>
                                        <p className="font-semibold text-green-600">
                                            {/* {session.confidenceScore ? Math.round(session.confidenceScore * 10) / 10 : '-'}/10 */}
                                        </p>
                                    </div>
                                </div>

                                <div className="flex items-center gap-3">
                                    <button
                                        onClick={() => handlePlayRecording(session.id)}
                                        className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                                    >
                                        <PlayCircle className="w-4 h-4" />
                                        Watch Recording
                                    </button>
                                    <Link
                                        href={`/dashboard/insights?session=${session.id}`}
                                        className="flex items-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                                    >
                                        <BarChart2 className="w-4 h-4" />
                                        View Insights
                                    </Link>
                                </div>
                            </div>

                            {/* Video Player Modal (Simulated) */}
                            {selectedVideo === session.id && (
                                <div className="mt-6 bg-black rounded-lg overflow-hidden aspect-video relative animate-fade-in group">
                                    <div className="absolute inset-0 flex items-center justify-center bg-black/60 group-hover:bg-black/40 transition-all cursor-pointer">
                                        <div className="bg-white/20 backdrop-blur-md p-6 rounded-full group-hover:scale-110 transition-transform">
                                            <PlayCircle className="w-16 h-16 text-white" />
                                        </div>
                                    </div>
                                    <div className="absolute bottom-4 left-4 right-4 text-white">
                                        <p className="text-sm font-medium opacity-90">Recording Simulation</p>
                                        <div className="w-full bg-white/30 h-1 rounded-full mt-2">
                                            <div className="w-1/3 bg-primary-500 h-full rounded-full"></div>
                                        </div>
                                    </div>
                                    {/* Close button for simulation */}
                                    <button
                                        onClick={(e) => { e.stopPropagation(); setSelectedVideo(null); }}
                                        className="absolute top-4 right-4 bg-black/50 text-white p-2 rounded-full hover:bg-black/70"
                                    >
                                        ×
                                    </button>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
