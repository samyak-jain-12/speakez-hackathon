'use client'

import { useState, useEffect } from 'react'
import { useStore } from '@/lib/store'
import { useRouter } from 'next/navigation'
import { Settings as SettingsIcon, Globe, User } from 'lucide-react'

export default function SettingsPage() {
  const { user, setUser } = useStore()
  const router = useRouter()
  const [formData, setFormData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    hasSpeechDisorder: user?.hasSpeechDisorder || false,
    goal: user?.goal || '',
    currentAccent: user?.currentAccent || '',
    targetAccent: user?.targetAccent || '',
    preferredPace: 'medium',
    cameraEnabled: true,
  })

  useEffect(() => {
    if (user) {
      setFormData({
        name: user.name || '',
        email: user.email || '',
        hasSpeechDisorder: user.hasSpeechDisorder || false,
        goal: user.goal || '',
        currentAccent: user.currentAccent || '',
        targetAccent: user.targetAccent || '',
        preferredPace: 'medium',
        cameraEnabled: true,
      })
    }
  }, [user])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const updatedUser = {
      ...user,
      ...formData,
    }
    setUser(updatedUser as any)
    localStorage.setItem('speakez_user', JSON.stringify(updatedUser))
    alert('Settings saved successfully!')
  }

  const accents = [
    'American English',
    'British English',
    'Australian English',
    'Canadian English',
    'Indian English',
    'Other',
  ]

  const goals = [
    { value: 'pitching', label: 'Pitching' },
    { value: 'interview', label: 'Interview' },
    { value: 'conversation', label: 'Daily Conversation' },
    { value: 'accent', label: 'Accent Adaptation' },
  ]

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-8 text-gray-900 flex items-center gap-3">
        <SettingsIcon className="w-8 h-8 text-primary-600" />
        Profile & Settings
      </h1>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Basic Information */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 text-gray-900 flex items-center gap-2">
            <User className="w-5 h-5" />
            Basic Information
          </h2>
          
          <div className="space-y-4">
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                Name
              </label>
              <input
                type="text"
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                Email
              </label>
              <input
                type="email"
                id="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
          </div>
        </div>

        {/* Speech Preferences */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 text-gray-900">Speech Preferences</h2>
          
          <div className="space-y-4">
            <div>
              <label htmlFor="goal" className="block text-sm font-medium text-gray-700 mb-1">
                Primary Goal
              </label>
              <select
                id="goal"
                value={formData.goal}
                onChange={(e) => setFormData({ ...formData, goal: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="">Select a goal</option>
                {goals.map((goal) => (
                  <option key={goal.value} value={goal.value}>
                    {goal.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="hasSpeechDisorder"
                checked={formData.hasSpeechDisorder}
                onChange={(e) => setFormData({ ...formData, hasSpeechDisorder: e.target.checked })}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label htmlFor="hasSpeechDisorder" className="ml-2 block text-sm text-gray-700">
                I have a speech disorder / want personalized pacing
              </label>
            </div>

            <div>
              <label htmlFor="preferredPace" className="block text-sm font-medium text-gray-700 mb-1">
                Preferred Practice Pace
              </label>
              <select
                id="preferredPace"
                value={formData.preferredPace}
                onChange={(e) => setFormData({ ...formData, preferredPace: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="slow">Slow</option>
                <option value="medium">Medium</option>
                <option value="fast">Fast</option>
              </select>
            </div>
          </div>
        </div>

        {/* Accent & Relocation Mode */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 text-gray-900 flex items-center gap-2">
            <Globe className="w-5 h-5" />
            Accent & Relocation Mode
          </h2>
          
          <div className="space-y-4">
            <div>
              <label htmlFor="currentAccent" className="block text-sm font-medium text-gray-700 mb-1">
                Current Accent
              </label>
              <select
                id="currentAccent"
                value={formData.currentAccent}
                onChange={(e) => setFormData({ ...formData, currentAccent: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="">Select current accent</option>
                {accents.map((accent) => (
                  <option key={accent} value={accent}>
                    {accent}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="targetAccent" className="block text-sm font-medium text-gray-700 mb-1">
                Target Accent / Region
              </label>
              <select
                id="targetAccent"
                value={formData.targetAccent}
                onChange={(e) => setFormData({ ...formData, targetAccent: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="">Select target accent</option>
                {accents.map((accent) => (
                  <option key={accent} value={accent}>
                    {accent}
                  </option>
                ))}
              </select>
            </div>

            {formData.currentAccent && formData.targetAccent && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  Practice guidance will adapt to help you transition from {formData.currentAccent} to {formData.targetAccent}.
                  This includes pronunciation tips, stress patterns, and tone differences.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Privacy & Preferences */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 text-gray-900">Privacy & Preferences</h2>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label htmlFor="cameraEnabled" className="block text-sm font-medium text-gray-700">
                  Enable Camera by Default
                </label>
                <p className="text-sm text-gray-500">
                  Camera helps analyze body language, but you can toggle it per session
                </p>
              </div>
              <input
                type="checkbox"
                id="cameraEnabled"
                checked={formData.cameraEnabled}
                onChange={(e) => setFormData({ ...formData, cameraEnabled: e.target.checked })}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
            </div>
          </div>
        </div>

        <button
          type="submit"
          className="w-full bg-primary-600 text-white py-3 rounded-lg font-semibold hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
        >
          Save Settings
        </button>
      </form>
    </div>
  )
}
