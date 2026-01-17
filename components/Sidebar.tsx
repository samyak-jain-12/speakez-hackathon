'use client'

import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import {
  LayoutDashboard,
  Mic,
  Target,
  Video,
  Map,
  BookOpen,
  Settings,
  LogOut,
  Calendar
} from 'lucide-react'
import { useStore } from '@/lib/store'


const navItems = [
  { href: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { href: '/dashboard/practice', icon: Video, label: 'Live Session' },
  { href: '/dashboard/history', icon: Calendar, label: 'Session History' },
  { href: '/dashboard/roadmap', icon: Map, label: 'Improvement Roadmap' },
  { href: '/dashboard/support', icon: BookOpen, label: 'Speech Pattern Support' },
  { href: '/dashboard/settings', icon: Settings, label: 'Profile & Settings' },
]

export default function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { setUser } = useStore()

  const handleLogout = () => {
    localStorage.removeItem('speakez_user')
    setUser(null)
    router.push('/')
  }

  return (
    <aside className="w-64 bg-white border-r border-gray-200 min-h-screen p-4">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-primary-600">SPEAKEZ</h1>
      </div>

      <nav className="space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = pathname === item.href

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${isActive
                ? 'bg-primary-50 text-primary-700 font-medium'
                : 'text-gray-700 hover:bg-gray-50'
                }`}
            >
              <Icon className="w-5 h-5" />
              <span>{item.label}</span>
            </Link>
          )
        })}
      </nav>

      <button
        onClick={handleLogout}
        className="mt-8 w-full flex items-center gap-3 px-4 py-3 rounded-lg text-gray-700 hover:bg-red-50 hover:text-red-700 transition-colors"
      >
        <LogOut className="w-5 h-5" />
        <span>Logout</span>
      </button>
    </aside>
  )
}
