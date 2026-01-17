'use client'

import { useState, ReactNode } from 'react'
import { LucideIcon } from 'lucide-react'

interface InteractiveCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: LucideIcon
  iconColor?: string
  onClick?: () => void
  children?: ReactNode
  trend?: number[]
}

export default function InteractiveCard({
  title,
  value,
  subtitle,
  icon: Icon,
  iconColor = 'text-primary-500',
  onClick,
  children,
  trend,
}: InteractiveCardProps) {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`bg-white p-6 rounded-lg shadow-sm border border-gray-200 transition-all cursor-pointer ${
        isHovered ? 'shadow-md border-primary-200 scale-[1.02]' : ''
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex-1">
          <p className="text-sm text-gray-600 mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>
        <div className={`${iconColor} transition-transform ${isHovered ? 'scale-110' : ''}`}>
          <Icon className="w-12 h-12" />
        </div>
      </div>
      
      {trend && trend.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <div className="flex items-center gap-1 h-8">
            {trend.map((point, idx) => (
              <div
                key={idx}
                className="flex-1 bg-primary-100 rounded-sm"
                style={{ height: `${(point / Math.max(...trend)) * 100}%` }}
              />
            ))}
          </div>
        </div>
      )}
      
      {children}
    </div>
  )
}
