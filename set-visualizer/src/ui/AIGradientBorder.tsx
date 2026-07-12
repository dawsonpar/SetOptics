import { useId, type ReactNode } from 'react'

// Animated AI-gradient ring (hover.dev pattern) with one important change:
// the highlight is an SVG stroke dash traveling a pathLength-normalized
// rounded-rect, so it moves along the PERIMETER at constant speed. A conic
// gradient rotating at constant angle looks eased on a wide pill: it crawls
// across the middle of the long edges and whips around the ends.
export function AIGradientBorder({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  const gradId = useId()
  return (
    <div className={`relative rounded-full border border-white/10 p-px ${className}`}>
      <svg aria-hidden className="absolute inset-0 h-full w-full overflow-visible">
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#f472b6" />
            <stop offset="0.2" stopColor="#c084fc" />
            <stop offset="0.4" stopColor="#818cf8" />
            <stop offset="0.6" stopColor="#38bdf8" />
            <stop offset="0.8" stopColor="#2dd4bf" />
            <stop offset="1" stopColor="#fbbf24" />
          </linearGradient>
        </defs>
        {/* soft glow following the comet */}
        <rect
          x="0"
          y="0"
          width="100%"
          height="100%"
          rx="999"
          pathLength={100}
          fill="none"
          stroke={`url(#${gradId})`}
          strokeWidth="5"
          strokeLinecap="round"
          className="ai-comet opacity-70 blur-[6px]"
        />
        {/* crisp ring segment */}
        <rect
          x="0"
          y="0"
          width="100%"
          height="100%"
          rx="999"
          pathLength={100}
          fill="none"
          stroke={`url(#${gradId})`}
          strokeWidth="1.5"
          strokeLinecap="round"
          className="ai-comet"
        />
      </svg>
      <div className="relative rounded-[inherit]">{children}</div>
    </div>
  )
}
