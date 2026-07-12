import type { ReactNode } from 'react'

// Animated AI-gradient comet around a pill button. The ring is an SVG
// rounded-rect drawn with pathLength=100 and a linear dashoffset animation,
// so the comet travels the PERIMETER at constant speed (a rotating conic
// gradient reads as eased on a wide pill). The full palette rides along the
// tail as six staggered solid-color dashes sharing the same animation with
// phase offsets (negative delays), which keeps every color visible at once
// without changing the speed.
const COLORS = ['#f472b6', '#c084fc', '#818cf8', '#38bdf8', '#2dd4bf', '#fbbf24'] // tail -> head
const STEP = 3.5 // perimeter units between colors (of pathLength 100)
const PERIOD = 3 // seconds per revolution, matches .ai-comet in index.css

function CometStrokes({ width, className = '' }: { width: number; className?: string }) {
  return (
    <>
      {COLORS.map((color, i) => (
        <rect
          key={color}
          x="0"
          y="0"
          width="100%"
          height="100%"
          rx="999"
          pathLength={100}
          fill="none"
          stroke={color}
          strokeWidth={width}
          strokeLinecap="round"
          className={`ai-comet ${className}`}
          style={{ animationDelay: `${(-i * STEP * PERIOD) / 100}s` }}
        />
      ))}
    </>
  )
}

export function AIGradientBorder({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div className={`relative rounded-full border border-white/10 p-px ${className}`}>
      {/* crisp ring, under the content */}
      <svg aria-hidden className="absolute inset-0 h-full w-full overflow-visible">
        <CometStrokes width={1.5} />
      </svg>
      <div className="relative rounded-[inherit]">{children}</div>
      {/* glow above the content so the button's opaque corners cannot bite
          dark holes into it; the blur fades before reaching the label */}
      <svg aria-hidden className="pointer-events-none absolute inset-0 h-full w-full overflow-visible">
        <CometStrokes width={5} className="opacity-60 blur-[6px]" />
      </svg>
    </div>
  )
}
