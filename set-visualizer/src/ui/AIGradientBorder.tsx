import type { ReactNode } from 'react'

// Animated conic-gradient ring with a soft outer glow and a masked inner
// spill (hover.dev "AI gradient" pattern), implemented dependency-free: the
// rotation animates a CSS @property angle (see .ai-gradient in index.css)
// instead of pulling in a motion library.
export function AIGradientBorder({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div className={`relative rounded-full border border-white/10 p-px ${className}`}>
      {/* soft glow bleeding outside the ring */}
      <div aria-hidden className="ai-gradient absolute inset-0 rounded-[inherit] opacity-60 blur-md" />
      {/* crisp ring */}
      <div aria-hidden className="ai-gradient absolute inset-0 rounded-[inherit]" />
      <div className="relative overflow-hidden rounded-[inherit]">
        <div className="relative">{children}</div>
        {/* inner spill, masked so the center stays readable */}
        <div
          aria-hidden
          className="ai-gradient ai-glow-spill-mask pointer-events-none absolute inset-[-40%] z-10 opacity-70 blur-xl"
        />
      </div>
    </div>
  )
}
