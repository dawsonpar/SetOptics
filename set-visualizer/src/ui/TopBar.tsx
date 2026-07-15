import { AIGradientBorder } from './AIGradientBorder'

const ArrowUpRight = () => (
  <svg width="11" height="11" viewBox="0 0 12 12" fill="none" aria-hidden>
    <path
      d="M3 9 L9 3 M9 3 H4.2 M9 3 V7.8"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

export function TopBar() {
  return (
    <div className="pointer-events-none flex items-center justify-between gap-2">
      <div className="pointer-events-auto glass flex items-center justify-center whitespace-nowrap rounded-full px-3 py-2 sm:px-4">
        <span className="font-mono text-[13px] font-medium tracking-tight text-white">SetOptics</span>
        <span className="ml-2 hidden font-mono text-[12px] text-muted sm:inline">
          Set Visualizer
        </span>
      </div>
      <div className="pointer-events-auto flex shrink-0 gap-2">
        <AIGradientBorder className="hidden sm:block">
          <a
            href="https://www.setoptics.com/visualizer"
            title="Save your sets and analyze your real footage on SetOptics"
            className="flex items-center gap-1.5 whitespace-nowrap rounded-full bg-surface px-3.5 py-2 font-mono text-[12px] text-white/90 transition hover:text-white [word-spacing:0.16em]"
          >
            <span>Open on <span className="font-medium">SetOptics</span></span>
            <ArrowUpRight />
          </a>
        </AIGradientBorder>
      </div>
    </div>
  )
}
