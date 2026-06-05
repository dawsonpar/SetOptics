import { useStore } from '../state/store'
import { ShareButton } from './ShareButton'

export function TopBar() {
  const resetSet = useStore((s) => s.resetSet)
  return (
    <div className="pointer-events-none flex items-center justify-between gap-2">
      <div className="pointer-events-auto glass whitespace-nowrap px-3 py-2 sm:px-4">
        <span className="font-mono text-[13px] tracking-tight text-white">SetOptics</span>
        <span className="ml-2 hidden font-mono text-[12px] text-muted sm:inline">
          Set Visualizer
        </span>
      </div>
      <div className="pointer-events-auto flex shrink-0 gap-2">
        <button onClick={resetSet} className="pill">Reset</button>
        <ShareButton />
      </div>
    </div>
  )
}
