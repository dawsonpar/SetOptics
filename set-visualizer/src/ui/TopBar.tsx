import { useStore } from '../state/store'
import { ShareButton } from './ShareButton'

export function TopBar() {
  const resetSet = useStore((s) => s.resetSet)
  return (
    <div className="pointer-events-none flex items-center justify-between">
      <div className="pointer-events-auto glass px-4 py-2">
        <span className="font-mono text-[13px] tracking-tight text-white">
          SetOptics
        </span>
        <span className="ml-2 font-mono text-[12px] text-muted">Set Visualizer</span>
      </div>
      <div className="pointer-events-auto flex gap-2">
        <button onClick={resetSet} className="pill">Reset</button>
        <ShareButton />
      </div>
    </div>
  )
}
