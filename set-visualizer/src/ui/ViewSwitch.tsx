import { useStore } from '../state/store'
import type { ViewMode } from '../lib/types'

const VIEWS: { key: ViewMode; label: string; hint: string }[] = [
  { key: 'front', label: 'Front', hint: 'Shift+1' },
  { key: 'side', label: 'Side', hint: 'Shift+2' },
  { key: 'orbit', label: 'Orbit', hint: 'Shift+3' },
]

export function ViewSwitch() {
  const view = useStore((s) => s.view)
  const setView = useStore((s) => s.setView)
  return (
    <div className="glass flex gap-1 p-1">
      {VIEWS.map((v) => (
        <button
          key={v.key}
          onClick={() => setView(v.key)}
          title={`${v.label} view (${v.hint})`}
          className={`pill border-0 ${view === v.key ? 'pill-active' : 'bg-transparent'}`}
        >
          {v.label}
        </button>
      ))}
    </div>
  )
}
