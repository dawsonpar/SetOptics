import { useStore } from '../state/store'
import { PRESETS } from '../lib/presets'

export function PresetBar() {
  const loadPreset = useStore((s) => s.loadPreset)
  return (
    <div className="glass flex max-w-[calc(100vw-1.5rem)] items-center gap-1.5 overflow-x-auto px-2 py-1.5 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
      <span className="shrink-0 px-2 font-mono text-[10px] uppercase tracking-wider text-muted">
        9-man sets
      </span>
      {PRESETS.map((p) => (
        <button key={p.key} onClick={() => loadPreset(p.key)} className="pill shrink-0 whitespace-nowrap">
          {p.label}
        </button>
      ))}
    </div>
  )
}
