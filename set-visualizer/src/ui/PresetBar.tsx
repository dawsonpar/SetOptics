import { useStore } from '../state/store'
import { PRESETS } from '../lib/presets'

export function PresetBar() {
  const loadPreset = useStore((s) => s.loadPreset)
  return (
    <div className="glass flex items-center gap-1.5 px-2 py-1.5">
      <span className="px-2 font-mono text-[10px] uppercase tracking-wider text-muted">
        9-man sets
      </span>
      {PRESETS.map((p) => (
        <button key={p.key} onClick={() => loadPreset(p.key)} className="pill">
          {p.label}
        </button>
      ))}
    </div>
  )
}
