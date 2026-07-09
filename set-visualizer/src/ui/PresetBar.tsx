import { useStore } from '../state/store'
import { FORMAT_KEYS, FORMATS } from '../lib/formats'
import { PRESETS } from '../lib/presets'

export function PresetBar() {
  const format = useStore((s) => s.format)
  const setFormat = useStore((s) => s.setFormat)
  const loadPreset = useStore((s) => s.loadPreset)
  return (
    <div className="glass flex max-w-[calc(100vw-1.5rem)] items-center gap-1.5 overflow-x-auto px-2 py-1.5 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
      <div className="flex shrink-0 gap-1" role="group" aria-label="Format">
        {FORMAT_KEYS.map((k) => (
          <button
            key={k}
            onClick={() => setFormat(k)}
            title={`${FORMATS[k].label} court`}
            className={`pill shrink-0 border-0 ${format === k ? 'pill-active' : 'bg-transparent'}`}
          >
            {FORMATS[k].label}
          </button>
        ))}
      </div>
      <span aria-hidden className="mx-0.5 h-4 w-px shrink-0 bg-white/10" />
      {PRESETS[format].map((p) => (
        <button key={p.key} onClick={() => loadPreset(p.key)} className="pill shrink-0 whitespace-nowrap">
          {p.label}
        </button>
      ))}
    </div>
  )
}
