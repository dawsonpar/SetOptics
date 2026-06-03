import { useState } from 'react'
import { useStore } from '../state/store'

export function CollectionsPanel() {
  const collections = useStore((s) => s.collections)
  const [open, setOpen] = useState<string | null>(null)

  const createCollection = useStore((s) => s.createCollection)
  const deleteCollection = useStore((s) => s.deleteCollection)
  const saveCurrent = useStore((s) => s.saveCurrentToCollection)
  const deleteSet = useStore((s) => s.deleteSet)
  const loadSet = useStore((s) => s.loadSet)

  const onNew = () => {
    const name = window.prompt('New collection name (e.g. "Hitter: Marcus" or "Sets to work on")')
    if (name) setOpen(createCollection(name))
  }

  const onSave = (id: string) => {
    const name = window.prompt('Name this set (e.g. "go ball", "tight pass option")')
    if (name) saveCurrent(id, name)
  }

  return (
    <div className="glass flex max-h-[60vh] w-64 flex-col p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted">
          Collections
        </span>
        <button onClick={onNew} className="pill px-2.5 py-1">+ new</button>
      </div>

      {collections.length === 0 && (
        <p className="px-1 py-2 text-[12px] leading-relaxed text-muted">
          Group sets into folders, one per hitter, or "sets to work on". Saved
          in your browser.
        </p>
      )}

      <div className="flex flex-col gap-1 overflow-y-auto">
        {collections.map((c) => (
          <div key={c.id} className="rounded-xl border border-white/5 bg-surface-dark/40">
            <div className="flex items-center gap-1 px-2 py-1.5">
              <button
                onClick={() => setOpen(open === c.id ? null : c.id)}
                title={open === c.id ? 'Collapse' : 'Expand'}
                className="flex flex-1 items-center gap-1.5 text-left text-[13px] text-white/90"
              >
                <svg
                  width="9"
                  height="9"
                  viewBox="0 0 10 10"
                  className={`shrink-0 text-muted transition-transform duration-150 ${
                    open === c.id ? 'rotate-90' : ''
                  } ${c.sets.length === 0 ? 'opacity-30' : ''}`}
                >
                  <path d="M3 1.5 L7 5 L3 8.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span className="truncate">{c.name}</span>
                <span className="text-muted">· {c.sets.length}</span>
              </button>
              <button
                onClick={() => onSave(c.id)}
                title="Save current set here"
                className="pill px-2 py-0.5 text-[11px]"
              >
                save
              </button>
              <button
                onClick={() => deleteCollection(c.id)}
                title="Delete collection"
                className="px-1 text-muted hover:text-white"
              >
                ×
              </button>
            </div>
            {open === c.id && c.sets.length > 0 && (
              <div className="flex flex-col gap-0.5 px-2 pb-2">
                {c.sets.map((st) => (
                  <div key={st.id} className="flex items-center gap-1">
                    <button
                      onClick={() => loadSet(st)}
                      className="flex-1 rounded-md px-2 py-1 text-left font-mono text-[12px] text-muted hover:bg-white/5 hover:text-white"
                    >
                      {st.name}
                    </button>
                    <button
                      onClick={() => deleteSet(c.id, st.id)}
                      className="px-1 text-muted hover:text-white"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
