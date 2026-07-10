import { useState } from 'react'
import { useStore } from '../state/store'

const Chevron = ({ open }: { open: boolean }) => (
  <svg
    width="9"
    height="9"
    viewBox="0 0 10 10"
    className={`shrink-0 text-muted transition-transform duration-150 ${open ? 'rotate-90' : ''}`}
  >
    <path d="M3 1.5 L7 5 L3 8.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)

// Saved-sets popover content. Visibility is owned by the caller (the Saved
// chip in the control sheet); this panel only renders its own contents.
export function CollectionsPanel({ onClose }: { onClose: () => void }) {
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
    <div className="glass flex max-h-[50vh] w-full flex-col p-2.5 text-left">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted">
          Collections
          {collections.length > 0 && <span className="ml-1.5">· {collections.length}</span>}
        </span>
        <div className="flex items-center gap-1.5">
          <button onClick={onNew} className="pill px-2.5 py-1">+ new</button>
          <button
            onClick={onClose}
            aria-label="Close saved sets"
            className="px-1 text-muted transition hover:text-white"
          >
            ×
          </button>
        </div>
      </div>

      <div className="mt-2 flex flex-col overflow-hidden">
        {collections.length === 0 && (
          <p className="px-1 py-1 text-[12px] leading-relaxed text-muted">
            Group sets into folders, one per hitter, or "sets to work on". Saved
            in your browser.
          </p>
        )}

        {/* ~three collection rows visible; the rest scroll */}
        <div className="flex max-h-[8.5rem] flex-col gap-1 overflow-y-auto overscroll-contain [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {collections.map((c) => (
            <div key={c.id} className="rounded-xl border border-white/5 bg-surface-dark/40">
              <div className="flex items-center gap-1 px-2 py-1.5">
                <button
                  onClick={() => setOpen(open === c.id ? null : c.id)}
                  title={open === c.id ? 'Collapse' : 'Expand'}
                  className="flex flex-1 items-center gap-1.5 overflow-hidden text-left text-[13px] text-white/90"
                >
                  <span className={c.sets.length === 0 ? 'opacity-30' : ''}>
                    <Chevron open={open === c.id} />
                  </span>
                  <span className="truncate">{c.name}</span>
                  <span className="shrink-0 text-muted">· {c.sets.length}</span>
                </button>
                <button
                  onClick={() => onSave(c.id)}
                  title="Save current set here"
                  className="pill shrink-0 px-2 py-0.5 text-[11px]"
                >
                  save
                </button>
                <button
                  onClick={() => deleteCollection(c.id)}
                  title="Delete collection"
                  className="shrink-0 px-1 text-muted hover:text-white"
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
                        className="flex-1 truncate rounded-md px-2 py-1 text-left font-mono text-[12px] text-muted hover:bg-white/5 hover:text-white"
                      >
                        {st.name}
                      </button>
                      <button
                        onClick={() => deleteSet(c.id, st.id)}
                        className="shrink-0 px-1 text-muted hover:text-white"
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
    </div>
  )
}
