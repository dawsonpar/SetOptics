import { useState, type ReactNode } from 'react'
import { useStore } from '../state/store'
import { FORMAT_KEYS, FORMATS } from '../lib/formats'
import { PRESETS } from '../lib/presets'
import { PlayBar } from './PlayBar'
import { ShareButton } from './ShareButton'
import { CollectionsPanel } from './CollectionsPanel'

// One labeled row of the pull-up sheet: label column on the left at sm+,
// label above the chips on mobile.
export function SheetRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-2.5">
      <span className="shrink-0 px-0.5 font-mono text-[10px] uppercase tracking-wider text-muted sm:w-14 sm:text-right">
        {label}
      </span>
      <div className="flex flex-wrap items-center gap-1">{children}</div>
    </div>
  )
}

const Chevron = ({ open }: { open: boolean }) => (
  <svg
    width="12"
    height="12"
    viewBox="0 0 12 12"
    className={`transition-transform duration-150 ${open ? 'rotate-180' : ''}`}
    aria-hidden
  >
    <path
      d="M2 7.5 L6 3.5 L10 7.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

// The bottom control stack, everything on one vertical axis: pull-up sheet
// (one row per category), the chevron that toggles it, and the play bar,
// which never moves and shares its row with nothing. Closed sheet = clean
// view. `topRows` lets a host app prepend rows (e.g. planning layers).
export function ControlSheet({ topRows }: { topRows?: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [savedOpen, setSavedOpen] = useState(false)
  const format = useStore((s) => s.format)
  const setFormat = useStore((s) => s.setFormat)
  const loadPreset = useStore((s) => s.loadPreset)
  const resetSet = useStore((s) => s.resetSet)

  const toggleSheet = () => {
    setOpen((v) => {
      if (v) setSavedOpen(false)
      return !v
    })
  }

  return (
    <div className="relative flex w-full flex-col items-center gap-1.5 sm:w-auto">
      {savedOpen && (
        <div className="absolute bottom-full z-30 mb-2 w-[min(18rem,calc(100vw-1.5rem))]">
          <CollectionsPanel onClose={() => setSavedOpen(false)} />
        </div>
      )}

      <button
        onClick={toggleSheet}
        aria-expanded={open}
        aria-label={open ? 'Hide controls' : 'More controls'}
        title={open ? 'Hide controls' : 'More controls'}
        className="glass rounded-full px-3.5 py-1 text-accent transition hover:text-white"
      >
        <Chevron open={open} />
      </button>

      {open && (
        <div className="glass w-full p-3 sm:w-auto sm:min-w-[26rem]">
          {/* mobile shows ~two rows and scrolls for the rest so the sheet
              never swallows the court; desktop shows everything */}
          <div className="-m-3 flex max-h-[10rem] flex-col gap-2 overflow-y-auto overscroll-contain p-3 [-ms-overflow-style:none] [scrollbar-width:none] sm:m-0 sm:max-h-none sm:overflow-visible sm:p-0 [&::-webkit-scrollbar]:hidden">
            {topRows}

            <SheetRow label="Format">
              {FORMAT_KEYS.map((k) => (
                <button
                  key={k}
                  onClick={() => setFormat(k)}
                  title={`${FORMATS[k].label} court`}
                  className={`pill ${format === k ? 'pill-active' : ''}`}
                >
                  {FORMATS[k].label}
                </button>
              ))}
            </SheetRow>

            <SheetRow label="Presets">
              {PRESETS[format].map((p) => (
                <button
                  key={p.key}
                  onClick={() => loadPreset(p.key)}
                  className="pill whitespace-nowrap"
                >
                  {p.label}
                </button>
              ))}
            </SheetRow>

            <SheetRow label="Actions">
              <button onClick={resetSet} className="pill">
                Reset
              </button>
              <ShareButton />
              <button
                onClick={() => setSavedOpen((v) => !v)}
                aria-expanded={savedOpen}
                title="Saved sets and collections"
                className={`pill ${savedOpen ? 'pill-active' : ''}`}
              >
                Saved
              </button>
            </SheetRow>
          </div>
        </div>
      )}

      <PlayBar />
    </div>
  )
}
