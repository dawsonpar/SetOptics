import { useStore } from '../state/store'

export function PlayBar() {
  const playing = useStore((s) => s.playing)
  const rate = useStore((s) => s.playbackRate)
  const playhead = useStore((s) => s.playhead)
  const togglePlay = useStore((s) => s.togglePlay)
  const setPlaybackRate = useStore((s) => s.setPlaybackRate)
  const setPlayhead = useStore((s) => s.setPlayhead)

  return (
    <div className="glass flex max-w-[calc(100vw-1.5rem)] items-center gap-2 px-3 py-2.5 sm:gap-4 sm:px-4 sm:py-3">
      <button
        onClick={togglePlay}
        aria-label={playing ? 'Pause' : 'Play set'}
        title={`${playing ? 'Pause' : 'Play'} (Space)`}
        className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-accent text-surface-dark shadow-glow transition hover:bg-accent/90 sm:h-12 sm:w-12"
      >
        {playing ? (
          <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor">
            <rect x="3" y="2" width="4" height="14" rx="1" />
            <rect x="11" y="2" width="4" height="14" rx="1" />
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor">
            <path d="M4 2.5v13l11-6.5z" />
          </svg>
        )}
      </button>

      <input
        type="range"
        min={0}
        max={1}
        step={0.001}
        value={playhead}
        onChange={(e) => {
          if (playing) useStore.getState().setPlaying(false)
          setPlayhead(Number(e.target.value))
        }}
        className="w-24 accent-accent sm:w-40"
        aria-label="Scrub set"
      />

      <div className="flex items-center gap-1.5 sm:gap-2">
        <span className="hidden font-mono text-[11px] text-muted sm:inline">speed</span>
        <input
          type="range"
          min={0.1}
          max={2}
          step={0.05}
          value={rate}
          onChange={(e) => setPlaybackRate(Number(e.target.value))}
          onDoubleClick={() => setPlaybackRate(1)}
          className="w-16 accent-primary-500 sm:w-28"
          aria-label="Playback speed"
          title="Double-click to reset to 1x"
        />
        <span className="w-9 shrink-0 font-mono text-[11px] text-white/70">{rate.toFixed(1)}x</span>
      </div>
    </div>
  )
}
