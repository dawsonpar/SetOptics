import { create } from 'zustand'
import { clamp, Y_MIN, Y_MAX } from '../lib/constants'
import { boundsFor, DEFAULT_FORMAT, FORMATS, type FormatKey } from '../lib/formats'
import { DEFAULT_SET, PRESETS } from '../lib/presets'
import { loadCollections, saveCollections, uid } from '../lib/storage'
import { readSetFromUrl } from '../lib/url'
import type { Collection, HandleId, SavedSet, SetState, Vec3, ViewMode } from '../lib/types'

const COACH_KEY = 'setviz.coachSeen'

const clampVecFor = (v: Vec3, f: FormatKey): Vec3 => {
  const b = boundsFor(f)
  return {
    x: clamp(v.x, b.xMin, b.xMax),
    y: clamp(v.y, Y_MIN, Y_MAX),
    z: clamp(v.z, b.zMin, b.zMax),
  }
}

// Clamp a whole set into a format's court, e.g. when switching formats or
// loading a set saved under a different (or unknown) format.
const clampSetFor = (s: SetState, f: FormatKey): SetState => {
  const b = boundsFor(f)
  return {
    setter: { x: clamp(s.setter.x, b.xMin, b.xMax), z: clamp(s.setter.z, b.zMin, b.zMax) },
    dest: clampVecFor(s.dest, f),
    peak: clampVecFor(s.peak, f),
    speed: s.speed,
  }
}

type Store = {
  set: SetState
  format: FormatKey
  view: ViewMode
  playing: boolean
  playhead: number
  dragging: HandleId | null
  // Orbit view is camera-first: pieces move there only after being armed by
  // a double-click. Only the setter is armable in the core tool.
  orbitArmed: 'setter' | null

  collections: Collection[]
  activeCollectionId: string | null
  showCoach: boolean
  playbackRate: number // YouTube-style: 1 = real tempo, 2 = 2x, 0.1 = slow-mo

  setSetter: (x: number, z: number) => void
  setDest: (v: Vec3) => void
  setPeak: (v: Vec3) => void
  setSpeed: (s: number) => void
  setFormat: (f: FormatKey) => void
  setPlaybackRate: (r: number) => void
  setView: (v: ViewMode) => void
  togglePlay: () => void
  setPlaying: (p: boolean) => void
  setPlayhead: (t: number) => void
  setDragging: (h: HandleId | null) => void
  setOrbitArmed: (v: 'setter' | null) => void

  loadPreset: (key: string) => void
  loadSet: (s: SetState & { format?: FormatKey }) => void
  resetSet: () => void

  setActiveCollection: (id: string | null) => void
  createCollection: (name: string) => string
  deleteCollection: (id: string) => void
  saveCurrentToCollection: (collectionId: string, name: string) => void
  deleteSet: (collectionId: string, setId: string) => void
  dismissCoach: () => void
}

const fromUrl = readSetFromUrl()
const initialFormat = fromUrl?.format ?? DEFAULT_FORMAT
const initialSet = fromUrl ? clampSetFor(fromUrl.set, initialFormat) : DEFAULT_SET
const sharedLink = fromUrl !== null
const coachSeen =
  typeof window !== 'undefined' && window.localStorage.getItem(COACH_KEY) === '1'

function persist(cols: Collection[]) {
  saveCollections(cols)
  return cols
}

export const useStore = create<Store>((setState) => ({
  set: initialSet,
  format: initialFormat,
  view: 'front',
  playing: false,
  playhead: 0,
  dragging: null,
  orbitArmed: null,
  collections: loadCollections(),
  activeCollectionId: null,
  showCoach: !sharedLink && !coachSeen,
  playbackRate: 1,

  setSetter: (x, z) =>
    setState((s) => {
      const b = boundsFor(s.format)
      return {
        set: { ...s.set, setter: { x: clamp(x, b.xMin, b.xMax), z: clamp(z, b.zMin, b.zMax) } },
      }
    }),
  setDest: (v) => setState((s) => ({ set: { ...s.set, dest: clampVecFor(v, s.format) } })),
  setPeak: (v) => setState((s) => ({ set: { ...s.set, peak: clampVecFor(v, s.format) } })),
  setSpeed: (sp) => setState((s) => ({ set: { ...s.set, speed: clamp(sp, 0.3, 3) } })),
  setFormat: (format) =>
    setState((s) => {
      if (!FORMATS[format] || format === s.format) return {}
      return { format, set: clampSetFor(s.set, format), playing: false }
    }),
  setPlaybackRate: (r) => setState({ playbackRate: clamp(r, 0.1, 2) }),
  setView: (view) => setState({ view, orbitArmed: null }),
  togglePlay: () =>
    setState((s) => {
      if (s.playing) return { playing: false } // pause, keep position
      // start: restart if finished, otherwise resume from current position
      return { playing: true, playhead: s.playhead >= 1 ? 0 : s.playhead }
    }),
  setPlaying: (playing) => setState({ playing }),
  setPlayhead: (playhead) => setState({ playhead }),
  setDragging: (dragging) => setState({ dragging }),
  setOrbitArmed: (orbitArmed) => setState({ orbitArmed }),

  loadPreset: (key) =>
    setState((s) => {
      const p = PRESETS[s.format].find((x) => x.key === key)
      if (!p) return {}
      return { set: structuredClone(p.set), playing: false, playhead: 0 }
    }),
  loadSet: (st) =>
    setState((s) => {
      const format = st.format && FORMATS[st.format] ? st.format : s.format
      const { setter, dest, peak, speed } = structuredClone(st)
      return {
        format,
        set: clampSetFor({ setter, dest, peak, speed }, format),
        playing: false,
        playhead: 0,
      }
    }),
  resetSet: () =>
    setState((s) => {
      const list = PRESETS[s.format]
      const p = list.find((x) => x.key === FORMATS[s.format].defaultPresetKey) ?? list[0]
      return { set: structuredClone(p.set), playing: false, playhead: 0 }
    }),

  setActiveCollection: (activeCollectionId) => setState({ activeCollectionId }),

  createCollection: (name) => {
    const col: Collection = { id: uid(), name: name.trim() || 'Untitled', sets: [] }
    setState((s) => ({
      collections: persist([...s.collections, col]),
      activeCollectionId: col.id,
    }))
    return col.id
  },

  deleteCollection: (id) =>
    setState((s) => ({
      collections: persist(s.collections.filter((c) => c.id !== id)),
      activeCollectionId: s.activeCollectionId === id ? null : s.activeCollectionId,
    })),

  saveCurrentToCollection: (collectionId, name) =>
    setState((s) => {
      const saved: SavedSet = {
        ...structuredClone(s.set),
        id: uid(),
        name: name.trim() || 'Set',
        format: s.format,
      }
      return {
        collections: persist(
          s.collections.map((c) =>
            c.id === collectionId ? { ...c, sets: [...c.sets, saved] } : c,
          ),
        ),
      }
    }),

  deleteSet: (collectionId, setId) =>
    setState((s) => ({
      collections: persist(
        s.collections.map((c) =>
          c.id === collectionId ? { ...c, sets: c.sets.filter((x) => x.id !== setId) } : c,
        ),
      ),
    })),

  dismissCoach: () => {
    if (typeof window !== 'undefined') window.localStorage.setItem(COACH_KEY, '1')
    setState({ showCoach: false })
  },
}))
