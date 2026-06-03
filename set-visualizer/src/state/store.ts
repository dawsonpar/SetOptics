import { create } from 'zustand'
import {
  clamp, X_MIN, X_MAX, Z_MIN, Z_MAX, Y_MIN, Y_MAX,
} from '../lib/constants'
import { DEFAULT_SET, PRESETS } from '../lib/presets'
import { loadCollections, saveCollections, uid } from '../lib/storage'
import { readSetFromUrl } from '../lib/url'
import type { Collection, HandleId, SetState, Vec3, ViewMode } from '../lib/types'

const COACH_KEY = 'setviz.coachSeen'

const clampVec = (v: Vec3): Vec3 => ({
  x: clamp(v.x, X_MIN, X_MAX),
  y: clamp(v.y, Y_MIN, Y_MAX),
  z: clamp(v.z, Z_MIN, Z_MAX),
})

type Store = {
  set: SetState
  view: ViewMode
  playing: boolean
  playhead: number
  dragging: HandleId | null

  collections: Collection[]
  activeCollectionId: string | null
  showCoach: boolean
  playbackRate: number // YouTube-style: 1 = real tempo, 2 = 2x, 0.1 = slow-mo

  setSetter: (x: number, z: number) => void
  setDest: (v: Vec3) => void
  setPeak: (v: Vec3) => void
  setSpeed: (s: number) => void
  setPlaybackRate: (r: number) => void
  setView: (v: ViewMode) => void
  togglePlay: () => void
  setPlaying: (p: boolean) => void
  setPlayhead: (t: number) => void
  setDragging: (h: HandleId | null) => void

  loadPreset: (key: string) => void
  loadSet: (s: SetState) => void
  resetSet: () => void

  setActiveCollection: (id: string | null) => void
  createCollection: (name: string) => string
  deleteCollection: (id: string) => void
  saveCurrentToCollection: (collectionId: string, name: string) => void
  deleteSet: (collectionId: string, setId: string) => void
  dismissCoach: () => void
}

const initialSet = readSetFromUrl() ?? DEFAULT_SET
const sharedLink = readSetFromUrl() !== null
const coachSeen =
  typeof window !== 'undefined' && window.localStorage.getItem(COACH_KEY) === '1'

function persist(cols: Collection[]) {
  saveCollections(cols)
  return cols
}

export const useStore = create<Store>((setState) => ({
  set: initialSet,
  view: 'front',
  playing: false,
  playhead: 0,
  dragging: null,
  collections: loadCollections(),
  activeCollectionId: null,
  showCoach: !sharedLink && !coachSeen,
  playbackRate: 1,

  setSetter: (x, z) =>
    setState((s) => ({
      set: { ...s.set, setter: { x: clamp(x, X_MIN, X_MAX), z: clamp(z, Z_MIN, Z_MAX) } },
    })),
  setDest: (v) => setState((s) => ({ set: { ...s.set, dest: clampVec(v) } })),
  setPeak: (v) => setState((s) => ({ set: { ...s.set, peak: clampVec(v) } })),
  setSpeed: (sp) => setState((s) => ({ set: { ...s.set, speed: clamp(sp, 0.3, 3) } })),
  setPlaybackRate: (r) => setState({ playbackRate: clamp(r, 0.1, 2) }),
  setView: (view) => setState({ view }),
  togglePlay: () =>
    setState((s) => {
      if (s.playing) return { playing: false } // pause, keep position
      // start: restart if finished, otherwise resume from current position
      return { playing: true, playhead: s.playhead >= 1 ? 0 : s.playhead }
    }),
  setPlaying: (playing) => setState({ playing }),
  setPlayhead: (playhead) => setState({ playhead }),
  setDragging: (dragging) => setState({ dragging }),

  loadPreset: (key) => {
    const p = PRESETS.find((x) => x.key === key)
    if (p) setState({ set: structuredClone(p.set), playing: false, playhead: 0 })
  },
  loadSet: (set) => setState({ set: structuredClone(set), playing: false, playhead: 0 }),
  resetSet: () => setState({ set: structuredClone(DEFAULT_SET), playing: false, playhead: 0 }),

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
      const saved = { ...structuredClone(s.set), id: uid(), name: name.trim() || 'Set' }
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
