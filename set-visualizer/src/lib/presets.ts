import type { SetState } from './types'

// 9-man set presets, laid out left -> right along the net to match Dawson's
// hand-drawn reference. Approximate geometry; the user adjusts live.
export type Preset = { key: string; label: string; set: SetState }

export const PRESETS: Preset[] = [
  {
    key: '4go',
    label: '4 / go',
    set: {
      setter: { x: 1, z: 2.2 },
      dest: { x: -4.2, y: 2.7, z: 0.5 },
      peak: { x: -1.5, y: 4.6, z: 1.3 },
      speed: 1.5,
    },
  },
  {
    key: '31',
    label: '31',
    set: {
      setter: { x: 1, z: 2.2 },
      dest: { x: -1.8, y: 2.9, z: 0.5 },
      peak: { x: -0.4, y: 3.3, z: 1.2 },
      speed: 0.7,
    },
  },
  {
    key: 'tball',
    label: 't-ball (2)',
    set: {
      setter: { x: 1.2, z: 2.2 },
      dest: { x: 0, y: 2.8, z: 0.5 },
      peak: { x: 0.6, y: 3.6, z: 1.2 },
      speed: 1.0,
    },
  },
  {
    key: 'fasthook',
    label: 'fast / hook',
    set: {
      setter: { x: 1.2, z: 2.0 },
      dest: { x: 1.9, y: 2.9, z: 0.4 },
      peak: { x: 1.5, y: 3.2, z: 1.0 },
      speed: 0.6,
    },
  },
  {
    key: '5red',
    label: '5 / red',
    set: {
      setter: { x: 1, z: 2.2 },
      dest: { x: 4.2, y: 2.7, z: 0.5 },
      peak: { x: 2.5, y: 4.4, z: 1.3 },
      speed: 1.5,
    },
  },
]

export const DEFAULT_SET: SetState = PRESETS[2].set // t-ball, centered
