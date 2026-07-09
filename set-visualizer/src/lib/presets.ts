import type { FormatKey } from './formats'
import type { SetState } from './types'

// Set presets per format, laid out left -> right along the net. Approximate
// geometry; the user adjusts live. Call names follow common usage (see the
// SetOptics domain research): lower/faster tempo = lower peak + shorter speed.
export type Preset = { key: string; label: string; set: SetState }

const NINE_MAN: Preset[] = [
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

// Indoor 6s: court X spans [-4.5, 4.5], net 2.43m, setter target right of
// center (~between zones 2 and 3), close to the net.
const INDOOR_6S: Preset[] = [
  {
    key: 'go',
    label: 'go',
    set: {
      setter: { x: 1.5, z: 1.5 },
      dest: { x: -3.9, y: 2.9, z: 0.5 },
      peak: { x: -1.2, y: 3.6, z: 1.0 },
      speed: 0.9,
    },
  },
  {
    key: 'hut',
    label: 'hut',
    set: {
      setter: { x: 1.5, z: 1.5 },
      dest: { x: -3.9, y: 2.8, z: 0.6 },
      peak: { x: -1.2, y: 4.8, z: 1.05 },
      speed: 1.4,
    },
  },
  {
    key: 'quick',
    label: '1 / quick',
    set: {
      setter: { x: 1.5, z: 1.5 },
      dest: { x: 1.1, y: 2.9, z: 0.5 },
      peak: { x: 1.3, y: 3.15, z: 1.0 },
      speed: 0.45,
    },
  },
  {
    key: '31',
    label: '31',
    set: {
      setter: { x: 1.5, z: 1.5 },
      dest: { x: -1.3, y: 2.9, z: 0.5 },
      peak: { x: 0.1, y: 3.2, z: 1.0 },
      speed: 0.6,
    },
  },
  {
    key: 'red',
    label: 'red (2)',
    set: {
      setter: { x: 1.5, z: 1.5 },
      dest: { x: 3.7, y: 2.9, z: 0.5 },
      peak: { x: 2.6, y: 3.4, z: 1.0 },
      speed: 0.8,
    },
  },
  {
    key: '5',
    label: '5',
    set: {
      setter: { x: 1.5, z: 1.5 },
      dest: { x: 3.9, y: 2.8, z: 0.6 },
      peak: { x: 2.7, y: 4.6, z: 1.05 },
      speed: 1.4,
    },
  },
  {
    key: 'pipe',
    label: 'pipe',
    set: {
      setter: { x: 1.5, z: 1.5 },
      dest: { x: 0, y: 2.6, z: 3.4 },
      peak: { x: 0.75, y: 4.4, z: 2.45 },
      speed: 1.3,
    },
  },
]

export const PRESETS: Record<FormatKey, Preset[]> = {
  '9man': NINE_MAN,
  indoor6s: INDOOR_6S,
}

export const DEFAULT_SET: SetState = NINE_MAN[2].set // t-ball, centered
