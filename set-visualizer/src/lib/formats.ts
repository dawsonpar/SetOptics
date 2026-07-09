import { Z_MIN } from './constants'

// Volleyball formats. The camera stage ("outer box") is constant; the court
// and net scale within it, so adding a format is purely a record entry here
// plus a preset list in presets.ts. Dimensions per the SetOptics domain
// research (indoor 6s 9x18m net 2.43m men; 9-man 10x20m net 2.35m NACIVT).
export type FormatKey = '9man' | 'indoor6s' // beach2s / grass4s: add here later

export type Format = {
  key: FormatKey
  label: string
  courtWidth: number // X span of the court, meters
  halfDepth: number // Z span of the setter's half, net at Z = 0
  netHeight: number
  netHalfWidth: number // net body extends ~0.5m past the antennas
  poleX: number // posts sit outside the sidelines
  attackLine: number | null // meters off the net; null = no line (beach)
  presetLabel: string
  defaultPresetKey: string // what Reset returns to in this format
}

export const FORMATS: Record<FormatKey, Format> = {
  '9man': {
    key: '9man',
    label: '9-man',
    courtWidth: 10,
    halfDepth: 10,
    netHeight: 2.35, // NACIVT
    netHalfWidth: 5.5,
    poleX: 6,
    attackLine: 3,
    presetLabel: '9-man sets',
    defaultPresetKey: 'tball',
  },
  indoor6s: {
    key: 'indoor6s',
    label: '6s',
    courtWidth: 9,
    halfDepth: 9,
    netHeight: 2.43, // men's indoor; women's 2.24 deliberately deferred
    netHalfWidth: 5,
    poleX: 5.5,
    attackLine: 3,
    presetLabel: '6s sets',
    defaultPresetKey: '31',
  },
}

export const DEFAULT_FORMAT: FormatKey = '9man'
export const FORMAT_KEYS = Object.keys(FORMATS) as FormatKey[]

export const isFormatKey = (v: string): v is FormatKey => v in FORMATS

// Draggable bounds for the active format. Y bounds live in constants.ts
// (they are about arc physics, not court size).
export function boundsFor(f: FormatKey) {
  const F = FORMATS[f]
  return {
    xMin: -F.courtWidth / 2,
    xMax: F.courtWidth / 2,
    zMin: Z_MIN,
    zMax: F.halfDepth,
  }
}
