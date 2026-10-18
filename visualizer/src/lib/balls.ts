// Per-ball render assets. Textures are authored procedurally from photos of
// the real balls, logo-free by construction; see docs/ball-textures.md and
// tools/ball-textures/ to regenerate.
import type { FormatKey } from './formats'

export type BallKey = 'molten' | 'mikasa'

export type BallAssets = {
  key: BallKey
  label: string
  albedo: string
  normal: string
  trail: string // motion-trail tint that suits the ball's palette
}

export const BALLS: Record<BallKey, BallAssets> = {
  molten: {
    key: 'molten',
    label: 'Molten V5M5000',
    albedo: '/textures/balls/molten_albedo.webp',
    normal: '/textures/balls/molten_normal.webp',
    trail: '#f3f1ea',
  },
  mikasa: {
    key: 'mikasa',
    label: 'Mikasa V200W',
    albedo: '/textures/balls/mikasa_albedo.webp',
    normal: '/textures/balls/mikasa_normal.webp',
    trail: '#ffd24a',
  },
}

// 9-man plays the Molten (NACIVT convention); indoor 6s the FIVB Mikasa.
export const FORMAT_BALL: Record<FormatKey, BallKey> = {
  '9man': 'molten',
  indoor6s: 'mikasa',
}
