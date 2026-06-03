import type { SetState } from './types'

// Compact URL encoding of a set so it can be shared with no backend.
// ?s=setterX,setterZ,destX,destY,destZ,peakX,peakY,peakZ,speed

const r = (n: number) => Math.round(n * 100) / 100

export function encodeSet(s: SetState): string {
  const nums = [
    s.setter.x, s.setter.z,
    s.dest.x, s.dest.y, s.dest.z,
    s.peak.x, s.peak.y, s.peak.z,
    s.speed,
  ].map(r)
  return nums.join(',')
}

export function decodeSet(param: string | null): SetState | null {
  if (!param) return null
  const n = param.split(',').map(Number)
  if (n.length !== 9 || n.some((v) => Number.isNaN(v))) return null
  return {
    setter: { x: n[0], z: n[1] },
    dest: { x: n[2], y: n[3], z: n[4] },
    peak: { x: n[5], y: n[6], z: n[7] },
    speed: n[8],
  }
}

export function readSetFromUrl(): SetState | null {
  if (typeof window === 'undefined') return null
  return decodeSet(new URLSearchParams(window.location.search).get('s'))
}

export function shareUrlFor(s: SetState): string {
  const base = window.location.origin + window.location.pathname
  return `${base}?s=${encodeSet(s)}`
}
