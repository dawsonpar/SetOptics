import { DEFAULT_FORMAT, isFormatKey, type FormatKey } from './formats'
import type { SetState } from './types'

// Compact URL encoding of a set so it can be shared with no backend.
// ?s=setterX,setterZ,destX,destY,destZ,peakX,peakY,peakZ,speed[,format]
// The trailing format token was added with multi-format support; legacy
// 9-number links decode as 9-man.

const r = (n: number) => Math.round(n * 100) / 100

export function encodeSet(s: SetState, format: FormatKey): string {
  const nums = [
    s.setter.x, s.setter.z,
    s.dest.x, s.dest.y, s.dest.z,
    s.peak.x, s.peak.y, s.peak.z,
    s.speed,
  ].map(r)
  return nums.join(',') + ',' + format
}

export function decodeSet(
  param: string | null,
): { set: SetState; format: FormatKey } | null {
  if (!param) return null
  const parts = param.split(',')
  let format: FormatKey = DEFAULT_FORMAT
  if (parts.length === 10) {
    const tail = parts.pop() as string
    if (!isFormatKey(tail)) return null
    format = tail
  } else if (parts.length !== 9) {
    return null
  }
  const n = parts.map(Number)
  if (n.some((v) => Number.isNaN(v))) return null
  return {
    set: {
      setter: { x: n[0], z: n[1] },
      dest: { x: n[2], y: n[3], z: n[4] },
      peak: { x: n[5], y: n[6], z: n[7] },
      speed: n[8],
    },
    format,
  }
}

export function readSetFromUrl(): { set: SetState; format: FormatKey } | null {
  if (typeof window === 'undefined') return null
  return decodeSet(new URLSearchParams(window.location.search).get('s'))
}

export function shareUrlFor(s: SetState, format: FormatKey): string {
  const base = window.location.origin + window.location.pathname
  return `${base}?s=${encodeSet(s, format)}`
}
