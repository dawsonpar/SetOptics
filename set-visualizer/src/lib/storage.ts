import type { Collection } from './types'

const KEY = 'setviz.collections.v1'

export function loadCollections(): Collection[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveCollections(cols: Collection[]): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(cols))
  } catch {
    // storage full / disabled — non-fatal for a local tool
  }
}

export function uid(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID()
  return 'id-' + Math.floor(performance.now() * 1000).toString(36)
}
