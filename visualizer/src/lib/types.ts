import type { FormatKey } from './formats'

export type Vec3 = { x: number; y: number; z: number }

export type ViewMode = 'front' | 'side' | 'orbit'

export type HandleId = 'setter' | 'dest' | 'peak'

// A set is fully described by these. Setter sits on the floor (y derived).
export type SetState = {
  setter: { x: number; z: number }
  dest: Vec3
  peak: Vec3
  speed: number // seconds for the ball to travel the arc
}

export type SavedSet = SetState & {
  id: string
  name: string
  format?: FormatKey // absent on pre-format saves; treated as 9-man
}

export type Collection = {
  id: string
  name: string
  sets: SavedSet[]
}
