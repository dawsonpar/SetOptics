// Scene + brand constants. Units are meters.
// Court/net dimensions are per-format and live in formats.ts; this file keeps
// what is format-independent: the fixed camera stage, arc physics bounds, net
// visual proportions, and the brand palette.

export const CONTACT_HEIGHT = 2.2 // setter hands height (ball start)

// Net visual proportions (heights come from the active format).
export const NET_BODY_HEIGHT = 1.8 // net mesh hangs this far below the tape
export const NET_BAND = 0.1 // visual net tape thickness
export const ANTENNA_ABOVE = 0.8 // antenna rises this far above the tape
export const ANTENNA_BAND = 0.2 // visual stripe height

// Drag bounds that are about arc physics, not court size.
export const Z_MIN = 0.2 // keep markers off the exact net line
export const Y_MIN = 0
export const Y_MAX = 7 // generous ceiling for high arcs

// The outer box: the camera stage is sized once for the largest court
// (9-man, 10m deep half) and never changes between formats — courts scale
// within it. CameraRig framings assume this value.
export const STAGE_HALF_DEPTH = 10

// SetOptics brand
export const BRAND = {
  bg: '#0a0a0f',
  surface: '#1a1a2e',
  surfaceLight: '#252540',
  cobalt: '#0761b2',
  accent: '#4f9cf7',
  accentDim: '#2d6bc4',
  muted: '#a0a0b8',
  court: '#14233f',
  courtLine: '#3a5680',
  net: '#cfe0f5',
  setter: '#4f9cf7',
  ball: '#ffd24a',
  arc: '#4f9cf7',
  antennaRed: '#e23b2e',
  antennaWhite: '#f2f2f2',
  pole: '#d8dde6',
} as const

export const clamp = (v: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, v))
