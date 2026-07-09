// Court + scene constants. Units are meters.
// 9-man court is 10 x 20m; the setter operates in one 10 x 10m half.

export const COURT_WIDTH = 10 // X spans [-5, 5]
export const HALF_DEPTH = 10 // Z spans [0, 10], net at Z = 0
export const NET_HEIGHT = 2.35 // NACIVT 9-man net (235cm); indoor 6s men is 2.43
export const NET_BODY_HEIGHT = 1.8 // net mesh hangs this far below the tape
export const NET_BOTTOM = NET_HEIGHT - NET_BODY_HEIGHT
export const NET_BAND = 0.1 // visual net tape thickness
export const CONTACT_HEIGHT = 2.2 // setter hands height (ball start)

// Net hardware. Antennas sit above the sidelines (X = +/-5); poles are 1m
// outside the sidelines; antenna rises 80cm above the net (1.8m total).
export const POLE_X = 6
export const NET_HALF_WIDTH = 5.5 // net body extends ~0.5m past the antennas
export const ANTENNA_X = COURT_WIDTH / 2 // 5, on the sidelines
export const ANTENNA_ABOVE = 0.8
export const ANTENNA_TOTAL = 1.8
export const ANTENNA_BAND = 0.2 // visual stripe height

export const X_MIN = -COURT_WIDTH / 2
export const X_MAX = COURT_WIDTH / 2
export const Z_MIN = 0.2 // keep markers off the exact net line
export const Z_MAX = HALF_DEPTH
export const Y_MIN = 0
export const Y_MAX = 7 // generous ceiling for high arcs

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
