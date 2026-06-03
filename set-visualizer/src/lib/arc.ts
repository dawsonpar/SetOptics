import * as THREE from 'three'
import { CONTACT_HEIGHT } from './constants'
import type { SetState, Vec3 } from './types'

// The set arc is a quadratic Bezier whose apex sits ON the curve (at t = 0.5).
// To keep every set realistic, the apex is constrained:
//   - laterally / in depth it is locked to the MIDPOINT of start and dest
//   - its height is the only adjustable axis, and never dips below the higher
//     endpoint (no negative / U-shaped arcs)
// The Bezier control point is then derived so the curve passes through the apex:
//   B(0.5) = 0.25*P0 + 0.5*C + 0.25*P2  ==>  C = 2*apex - 0.5*(P0 + P2)

export function setterContact(s: SetState): Vec3 {
  return { x: s.setter.x, y: CONTACT_HEIGHT, z: s.setter.z }
}

// Minimum apex height so the arc always travels upward.
export function minApexHeight(s: SetState): number {
  return Math.max(CONTACT_HEIGHT, s.dest.y)
}

// The apex point actually used: midpoint in X/Z, clamped height in Y.
export function apexPoint(s: SetState): Vec3 {
  const p0 = setterContact(s)
  return {
    x: (p0.x + s.dest.x) / 2,
    y: Math.max(s.peak.y, minApexHeight(s)),
    z: (p0.z + s.dest.z) / 2,
  }
}

export function controlPoints(s: SetState): [THREE.Vector3, THREE.Vector3, THREE.Vector3] {
  const p0v = setterContact(s)
  const av = apexPoint(s)
  const p0 = new THREE.Vector3(p0v.x, p0v.y, p0v.z)
  const p2 = new THREE.Vector3(s.dest.x, s.dest.y, s.dest.z)
  const apex = new THREE.Vector3(av.x, av.y, av.z)
  // control = 2*apex - 0.5*(p0 + p2)
  const control = apex.clone().multiplyScalar(2).addScaledVector(p0, -0.5).addScaledVector(p2, -0.5)
  return [p0, control, p2]
}

export function makeCurve(s: SetState): THREE.QuadraticBezierCurve3 {
  const [p0, p1, p2] = controlPoints(s)
  return new THREE.QuadraticBezierCurve3(p0, p1, p2)
}

// Sample the curve into points for line rendering.
export function curvePoints(s: SetState, segments = 64): THREE.Vector3[] {
  return makeCurve(s).getPoints(segments)
}

// Position along the arc at parameter t in [0, 1].
export function pointAt(s: SetState, t: number): THREE.Vector3 {
  return makeCurve(s).getPoint(THREE.MathUtils.clamp(t, 0, 1))
}
