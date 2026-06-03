import { Line } from '@react-three/drei'
import { BRAND, HALF_DEPTH, X_MAX, X_MIN } from '../lib/constants'

// Half-court floor (the setter's side) with boundary + attack lines.
export function Court() {
  const z0 = 0
  const z1 = HALF_DEPTH
  const attackZ = 3 // attack line ~3m off the net

  const rect: [number, number, number][] = [
    [X_MIN, 0.01, z0], [X_MAX, 0.01, z0], [X_MAX, 0.01, z1],
    [X_MIN, 0.01, z1], [X_MIN, 0.01, z0],
  ]
  const attack: [number, number, number][] = [
    [X_MIN, 0.01, attackZ], [X_MAX, 0.01, attackZ],
  ]

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, HALF_DEPTH / 2]} receiveShadow>
        <planeGeometry args={[X_MAX - X_MIN, HALF_DEPTH]} />
        <meshStandardMaterial color={BRAND.court} roughness={0.95} metalness={0} />
      </mesh>
      <Line points={rect} color={BRAND.courtLine} lineWidth={2} />
      <Line points={attack} color={BRAND.courtLine} lineWidth={1.5} />
    </group>
  )
}
