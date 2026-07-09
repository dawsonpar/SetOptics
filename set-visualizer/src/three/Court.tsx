import { Line } from '@react-three/drei'
import { BRAND } from '../lib/constants'
import { FORMATS } from '../lib/formats'
import { useStore } from '../state/store'

// Half-court floor (the setter's side) with boundary + attack lines, sized by
// the active format. The camera stage is constant; the court scales within it.
export function Court() {
  const format = useStore((s) => FORMATS[s.format])
  const xMax = format.courtWidth / 2
  const xMin = -xMax
  const z0 = 0
  const z1 = format.halfDepth

  const rect: [number, number, number][] = [
    [xMin, 0.01, z0], [xMax, 0.01, z0], [xMax, 0.01, z1],
    [xMin, 0.01, z1], [xMin, 0.01, z0],
  ]

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, format.halfDepth / 2]} receiveShadow>
        <planeGeometry args={[format.courtWidth, format.halfDepth]} />
        <meshStandardMaterial color={BRAND.court} roughness={0.95} metalness={0} />
      </mesh>
      <Line points={rect} color={BRAND.courtLine} lineWidth={2} />
      {format.attackLine !== null && (
        <Line
          points={[[xMin, 0.01, format.attackLine], [xMax, 0.01, format.attackLine]]}
          color={BRAND.courtLine}
          lineWidth={1.5}
        />
      )}
    </group>
  )
}
