import { Line } from '@react-three/drei'
import { BRAND } from '../lib/constants'
import { FORMATS } from '../lib/formats'
import { useStore } from '../state/store'

// The full court (both halves), sized by the active format and always
// visible: boundary, attack lines on both sides, and the center line under
// the net. The camera stage and the surrounding StageBox are constant; the
// court scales within them.
export function Court() {
  const format = useStore((s) => FORMATS[s.format])
  const xMax = format.courtWidth / 2
  const xMin = -xMax
  const zMax = format.halfDepth

  const rect: [number, number, number][] = [
    [xMin, 0.01, -zMax], [xMax, 0.01, -zMax], [xMax, 0.01, zMax],
    [xMin, 0.01, zMax], [xMin, 0.01, -zMax],
  ]

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[format.courtWidth, format.halfDepth * 2]} />
        <meshStandardMaterial color={BRAND.court} roughness={0.95} metalness={0} />
      </mesh>
      <Line points={rect} color={BRAND.courtLine} lineWidth={2} />
      {/* center line under the net */}
      <Line points={[[xMin, 0.01, 0], [xMax, 0.01, 0]]} color={BRAND.courtLine} lineWidth={1.5} />
      {format.attackLine !== null && (
        <>
          <Line
            points={[[xMin, 0.01, format.attackLine], [xMax, 0.01, format.attackLine]]}
            color={BRAND.courtLine}
            lineWidth={1.5}
          />
          <Line
            points={[[xMin, 0.01, -format.attackLine], [xMax, 0.01, -format.attackLine]]}
            color={BRAND.courtLine}
            lineWidth={1.5}
          />
        </>
      )}
    </group>
  )
}
