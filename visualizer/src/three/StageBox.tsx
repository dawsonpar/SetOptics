import { Line } from '@react-three/drei'
import { BRAND, STAGE_BOX_DEPTH, STAGE_BOX_WIDTH } from '../lib/constants'

// The visible outer box: a constant subtle-grey stage floor with a boundary
// line, surrounding the court on both sides of the net. Because it never
// scales with the format, court size and net height changes read against a
// fixed reference when switching between 9-man and 6s.
export function StageBox() {
  const hw = STAGE_BOX_WIDTH / 2
  const hd = STAGE_BOX_DEPTH / 2
  const rect: [number, number, number][] = [
    [-hw, 0.005, -hd], [hw, 0.005, -hd], [hw, 0.005, hd],
    [-hw, 0.005, hd], [-hw, 0.005, -hd],
  ]
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]} receiveShadow>
        <planeGeometry args={[STAGE_BOX_WIDTH, STAGE_BOX_DEPTH]} />
        <meshStandardMaterial color={BRAND.boxFloor} roughness={1} metalness={0} />
      </mesh>
      <Line points={rect} color={BRAND.boxLine} lineWidth={1} transparent opacity={0.9} />
    </group>
  )
}
