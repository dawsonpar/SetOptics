import {
  ANTENNA_ABOVE, ANTENNA_BAND, ANTENNA_X, BRAND, NET_BAND, NET_BODY_HEIGHT,
  NET_BOTTOM, NET_HEIGHT, POLE_X,
} from '../lib/constants'

// A red/white striped antenna at a sideline, from the net bottom up to 80cm
// above the net tape.
function Antenna({ x }: { x: number }) {
  const top = NET_HEIGHT + ANTENNA_ABOVE
  const total = top - NET_BOTTOM
  const bands = Math.round(total / ANTENNA_BAND)
  const band = total / bands
  return (
    <group position={[x, 0, 0]}>
      {Array.from({ length: bands }, (_, i) => (
        <mesh key={i} position={[0, NET_BOTTOM + band * (i + 0.5), 0]}>
          <cylinderGeometry args={[0.025, 0.025, band, 10]} />
          <meshStandardMaterial
            color={i % 2 === 0 ? BRAND.antennaWhite : BRAND.antennaRed}
            emissive={i % 2 === 0 ? BRAND.antennaWhite : BRAND.antennaRed}
            emissiveIntensity={0.18}
          />
        </mesh>
      ))}
    </group>
  )
}

// Net plane at Z = 0: body spans pole to pole, solid top tape, posts at the
// ends, antennas above the sidelines reaching the net bottom.
export function Net() {
  const width = POLE_X * 2
  return (
    <group position={[0, 0, 0]}>
      {/* translucent net body, pole to pole */}
      <mesh position={[0, NET_HEIGHT - NET_BODY_HEIGHT / 2, 0]}>
        <planeGeometry args={[width, NET_BODY_HEIGHT]} />
        <meshBasicMaterial color={BRAND.net} transparent opacity={0.12} depthWrite={false} />
      </mesh>
      {/* top tape */}
      <mesh position={[0, NET_HEIGHT, 0]}>
        <boxGeometry args={[width, NET_BAND, 0.04]} />
        <meshStandardMaterial color={BRAND.net} emissive={BRAND.net} emissiveIntensity={0.15} />
      </mesh>
      {/* posts at the net ends (1m outside the sidelines) */}
      {[-POLE_X, POLE_X].map((x) => (
        <mesh key={x} position={[x, NET_HEIGHT / 2, 0]} castShadow>
          <cylinderGeometry args={[0.05, 0.05, NET_HEIGHT, 12]} />
          <meshStandardMaterial color={BRAND.pole} metalness={0.3} roughness={0.6} />
        </mesh>
      ))}
      <Antenna x={-ANTENNA_X} />
      <Antenna x={ANTENNA_X} />
    </group>
  )
}
