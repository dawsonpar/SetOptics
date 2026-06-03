import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Trail } from '@react-three/drei'
import * as THREE from 'three'
import { useStore } from '../state/store'
import { pointAt } from '../lib/arc'
import { BRAND } from '../lib/constants'

export function Ball() {
  const ref = useRef<THREE.Mesh>(null)

  useFrame((_, dtRaw) => {
    const dt = Math.min(dtRaw, 0.05)
    const s = useStore.getState()
    let t = s.playhead
    if (s.playing) {
      // duration = set tempo / playback rate (higher rate = faster)
      const duration = Math.max(s.set.speed, 0.15) / s.playbackRate
      t = s.playhead + dt / duration
      if (t >= 1) { t = 1; s.setPlaying(false) }
      s.setPlayhead(t)
    }
    const p = pointAt(s.set, t)
    if (ref.current) {
      ref.current.position.set(p.x, p.y, p.z)
      // squash a touch near the landing
      const squash = t > 0.92 ? 1 - (t - 0.92) * 3 : 1
      ref.current.scale.set(1, Math.max(0.6, squash), 1)
    }
  })

  return (
    <Trail width={1.4} length={5} color={BRAND.ball} attenuation={(w) => w * w} decay={1.2}>
      <mesh ref={ref} castShadow>
        <sphereGeometry args={[0.14, 24, 24]} />
        <meshStandardMaterial color={BRAND.ball} emissive={BRAND.ball} emissiveIntensity={0.6} />
      </mesh>
    </Trail>
  )
}
