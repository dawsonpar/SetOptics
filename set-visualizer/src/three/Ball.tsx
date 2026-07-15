import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Trail, useTexture } from '@react-three/drei'
import * as THREE from 'three'
import { useStore } from '../state/store'
import { pointAt } from '../lib/arc'
import { BALLS, FORMAT_BALL } from '../lib/balls'

const SPIN_RATE = 7 // rad/s of backspin while the ball is in flight

// Preload both balls so a format switch doesn't suspend mid-session.
Object.values(BALLS).forEach((b) => useTexture.preload([b.albedo, b.normal]))

const spinAxis = new THREE.Vector3()

export function Ball() {
  const group = useRef<THREE.Group>(null)
  const mesh = useRef<THREE.Mesh>(null)
  const format = useStore((s) => s.format)
  const ball = BALLS[FORMAT_BALL[format]]
  const [albedo, normal] = useTexture([ball.albedo, ball.normal])
  albedo.colorSpace = THREE.SRGBColorSpace

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
    if (group.current) {
      group.current.position.set(p.x, p.y, p.z)
      // squash a touch near the landing
      const squash = t > 0.92 ? 1 - (t - 0.92) * 3 : 1
      group.current.scale.set(1, Math.max(0.6, squash), 1)
    }
    if (mesh.current && s.playing) {
      // backspin about the horizontal axis perpendicular to travel
      const dx = s.set.dest.x - s.set.setter.x
      const dz = s.set.dest.z - s.set.setter.z
      spinAxis.set(-dz, 0, dx).normalize()
      mesh.current.rotateOnWorldAxis(spinAxis, dt * SPIN_RATE)
    }
  })

  return (
    <Trail width={1.4} length={5} color={ball.trail} attenuation={(w) => w * w} decay={1.2}>
      <group ref={group}>
        <mesh ref={mesh} castShadow>
          <sphereGeometry args={[0.14, 32, 32]} />
          <meshStandardMaterial
            map={albedo}
            normalMap={normal}
            roughness={0.55}
            metalness={0}
            emissive="#ffffff"
            emissiveMap={albedo}
            emissiveIntensity={0.22}
          />
        </mesh>
      </group>
    </Trail>
  )
}
