import { useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useStore } from '../state/store'
import type { ViewMode } from '../lib/types'

// Each view defines what to look at (target), from which direction, and how
// large an area must stay in frame (halfW x halfH, in world units). The camera
// distance is then computed from the live viewport aspect so the court always
// fits, on a wide monitor or a narrow phone, instead of being cropped.
type Framing = { target: THREE.Vector3; dir: THREE.Vector3; halfW: number; halfH: number }

const FRAMINGS: Record<ViewMode, Framing> = {
  front: {
    target: new THREE.Vector3(0, 2.5, 0.6),
    dir: new THREE.Vector3(0, 0.18, 1).normalize(),
    halfW: 6.8, // half the pole-to-pole width + margin
    halfH: 2.9,
  },
  side: {
    target: new THREE.Vector3(0, 2.5, 3.5),
    dir: new THREE.Vector3(1, 0.18, 0).normalize(),
    halfW: 4.4,
    halfH: 2.9,
  },
  orbit: {
    target: new THREE.Vector3(0, 1.8, 3),
    dir: new THREE.Vector3(0.7, 0.55, 1).normalize(),
    halfW: 7.4,
    halfH: 3.6,
  },
}
const PADDING = 1.12

function fitDistance(f: Framing, aspect: number, fovDeg: number): number {
  const tanHalf = Math.tan((fovDeg * Math.PI) / 180 / 2)
  const distH = f.halfH / tanHalf
  const distW = f.halfW / (tanHalf * Math.max(aspect, 0.0001))
  return Math.max(distH, distW) * PADDING
}

const tmpPos = new THREE.Vector3()

export function CameraRig() {
  const controls = useRef<OrbitControlsImpl>(null)
  const { camera, size } = useThree()
  const view = useStore((s) => s.view)
  const dragging = useStore((s) => s.dragging)
  const orbitInit = useRef(false)

  useFrame(() => {
    const c = controls.current
    if (!c) return
    const aspect = size.width / size.height
    const fovDeg = (camera as THREE.PerspectiveCamera).fov

    if (view === 'orbit') {
      c.enabled = !dragging
      if (!orbitInit.current) {
        orbitInit.current = true
        const f = FRAMINGS.orbit
        const d = fitDistance(f, aspect, fovDeg)
        camera.position.copy(f.target).addScaledVector(f.dir, d)
        c.target.copy(f.target)
        c.update()
      }
      return
    }
    orbitInit.current = false

    // fixed view: animate toward the aspect-fitted pose, lock user input
    const f = FRAMINGS[view]
    const d = fitDistance(f, aspect, fovDeg)
    tmpPos.copy(f.target).addScaledVector(f.dir, d)
    camera.position.lerp(tmpPos, 0.12)
    c.target.lerp(f.target, 0.12)
    c.enabled = false
    c.update()
  })

  return (
    <OrbitControls
      ref={controls}
      enablePan={false}
      minDistance={3}
      maxDistance={100}
      maxPolarAngle={Math.PI / 2 - 0.02}
      makeDefault
    />
  )
}
