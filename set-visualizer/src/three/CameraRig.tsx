import { useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useStore } from '../state/store'
import type { ViewMode } from '../lib/types'

// Camera presets per view. OrbitControls is always the controller; for the
// fixed views we lerp position + target toward the preset and lock input.
const VIEWS: Record<Exclude<ViewMode, 'orbit'>, { pos: THREE.Vector3; target: THREE.Vector3 }> = {
  front: { pos: new THREE.Vector3(0, 3.1, 16), target: new THREE.Vector3(0, 2.3, 1) },
  side: { pos: new THREE.Vector3(17, 3.6, 4.2), target: new THREE.Vector3(0, 2.3, 4.2) },
}
const ORBIT_START = { pos: new THREE.Vector3(12, 8, 16), target: new THREE.Vector3(0, 1.6, 3) }

export function CameraRig() {
  const controls = useRef<OrbitControlsImpl>(null)
  const { camera } = useThree()
  const view = useStore((s) => s.view)
  const dragging = useStore((s) => s.dragging)
  const initialized = useRef(false)

  useFrame(() => {
    const c = controls.current
    if (!c) return

    if (view === 'orbit') {
      c.enabled = !dragging
      return
    }

    // fixed view: animate toward preset, lock user input
    c.enabled = false
    const target = VIEWS[view]
    camera.position.lerp(target.pos, 0.12)
    c.target.lerp(target.target, 0.12)
    c.update()
  })

  // when entering orbit the first time, seed a pleasant angle
  if (view === 'orbit' && !initialized.current && controls.current) {
    initialized.current = true
    camera.position.copy(ORBIT_START.pos)
    controls.current.target.copy(ORBIT_START.target)
  }
  if (view !== 'orbit') initialized.current = false

  return (
    <OrbitControls
      ref={controls}
      enablePan={false}
      minDistance={6}
      maxDistance={40}
      maxPolarAngle={Math.PI / 2 - 0.02}
      makeDefault
    />
  )
}
