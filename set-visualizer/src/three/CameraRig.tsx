import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useStore } from '../state/store'
import type { ViewMode } from '../lib/types'

// Each view defines what to look at (target), from which direction, and how
// large an area must stay in frame (halfW x halfH, world units). The camera
// distance is computed from the live viewport aspect so the whole court fits
// on any screen. A scroll-driven zoom factor then scales that base distance.
type Framing = { target: THREE.Vector3; dir: THREE.Vector3; halfW: number; halfH: number }

const FRAMINGS: Record<ViewMode, Framing> = {
  front: {
    target: new THREE.Vector3(0, 2.2, 3),
    dir: new THREE.Vector3(0, 0.32, 1).normalize(),
    halfW: 9.2,
    halfH: 4.6,
  },
  side: {
    target: new THREE.Vector3(0, 2.2, 2.6),
    dir: new THREE.Vector3(1, 0.32, 0).normalize(),
    halfW: 5.4,
    halfH: 4.6,
  },
  orbit: {
    target: new THREE.Vector3(0, 1.8, 4),
    dir: new THREE.Vector3(0.7, 0.5, 1).normalize(),
    halfW: 9,
    halfH: 5,
  },
}
const PADDING = 1.08
const MIN_ZOOM = 0.45 // closest (zoomed in)
const MAX_ZOOM = 2.4 // farthest (zoomed out)

function fitDistance(f: Framing, aspect: number, fovDeg: number): number {
  const tanHalf = Math.tan((fovDeg * Math.PI) / 180 / 2)
  const distH = f.halfH / tanHalf
  const distW = f.halfW / (tanHalf * Math.max(aspect, 0.0001))
  return Math.max(distH, distW) * PADDING
}

const tmpPos = new THREE.Vector3()

export function CameraRig() {
  const controls = useRef<OrbitControlsImpl>(null)
  const { camera, size, gl } = useThree()
  const view = useStore((s) => s.view)
  const dragging = useStore((s) => s.dragging)
  const orbitInit = useRef(false)
  const zoom = useRef(1)
  const prevView = useRef<ViewMode>(view)

  // Scroll to zoom in the fixed views. Orbit keeps OrbitControls' native zoom.
  useEffect(() => {
    const el = gl.domElement
    const onWheel = (e: WheelEvent) => {
      if (useStore.getState().view === 'orbit') return
      e.preventDefault()
      zoom.current = THREE.MathUtils.clamp(
        zoom.current * Math.exp(e.deltaY * 0.0015),
        MIN_ZOOM,
        MAX_ZOOM,
      )
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [gl])

  useFrame(() => {
    const c = controls.current
    if (!c) return
    if (view !== prevView.current) {
      prevView.current = view
      zoom.current = 1 // reset zoom when switching views
    }
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

    // fixed view: animate toward the aspect-fitted, zoom-scaled pose
    const f = FRAMINGS[view]
    const d = fitDistance(f, aspect, fovDeg) * zoom.current
    tmpPos.copy(f.target).addScaledVector(f.dir, d)
    camera.position.lerp(tmpPos, 0.15)
    c.target.lerp(f.target, 0.15)
    c.enabled = false
    c.update()
  })

  return (
    <OrbitControls
      ref={controls}
      enablePan={false}
      minDistance={2}
      maxDistance={200}
      maxPolarAngle={Math.PI / 2 - 0.02}
      makeDefault
    />
  )
}
