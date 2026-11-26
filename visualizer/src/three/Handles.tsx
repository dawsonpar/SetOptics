import { Line } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import { useCoreStore } from '../state/store'
import { apexPoint, minApexHeight } from '../lib/arc'
import { BRAND, CONTACT_HEIGHT } from '../lib/constants'
import type { HandleId } from '../lib/types'

const FRONT_NORMAL = new THREE.Vector3(0, 0, 1)
const SIDE_NORMAL = new THREE.Vector3(1, 0, 0)
const UP_NORMAL = new THREE.Vector3(0, 1, 0)
const plane = new THREE.Plane()
const noRaycast = () => null

// Canonical R3F drag: capture the pointer to the handle so move/up fire even
// over empty space, then intersect the pointer ray with the editing plane.
// Front view edits X + height (plane normal +Z); side view edits depth + height.
// The peak is special: it is locked to the path midpoint and only moves up/down.
// Orbit view is camera-first: only the setter moves there, and only after a
// double-click arms it; the drag then runs on the floor plane (X + depth).
function useDragHandlers(id: HandleId) {
  const view = useCoreStore((s) => s.view)
  const isArmed = useCoreStore((s) => s.orbitArmed === 'setter' && id === 'setter')
  const isEditable = view !== 'orbit' || isArmed

  const onPointerDown = (e: ThreeEvent<PointerEvent>) => {
    if (!isEditable) return
    e.stopPropagation()
    ;(e.target as Element).setPointerCapture(e.pointerId)
    useCoreStore.getState().setDragging(id)
    document.body.style.cursor = 'grabbing'
  }

  const onPointerMove = (e: ThreeEvent<PointerEvent>) => {
    const s = useCoreStore.getState()
    if (s.dragging !== id) return
    e.stopPropagation()
    const handlePoint =
      id === 'setter'
        ? new THREE.Vector3(s.set.setter.x, 0, s.set.setter.z)
        : id === 'dest'
          ? new THREE.Vector3(s.set.dest.x, s.set.dest.y, s.set.dest.z)
          : (() => { const a = apexPoint(s.set); return new THREE.Vector3(a.x, a.y, a.z) })()

    if (s.view === 'orbit') {
      // armed setter: move freely on the floor plane, clamped by the store
      plane.setFromNormalAndCoplanarPoint(UP_NORMAL, handlePoint)
      const hit = new THREE.Vector3()
      if (!e.ray.intersectPlane(plane, hit)) return
      if (id === 'setter') s.setSetter(hit.x, hit.z)
      return
    }

    const isFrontView = s.view === 'front'
    plane.setFromNormalAndCoplanarPoint(isFrontView ? FRONT_NORMAL : SIDE_NORMAL, handlePoint)
    const hit = new THREE.Vector3()
    if (!e.ray.intersectPlane(plane, hit)) return
    if (id === 'setter') {
      isFrontView ? s.setSetter(hit.x, s.set.setter.z) : s.setSetter(s.set.setter.x, hit.z)
    } else if (id === 'dest') {
      isFrontView ? s.setDest({ x: hit.x, y: hit.y, z: s.set.dest.z }) : s.setDest({ x: s.set.dest.x, y: hit.y, z: hit.z })
    } else {
      // peak: midpoint-locked, vertical only, never below the higher endpoint
      const a = apexPoint(s.set)
      s.setPeak({ x: a.x, y: Math.max(hit.y, minApexHeight(s.set)), z: a.z })
    }
  }

  const onPointerUp = (e: ThreeEvent<PointerEvent>) => {
    if (useCoreStore.getState().dragging !== id) return
    e.stopPropagation()
    try {
      ;(e.target as Element).releasePointerCapture(e.pointerId)
    } catch {
      /* capture may already be gone */
    }
    useCoreStore.getState().setDragging(null)
    document.body.style.cursor = 'auto'
  }

  const onPointerOver = () => isEditable && (document.body.style.cursor = 'grab')
  const onPointerOut = () => {
    if (useCoreStore.getState().dragging !== id) document.body.style.cursor = 'auto'
  }

  return { onPointerDown, onPointerMove, onPointerUp, onPointerOver, onPointerOut }
}

// Faint cone arrows hinting which way the setter drags in the active view.
// In orbit they appear only once the setter is armed, showing both axes.
function DragArrows() {
  const view = useCoreStore((s) => s.view)
  const dragging = useCoreStore((s) => s.dragging)
  const isArmed = useCoreStore((s) => s.orbitArmed === 'setter')
  if (dragging) return null
  if (view === 'orbit' && !isArmed) return null
  // front: left/right along X; side: toward/away the net along Z; orbit: both
  const X_DIRS: [number, number, number, [number, number, number]][] = [
    [0.85, 0.16, 0, [0, 0, -Math.PI / 2]],
    [-0.85, 0.16, 0, [0, 0, Math.PI / 2]],
  ]
  const Z_DIRS: [number, number, number, [number, number, number]][] = [
    [0, 0.16, 0.85, [Math.PI / 2, 0, 0]],
    [0, 0.16, -0.85, [-Math.PI / 2, 0, 0]],
  ]
  const dirs =
    view === 'front' ? X_DIRS : view === 'side' ? Z_DIRS : [...X_DIRS, ...Z_DIRS]
  return (
    <>
      {dirs.map(([x, y, z, rot], i) => (
        <mesh key={i} position={[x, y, z]} rotation={rot} raycast={noRaycast}>
          <coneGeometry args={[0.12, 0.3, 12]} />
          <meshBasicMaterial color={BRAND.accent} transparent opacity={0.35} />
        </mesh>
      ))}
    </>
  )
}

// Setter: a capsule body with a floating head, a contact stalk up to the ball
// release height, and faint drag-direction arrows. Double-clicking in orbit
// arms it for repositioning.
export function SetterHandle() {
  const setter = useCoreStore((s) => s.set.setter)
  const dragHandlers = useDragHandlers('setter')
  const armSetter = () => {
    if (useCoreStore.getState().view === 'orbit') useCoreStore.getState().setOrbitArmed('setter')
  }
  return (
    <group position={[setter.x, 0, setter.z]}>
      <mesh position={[0, 0.55, 0]} castShadow onDoubleClick={armSetter} {...dragHandlers}>
        <capsuleGeometry args={[0.22, 0.7, 6, 14]} />
        <meshStandardMaterial color={BRAND.setter} emissive={BRAND.setter} emissiveIntensity={0.25} />
      </mesh>
      <mesh position={[0, 1.5, 0]} castShadow onDoubleClick={armSetter} {...dragHandlers}>
        <sphereGeometry args={[0.2, 24, 24]} />
        <meshStandardMaterial color={BRAND.setter} emissive={BRAND.setter} emissiveIntensity={0.3} />
      </mesh>
      <Line
        points={[[0, 1.78, 0], [0, CONTACT_HEIGHT, 0]]}
        color={BRAND.setter}
        lineWidth={1}
        dashed
        dashSize={0.12}
        gapSize={0.08}
        transparent
        opacity={0.5}
      />
      <DragArrows />
    </group>
  )
}

export function DestHandle() {
  const dest = useCoreStore((s) => s.set.dest)
  const dragHandlers = useDragHandlers('dest')
  return (
    <group>
      <mesh position={[dest.x, dest.y, dest.z]} castShadow {...dragHandlers}>
        <sphereGeometry args={[0.2, 24, 24]} />
        <meshStandardMaterial color={BRAND.ball} emissive={BRAND.ball} emissiveIntensity={0.5} />
      </mesh>
    </group>
  )
}

export function PeakHandle() {
  const set = useCoreStore((s) => s.set)
  const apex = apexPoint(set)
  const dragHandlers = useDragHandlers('peak')
  return (
    <group>
      <mesh position={[apex.x, apex.y, apex.z]} castShadow {...dragHandlers}>
        <sphereGeometry args={[0.096, 24, 24]} />
        <meshStandardMaterial color={BRAND.accent} emissive={BRAND.accent} emissiveIntensity={0.5} />
      </mesh>
    </group>
  )
}
