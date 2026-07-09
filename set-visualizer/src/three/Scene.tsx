import { Canvas } from '@react-three/fiber'
import { ContactShadows } from '@react-three/drei'
import { BRAND, STAGE_HALF_DEPTH } from '../lib/constants'
import { Court } from './Court'
import { Net } from './Net'
import { ArcLine } from './ArcLine'
import { Ball } from './Ball'
import { SetterHandle, DestHandle, PeakHandle } from './Handles'
import { CameraRig } from './CameraRig'

export function Scene() {
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      camera={{ position: [0, 3.1, 16], fov: 42, near: 0.1, far: 120 }}
      gl={{ antialias: true }}
    >
      <color attach="background" args={[BRAND.bg]} />
      <fog attach="fog" args={[BRAND.bg, 26, 70]} />

      <ambientLight intensity={0.55} />
      <hemisphereLight args={['#9ec6e8', '#0a0a0f', 0.5]} />
      <directionalLight
        position={[7, 13, 9]}
        intensity={1.15}
        castShadow
        shadow-mapSize={[1024, 1024]}
        shadow-camera-left={-12}
        shadow-camera-right={12}
        shadow-camera-top={12}
        shadow-camera-bottom={-12}
      />

      <Court />
      <Net />
      <ArcLine />
      <Ball />
      <SetterHandle />
      <DestHandle />
      <PeakHandle />

      <ContactShadows
        position={[0, 0.02, STAGE_HALF_DEPTH / 2]}
        scale={26}
        resolution={1024}
        blur={2.4}
        opacity={0.35}
        far={8}
        color="#000000"
      />

      <CameraRig />
    </Canvas>
  )
}
