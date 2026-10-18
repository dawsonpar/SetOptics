import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import { useStore } from '../state/store'
import { curvePoints } from '../lib/arc'
import { BRAND } from '../lib/constants'

export function ArcLine() {
  const set = useStore((s) => s.set)
  const points = useMemo(
    () => curvePoints(set).map((p) => [p.x, p.y, p.z] as [number, number, number]),
    [set],
  )
  return <Line points={points} color={BRAND.arc} lineWidth={3} transparent opacity={0.95} />
}
