import { useState } from 'react'
import { useStore } from '../state/store'

const STEPS = [
  { title: 'Drag the setter & ball', body: 'Drag the blue setter and the yellow ball destination on the court to place where the set starts and ends.' },
  { title: 'Shape the arc', body: 'Drag the glowing peak up or down to change the height of the set. It stays centered on the path so the arc always looks realistic.' },
  { title: 'Front vs Side', body: 'Front view edits left/right and height. Side view edits depth off the net and height. Switch views with Shift+1 (front), Shift+2 (side), Shift+3 (orbit).' },
  { title: 'Play it', body: 'Press Space (or the play button) to watch the ball fly the set. Adjust speed for tempo. Load a 9-man preset to start fast.' },
]

export function Coach() {
  const showCoach = useStore((s) => s.showCoach)
  const dismiss = useStore((s) => s.dismissCoach)
  const [i, setI] = useState(0)
  if (!showCoach) return null

  const last = i === STEPS.length - 1
  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="glass w-[380px] p-6">
        <div className="mb-1 font-mono text-[11px] uppercase tracking-wider text-accent">
          Getting started
        </div>
        <h2 className="mb-2 text-xl font-normal">{STEPS[i].title}</h2>
        <p className="mb-5 text-[14px] leading-relaxed text-white/70">{STEPS[i].body}</p>
        <div className="flex items-center justify-between">
          <div className="flex gap-1.5">
            {STEPS.map((_, k) => (
              <span
                key={k}
                className={`h-1.5 w-1.5 rounded-full ${k === i ? 'bg-accent' : 'bg-white/20'}`}
              />
            ))}
          </div>
          <div className="flex gap-2">
            <button onClick={dismiss} className="pill">Skip</button>
            <button
              onClick={() => (last ? dismiss() : setI(i + 1))}
              className="rounded-full bg-accent px-4 py-2 font-mono text-[12px] text-surface-dark shadow-glow"
            >
              {last ? "Let's go" : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
