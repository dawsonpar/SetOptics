import { Scene } from './three/Scene'
import { TopBar } from './ui/TopBar'
import { ViewSwitch } from './ui/ViewSwitch'
import { ControlSheet } from './ui/ControlSheet'
import { Coach } from './ui/Coach'
import { useKeyboardShortcuts } from './ui/useKeyboardShortcuts'

export function App() {
  useKeyboardShortcuts()
  return (
    <div className="relative h-full w-full overflow-hidden">
      <Scene />

      {/* UI overlay layer: container ignores pointers, controls re-enable them.
          One vertical axis: view switch top center, control sheet + play bar
          bottom center. */}
      <div className="pointer-events-none absolute inset-0 p-4">
        <div className="absolute inset-x-4 top-4">
          <TopBar />
        </div>

        {/* below the brand chip on mobile, in line with the top bar on desktop */}
        <div className="pointer-events-auto absolute left-1/2 top-16 -translate-x-1/2 sm:top-4">
          <ViewSwitch />
        </div>

        <div className="absolute inset-x-0 bottom-3 flex justify-center px-2 sm:bottom-5">
          <div className="pointer-events-auto flex w-full justify-center sm:w-auto">
            <ControlSheet />
          </div>
        </div>
      </div>

      <Coach />
    </div>
  )
}
