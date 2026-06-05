import { Scene } from './three/Scene'
import { TopBar } from './ui/TopBar'
import { ViewSwitch } from './ui/ViewSwitch'
import { PlayBar } from './ui/PlayBar'
import { PresetBar } from './ui/PresetBar'
import { CollectionsPanel } from './ui/CollectionsPanel'
import { Coach } from './ui/Coach'
import { useKeyboardShortcuts } from './ui/useKeyboardShortcuts'

export function App() {
  useKeyboardShortcuts()
  return (
    <div className="relative h-full w-full overflow-hidden">
      <Scene />

      {/* UI overlay layer: container ignores pointers, controls re-enable them */}
      <div className="pointer-events-none absolute inset-0 p-4">
        <div className="absolute inset-x-4 top-4">
          <TopBar />
        </div>

        {/* left on mobile (clears the collections toggle), centered on desktop */}
        <div className="pointer-events-auto absolute left-4 top-20 sm:left-1/2 sm:-translate-x-1/2">
          <ViewSwitch />
        </div>

        <div className="pointer-events-auto absolute right-4 top-20">
          <CollectionsPanel />
        </div>

        <div className="absolute inset-x-0 bottom-4 flex flex-col items-center gap-2 px-2 sm:bottom-5 sm:gap-3">
          <div className="pointer-events-auto max-w-full">
            <PresetBar />
          </div>
          <div className="pointer-events-auto max-w-full">
            <PlayBar />
          </div>
        </div>
      </div>

      <Coach />
    </div>
  )
}
