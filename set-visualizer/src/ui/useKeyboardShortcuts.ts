import { useEffect } from 'react'
import { useStore } from '../state/store'

function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  if (!el) return false
  return (
    el.tagName === 'INPUT' ||
    el.tagName === 'TEXTAREA' ||
    el.tagName === 'SELECT' ||
    el.isContentEditable
  )
}

// Shift+1/2/3 switch views (Cmd+number is avoided — it toggles browser tabs).
// Space plays / pauses the set.
export function useKeyboardShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target)) return
      const s = useStore.getState()
      if (e.shiftKey && e.code === 'Digit1') {
        e.preventDefault()
        s.setView('front')
      } else if (e.shiftKey && e.code === 'Digit2') {
        e.preventDefault()
        s.setView('side')
      } else if (e.shiftKey && e.code === 'Digit3') {
        e.preventDefault()
        s.setView('orbit')
      } else if (e.code === 'Space' && !e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault()
        s.togglePlay()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}
