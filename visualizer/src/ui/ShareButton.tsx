import { useState } from 'react'
import { useStore } from '../state/store'
import { shareUrlFor } from '../lib/url'

export function ShareButton() {
  const set = useStore((s) => s.set)
  const format = useStore((s) => s.format)
  const [copied, setCopied] = useState(false)

  const onShare = async () => {
    const url = shareUrlFor(set, format)
    try {
      await navigator.clipboard.writeText(url)
    } catch {
      window.prompt('Copy this link to share the set:', url)
    }
    setCopied(true)
    window.history.replaceState(null, '', url)
    setTimeout(() => setCopied(false), 1600)
  }

  return (
    <button onClick={onShare} className="pill">
      {copied ? 'Link copied' : 'Share set'}
    </button>
  )
}
