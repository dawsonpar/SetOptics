import { useState } from 'react'
import { useCoreStore } from '../state/store'
import { shareUrlFor } from '../lib/url'

export function ShareButton() {
  const set = useCoreStore((s) => s.set)
  const format = useCoreStore((s) => s.format)
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
