type ElectronApi = Window['electronAPI']

const CURRENT_VIDEO_KEY = 'web-annotation:current-video'
const UNSUPPORTED = 'Not available in browser mode. Use the Electron app.'

async function post<T>(route: string, body: unknown): Promise<T> {
  const res = await fetch(`/web-api/${route}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (res.status === 401) throw new Error('Session expired. Reopen the link with ?key=<token>.')
  return res.json().catch(() => ({ success: false, error: `HTTP ${res.status}` })) as Promise<T>
}

type VideoContext = Awaited<ReturnType<ElectronApi['detectVideoContext']>>

async function detectContext(videoPath: string): Promise<VideoContext> {
  const context = await post<Partial<VideoContext>>('detect-context', { videoPath })
  if (context.videoType) return context as VideoContext
  return { videoType: 'unknown', rallyEditMap: null, correctedAnnotation: null, rawAnnotation: null, error: context.error }
}

function storage(): Storage | null {
  try {
    return window.localStorage
  } catch {
    return null
  }
}

function pickVideo(): Promise<string | null> {
  return new Promise((resolve) => {
    const overlay = document.createElement('div')
    overlay.style.cssText = 'position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;padding:16px'
    const panel = document.createElement('div')
    panel.style.cssText = 'background:#16161a;color:#eee;border-radius:12px;width:min(560px,100%);max-height:80dvh;overflow:auto;padding:16px;font:14px system-ui'
    panel.textContent = 'Loading videos...'
    overlay.appendChild(panel)
    const close = (value: string | null) => {
      overlay.remove()
      resolve(value)
    }
    overlay.addEventListener('click', (e) => e.target === overlay && close(null))
    document.body.appendChild(overlay)

    fetch('/web-api/videos')
      .then((r) => r.json())
      .then(({ videos }: { videos: { path: string; name: string; sizeMb: number }[] }) => {
        panel.textContent = ''
        const title = document.createElement('div')
        title.textContent = videos.length ? 'Choose a video' : 'No videos found'
        title.style.cssText = 'margin-bottom:12px;opacity:.7'
        panel.appendChild(title)
        for (const video of videos) {
          const row = document.createElement('button')
          row.textContent = `${video.name}  ·  ${video.sizeMb} MB`
          row.title = video.path
          row.style.cssText = 'display:block;width:100%;text-align:left;padding:10px 12px;margin:4px 0;border-radius:8px;border:1px solid #333;background:#1f1f24;color:inherit;cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'
          row.onclick = () => close(video.path)
          panel.appendChild(row)
        }
      })
      .catch((err) => {
        panel.textContent = `Failed to list videos: ${err}`
      })
  })
}

/** Installs a browser-backed window.electronAPI that talks to the Vite web API (web/fileApi.ts). */
export function installWebElectronApi() {
  if (window.electronAPI) return

  const noopUnsubscribe = () => () => {}

  // Recording methods are omitted: the annotation editor never calls them.
  const api: Partial<ElectronApi> = {
    openExternalUrl: async (url) => {
      window.open(url, '_blank', 'noopener')
      return { success: true }
    },
    saveExportedVideo: async () => ({ success: false, message: UNSUPPORTED }),
    openVideoFilePicker: async () => {
      const picked = await pickVideo()
      if (!picked) return { success: false, cancelled: true }
      storage()?.setItem(CURRENT_VIDEO_KEY, picked)
      return { success: true, path: picked }
    },
    setCurrentVideoPath: async (p) => {
      storage()?.setItem(CURRENT_VIDEO_KEY, p)
      return { success: true }
    },
    getCurrentVideoPath: async () => {
      const p = storage()?.getItem(CURRENT_VIDEO_KEY)
      return p ? { success: true, path: p } : { success: false }
    },
    clearCurrentVideoPath: async () => {
      storage()?.removeItem(CURRENT_VIDEO_KEY)
      return { success: true }
    },
    getPlatform: async () => (navigator.platform.toLowerCase().includes('mac') ? 'darwin' : 'web'),
    detectVideoContext: detectContext,
    openAnnotationFilePicker: async () => ({ success: false, error: UNSUPPORTED }),
    runAnnotation: async () => ({ success: false, error: UNSUPPORTED }),
    stopAnnotation: async () => ({ success: false, message: UNSUPPORTED }),
    getAnnotationStatus: async () => ({ running: false, output: '', error: '' }),
    readAnnotationFile: (filePath) => post('read', { filePath }),
    saveJsonFile: async (defaultPath, jsonData) => {
      const blob = new Blob([jsonData], { type: 'application/json' })
      const link = document.createElement('a')
      link.href = URL.createObjectURL(blob)
      link.download = defaultPath.split('/').pop() || 'annotations.json'
      link.click()
      setTimeout(() => URL.revokeObjectURL(link.href), 0)
      return { success: true, path: link.download }
    },
    saveCorrectedAnnotation: (videoPath, jsonData) => post('save-corrected', { videoPath, jsonData }),
    onAnnotationProgress: noopUnsubscribe,
    onAnnotationComplete: noopUnsubscribe,
    onAnnotationError: noopUnsubscribe,
    onAnnotationCancelled: noopUnsubscribe,
    mediaUrl: (p: string) => `/web-api/media?path=${encodeURIComponent(p)}`,
  }

  window.electronAPI = api as ElectronApi
}
