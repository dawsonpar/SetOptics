import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('electronAPI', {
  getAssetBasePath: async () => {
    return await ipcRenderer.invoke('get-asset-base-path')
  },
  openExternalUrl: (url: string) => {
    return ipcRenderer.invoke('open-external-url', url)
  },
  saveExportedVideo: (videoData: ArrayBuffer, fileName: string) => {
    return ipcRenderer.invoke('save-exported-video', videoData, fileName)
  },
  openVideoFilePicker: () => {
    return ipcRenderer.invoke('open-video-file-picker')
  },
  setCurrentVideoPath: (path: string) => {
    return ipcRenderer.invoke('set-current-video-path', path)
  },
  getCurrentVideoPath: () => {
    return ipcRenderer.invoke('get-current-video-path')
  },
  clearCurrentVideoPath: () => {
    return ipcRenderer.invoke('clear-current-video-path')
  },
  getPlatform: () => {
    return ipcRenderer.invoke('get-platform')
  },

  // Annotation methods
  detectVideoContext: (videoPath: string) => {
    return ipcRenderer.invoke('detect-video-context', videoPath)
  },
  openAnnotationFilePicker: () => {
    return ipcRenderer.invoke('open-annotation-file-picker')
  },
  runAnnotation: (videoPath: string, options?: {
    startBatch?: number;
    continueOnError?: boolean;
    outputDir?: string;
  }) => {
    return ipcRenderer.invoke('run-annotation', videoPath, options)
  },
  stopAnnotation: () => {
    return ipcRenderer.invoke('stop-annotation')
  },
  getAnnotationStatus: () => {
    return ipcRenderer.invoke('get-annotation-status')
  },
  readAnnotationFile: (filePath: string) => {
    return ipcRenderer.invoke('read-annotation-file', filePath)
  },
  saveJsonFile: (defaultPath: string, jsonData: string) => {
    return ipcRenderer.invoke('save-json-file', defaultPath, jsonData)
  },
  saveCorrectedAnnotation: (videoPath: string, jsonData: string) => {
    return ipcRenderer.invoke('save-corrected-annotation', videoPath, jsonData)
  },

  // Annotation event listeners
  onAnnotationProgress: (callback: (progress: { currentBatch: number; totalBatches: number; percentage: number }) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, progress: { currentBatch: number; totalBatches: number; percentage: number }) => callback(progress)
    ipcRenderer.on('annotation-progress', listener)
    return () => ipcRenderer.removeListener('annotation-progress', listener)
  },
  onAnnotationComplete: (callback: (result: { success: boolean; annotationPath: string }) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, result: { success: boolean; annotationPath: string }) => callback(result)
    ipcRenderer.on('annotation-complete', listener)
    return () => ipcRenderer.removeListener('annotation-complete', listener)
  },
  onAnnotationError: (callback: (error: { code: number; error: string; lastBatch: number; output: string }) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, error: { code: number; error: string; lastBatch: number; output: string }) => callback(error)
    ipcRenderer.on('annotation-error', listener)
    return () => ipcRenderer.removeListener('annotation-error', listener)
  },
  onAnnotationCancelled: (callback: () => void) => {
    const listener = () => callback()
    ipcRenderer.on('annotation-cancelled', listener)
    return () => ipcRenderer.removeListener('annotation-cancelled', listener)
  },
})