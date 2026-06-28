/// <reference types="vite/client" />
/// <reference types="../electron/electron-env" />

interface AnnotationProgress {
  currentBatch: number;
  totalBatches: number;
  percentage: number;
}

interface AnnotationComplete {
  success: boolean;
  annotationPath: string;
}

interface AnnotationError {
  code: number;
  error: string;
  lastBatch: number;
  output: string;
}

interface AnnotationOptions {
  startBatch?: number;
  continueOnError?: boolean;
  outputDir?: string;
}

interface Window {
  electronAPI: {
    getAssetBasePath: () => Promise<string | null>
    openExternalUrl: (url: string) => Promise<{ success: boolean; error?: string }>
    saveExportedVideo: (videoData: ArrayBuffer, fileName: string) => Promise<{
      success: boolean
      path?: string
      message?: string
      cancelled?: boolean
    }>
    openVideoFilePicker: () => Promise<{ success: boolean; path?: string; cancelled?: boolean }>
    setCurrentVideoPath: (path: string) => Promise<{ success: boolean }>
    getCurrentVideoPath: () => Promise<{ success: boolean; path?: string }>
    clearCurrentVideoPath: () => Promise<{ success: boolean }>
    getPlatform: () => Promise<string>

    // Annotation methods
    detectVideoContext: (videoPath: string) => Promise<{
      videoType: 'rally-edit' | 'raw-footage' | 'unknown';
      rallyEditMap: { path: string; segmentCount: number; totalDurationSec: number } | null;
      correctedAnnotation: { path: string } | null;
      rawAnnotation: { path: string } | null;
      error?: string;
    }>
    openAnnotationFilePicker: () => Promise<{ success: boolean; path?: string; cancelled?: boolean; error?: string }>
    runAnnotation: (videoPath: string, options?: AnnotationOptions) => Promise<{ success: boolean; message?: string; error?: string }>
    stopAnnotation: () => Promise<{ success: boolean; message?: string }>
    getAnnotationStatus: () => Promise<{ running: boolean; output: string; error: string }>
    readAnnotationFile: (filePath: string) => Promise<{ success: boolean; data?: any; error?: string }>
    saveJsonFile: (defaultPath: string, jsonData: string) => Promise<{ success: boolean; path?: string; cancelled?: boolean; error?: string }>
    saveCorrectedAnnotation: (videoPath: string, jsonData: string) => Promise<{
      success: boolean;
      path?: string;
      cancelled?: boolean;
      error?: string;
      warning?: { outOfRangeCount: number; videoDurationSec: number; maxEndSec: number } | null;
    }>

    // Annotation event listeners
    onAnnotationProgress: (callback: (progress: AnnotationProgress) => void) => () => void
    onAnnotationComplete: (callback: (result: AnnotationComplete) => void) => () => void
    onAnnotationError: (callback: (error: AnnotationError) => void) => () => void
    onAnnotationCancelled: (callback: () => void) => () => void
  }
}
