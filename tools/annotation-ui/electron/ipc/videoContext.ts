import fs from 'node:fs/promises'
import path from 'node:path'

export interface VideoContext {
  videoType: 'rally-edit' | 'raw-footage' | 'unknown'
  rallyEditMap: { path: string; segmentCount: number; totalDurationSec: number } | null
  correctedAnnotation: { path: string } | null
  rawAnnotation: { path: string } | null
  error?: string
}

export interface OverrunWarning {
  outOfRangeCount: number
  videoDurationSec: number
  maxEndSec: number
}

export async function fileExists(p: string): Promise<boolean> {
  try {
    await fs.access(p)
    return true
  } catch {
    return false
  }
}

export async function findFirst(paths: string[]): Promise<string | null> {
  for (const p of paths) {
    if (await fileExists(p)) return p
  }
  return null
}

export function correctedAnnotationPath(videoPath: string): string {
  const videoName = path.basename(videoPath, path.extname(videoPath))
  return path.join(path.dirname(videoPath), `${videoName}_annotations_corrected.json`)
}

/** Find the corrected, raw and rally-edit files that belong to a video. */
export async function detectVideoContext(videoPath: string, setOpticsRoot: string): Promise<VideoContext> {
  try {
    const videoName = path.basename(videoPath, path.extname(videoPath))
    const videoDir = path.dirname(videoPath)
    // `data/samples` is the legacy ground-truth dir, still checked for older layouts.
    const rallyGtDir = path.join(setOpticsRoot, 'data', 'rally-gt')
    const legacySamplesDir = path.join(setOpticsRoot, 'data', 'samples')
    const rallyEditsDir = path.join(setOpticsRoot, 'data', 'rally-edits')

    let rallyEditMap = null
    if (videoName.endsWith('_rally_edit')) {
      const mapPath = await findFirst([
        path.join(videoDir, `${videoName}_map.json`),
        path.join(rallyEditsDir, `${videoName}_map.json`),
      ])
      if (mapPath) {
        const data = JSON.parse(await fs.readFile(mapPath, 'utf-8'))
        rallyEditMap = {
          path: mapPath,
          segmentCount: data.segment_count as number,
          totalDurationSec: data.total_concat_duration_sec as number,
        }
      }
    }

    const correctedPath = await findFirst([
      correctedAnnotationPath(videoPath),
      path.join(rallyGtDir, `${videoName}_annotations_corrected.json`),
      path.join(legacySamplesDir, `${videoName}_annotations_corrected.json`),
    ])

    // Colocated sliding-window output first: a stale annotate_fast.py file can belong to another video.
    const rawPath = await findFirst([
      path.join(videoDir, `${videoName}_raw_annotations.json`),
      path.join(rallyGtDir, `${videoName}_raw_annotations.json`),
      path.join(videoDir, `${videoName}_rally_annotations.json`),
      path.join(setOpticsRoot, 'tools', 'annotation', 'annotations', videoName, 'rally', `${videoName}_rally_annotations.json`),
    ])

    const videoType = rallyEditMap
      ? 'rally-edit'
      : correctedPath || rawPath
      ? 'raw-footage'
      : 'unknown'

    return {
      videoType,
      rallyEditMap,
      correctedAnnotation: correctedPath ? { path: correctedPath } : null,
      rawAnnotation: rawPath ? { path: rawPath } : null,
    }
  } catch (error) {
    console.error('Error detecting video context:', error)
    return {
      videoType: 'unknown',
      rallyEditMap: null,
      correctedAnnotation: null,
      rawAnnotation: null,
      error: String(error),
    }
  }
}

/** Segments ending past the video's duration (usually a stale raw file); callers save anyway and warn. */
export function overrunWarning(jsonData: string): OverrunWarning | null {
  try {
    const parsed = JSON.parse(jsonData)
    const durSec = parsed?.video_metadata?.duration_seconds
    const segs: any[] = Array.isArray(parsed?.segments) ? parsed.segments : []
    if (typeof durSec !== 'number' || durSec <= 0 || segs.length === 0) return null
    const limitMs = durSec * 1000
    const overrun = segs.filter((s: any) => typeof s?.end_ms === 'number' && s.end_ms > limitMs + 100)
    if (overrun.length === 0) return null
    const maxEnd = Math.max(...segs.map((s: any) => (typeof s?.end_ms === 'number' ? s.end_ms : 0)))
    console.warn(
      `[save-corrected-annotation] ${overrun.length} segment(s) extend past video duration `
      + `(duration=${durSec}s, maxEnd=${(maxEnd / 1000).toFixed(1)}s). Saving anyway.`
    )
    return {
      outOfRangeCount: overrun.length,
      videoDurationSec: Number(durSec.toFixed(3)),
      maxEndSec: Number((maxEnd / 1000).toFixed(3)),
    }
  } catch (validationErr) {
    // Malformed JSON: let the save still attempt; the read path will complain.
    console.warn('[save-corrected-annotation] validation parse failed:', validationErr)
    return null
  }
}
