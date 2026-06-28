import { ipcMain, BrowserWindow, shell, app, dialog } from 'electron'
import { spawn, ChildProcess } from 'node:child_process'
import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Path to the annotation script - relative to BUNDLED location in dist-electron/
// After bundling, __dirname = set-optics/tools/annotation-ui/dist-electron/
// So we go up 3 levels to reach set-optics/
const SET_OPTICS_ROOT = path.resolve(__dirname, '..', '..', '..')
const ANNOTATION_SCRIPT_DIR = path.join(SET_OPTICS_ROOT, 'tools', 'annotation')
const ANNOTATION_SCRIPT = path.join(ANNOTATION_SCRIPT_DIR, 'annotate_fast.py')
const PYTHON_PATH = path.join(SET_OPTICS_ROOT, '.venv', 'bin', 'python3')

async function fileExists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

async function findFirst(paths: string[]): Promise<string | null> {
  for (const p of paths) {
    if (await fileExists(p)) return p;
  }
  return null;
}

export function registerIpcHandlers(
  getMainWindow: () => BrowserWindow | null
) {
  // Video path tracking
  let currentVideoPath: string | null = null

  // Annotation process tracking
  let annotationProcess: ChildProcess | null = null
  let annotationOutput: string = ''
  let annotationError: string = ''


  ipcMain.handle('open-external-url', async (_, url: string) => {
    try {
      await shell.openExternal(url)
      return { success: true }
    } catch (error) {
      console.error('Failed to open URL:', error)
      return { success: false, error: String(error) }
    }
  })

  // Return base path for assets so renderer can resolve file:// paths in production
  ipcMain.handle('get-asset-base-path', () => {
    try {
      if (app.isPackaged) {
        return path.join(process.resourcesPath, 'assets')
      }
      return path.join(app.getAppPath(), 'public', 'assets')
    } catch (err) {
      console.error('Failed to resolve asset base path:', err)
      return null
    }
  })

  ipcMain.handle('save-exported-video', async (_, videoData: ArrayBuffer, fileName: string) => {
    try {
      // Determine file type from extension
      const isGif = fileName.toLowerCase().endsWith('.gif');
      const filters = isGif 
        ? [{ name: 'GIF Image', extensions: ['gif'] }]
        : [{ name: 'MP4 Video', extensions: ['mp4'] }];

      const result = await dialog.showSaveDialog({
        title: isGif ? 'Save Exported GIF' : 'Save Exported Video',
        defaultPath: path.join(app.getPath('downloads'), fileName),
        filters,
        properties: ['createDirectory', 'showOverwriteConfirmation']
      });

      if (result.canceled || !result.filePath) {
        return {
          success: false,
          cancelled: true,
          message: 'Export cancelled'
        };
      }

      await fs.writeFile(result.filePath, Buffer.from(videoData));

      return {
        success: true,
        path: result.filePath,
        message: 'Video exported successfully'
      };
    } catch (error) {
      console.error('Failed to save exported video:', error)
      return {
        success: false,
        message: 'Failed to save exported video',
        error: String(error)
      }
    }
  })

  ipcMain.handle('open-video-file-picker', async () => {
    try {
      const result = await dialog.showOpenDialog({
        title: 'Select Video File',
        defaultPath: app.getPath('videos'),
        filters: [
          { name: 'Video Files', extensions: ['webm', 'mp4', 'mov', 'avi', 'mkv'] },
          { name: 'All Files', extensions: ['*'] }
        ],
        properties: ['openFile']
      });

      if (result.canceled || result.filePaths.length === 0) {
        return { success: false, cancelled: true };
      }

      return {
        success: true,
        path: result.filePaths[0]
      };
    } catch (error) {
      console.error('Failed to open file picker:', error);
      return {
        success: false,
        message: 'Failed to open file picker',
        error: String(error)
      };
    }
  });

  ipcMain.handle('set-current-video-path', (_, videoPath: string) => {
    currentVideoPath = videoPath;
    return { success: true };
  });

  ipcMain.handle('get-current-video-path', () => {
    return currentVideoPath ? { success: true, path: currentVideoPath } : { success: false };
  });

  ipcMain.handle('clear-current-video-path', () => {
    currentVideoPath = null;
    return { success: true };
  });

  ipcMain.handle('get-platform', () => {
    return process.platform;
  });

  // Detect what files are available for a video and return full context.
  // Replaces the old check-existing-annotation with richer information
  // so the UI can offer the right actions rather than auto-deciding.
  ipcMain.handle('detect-video-context', async (_, videoPath: string) => {
    try {
      const videoName = path.basename(videoPath, path.extname(videoPath));
      const videoDir = path.dirname(videoPath);
      // Primary GT data dir (post-2026-04-22 rename from `data/samples`).
      const rallyGtDir = path.join(SET_OPTICS_ROOT, 'data', 'rally-gt');
      const legacySamplesDir = path.join(SET_OPTICS_ROOT, 'data', 'samples');
      const rallyEditsDir = path.join(SET_OPTICS_ROOT, 'data', 'rally-edits');

      // Rally edit detection: video name ends with _rally_edit and has a matching map
      let rallyEditMap = null;
      if (videoName.endsWith('_rally_edit')) {
        const mapPath = await findFirst([
          path.join(videoDir, `${videoName}_map.json`),
          path.join(rallyEditsDir, `${videoName}_map.json`),
        ]);
        if (mapPath) {
          const data = JSON.parse(await fs.readFile(mapPath, 'utf-8'));
          rallyEditMap = {
            path: mapPath,
            segmentCount: data.segment_count as number,
            totalDurationSec: data.total_concat_duration_sec as number,
          };
        }
      }

      // Corrected annotation
      const correctedPath = await findFirst([
        path.join(videoDir, `${videoName}_annotations_corrected.json`),
        path.join(rallyGtDir, `${videoName}_annotations_corrected.json`),
        path.join(legacySamplesDir, `${videoName}_annotations_corrected.json`),
      ]);

      // Raw annotation. Priority order: ground-truth pipeline output
      // (`_raw_annotations.json` from annotate_sliding_window.py, written
      // next to the video) comes before the fast-pipeline output
      // (`_rally_annotations.json` from annotate_fast.py). A stale fast
      // output under `tools/annotation/annotations/<name>/rally/` can
      // reference a completely different video that once shared the name
      // — prefer the colocated file so the UI doesn't silently load
      // annotations from some other recording.
      const rawPath = await findFirst([
        path.join(videoDir, `${videoName}_raw_annotations.json`),
        path.join(rallyGtDir, `${videoName}_raw_annotations.json`),
        path.join(videoDir, `${videoName}_rally_annotations.json`),
        path.join(SET_OPTICS_ROOT, 'tools', 'annotation', 'annotations', videoName, 'rally', `${videoName}_rally_annotations.json`),
      ]);

      const videoType = rallyEditMap
        ? 'rally-edit'
        : correctedPath || rawPath
        ? 'raw-footage'
        : 'unknown';

      return {
        videoType,
        rallyEditMap,
        correctedAnnotation: correctedPath ? { path: correctedPath } : null,
        rawAnnotation: rawPath ? { path: rawPath } : null,
      };
    } catch (error) {
      console.error('Error detecting video context:', error);
      return {
        videoType: 'unknown',
        rallyEditMap: null,
        correctedAnnotation: null,
        rawAnnotation: null,
        error: String(error),
      };
    }
  });

  // Open a file picker scoped to JSON annotation files
  ipcMain.handle('open-annotation-file-picker', async () => {
    try {
      const result = await dialog.showOpenDialog({
        title: 'Load Annotation File',
        defaultPath: path.join(SET_OPTICS_ROOT, 'data', 'samples'),
        filters: [
          { name: 'JSON Annotation Files', extensions: ['json'] },
          { name: 'All Files', extensions: ['*'] },
        ],
        properties: ['openFile'],
      });
      if (result.canceled || result.filePaths.length === 0) {
        return { success: false, cancelled: true };
      }
      return { success: true, path: result.filePaths[0] };
    } catch (error) {
      return { success: false, error: String(error) };
    }
  });

  // Run the annotation script
  ipcMain.handle('run-annotation', async (_event, videoPath: string, options?: {
    startBatch?: number;
    continueOnError?: boolean;
    outputDir?: string;
  }) => {
    try {
      // Kill any existing annotation process
      if (annotationProcess) {
        annotationProcess.kill();
        annotationProcess = null;
      }

      // Reset output tracking
      annotationOutput = '';
      annotationError = '';

      const mainWindow = getMainWindow();
      const videoName = path.basename(videoPath, path.extname(videoPath));

      // The AI draft needs the Python env. If it isn't set up, fail with a clear,
      // actionable message instead of a raw spawn ENOENT. Manual annotation
      // ("Start from scratch") needs none of this.
      if (!(await fileExists(PYTHON_PATH)) || !(await fileExists(ANNOTATION_SCRIPT))) {
        return {
          success: false,
          error: 'AI draft needs the Python environment. Run ./setup.sh and set GEMINI_API_KEY in .env, or use "Start from scratch" to annotate manually.',
        };
      }

      // Build command arguments — annotate_fast.py expects a bare name, not a path
      const args = [ANNOTATION_SCRIPT, '--video', videoName];

      if (options?.startBatch && options.startBatch > 1) {
        args.push('--start-batch', String(options.startBatch));
      }
      if (options?.continueOnError) {
        args.push('--continue-on-error');
      }
      if (options?.outputDir) {
        args.push('--output-dir', options.outputDir);
      }

      // Spawn the Python process using the virtual environment
      annotationProcess = spawn(PYTHON_PATH, args, {
        cwd: ANNOTATION_SCRIPT_DIR,
        env: { ...process.env },
      });

      let totalBatches = 0;
      let currentBatch = 0;

      annotationProcess.stdout?.on('data', (data: Buffer) => {
        const text = data.toString();
        annotationOutput += text;

        // Parse progress from output
        // Look for "Created X batches" to get total
        const totalMatch = text.match(/Created (\d+) batches/);
        if (totalMatch) {
          totalBatches = parseInt(totalMatch[1], 10);
        }

        // Look for "Processing batch X/Y..."
        const progressMatch = text.match(/Processing batch (\d+)\/(\d+)/);
        if (progressMatch) {
          currentBatch = parseInt(progressMatch[1], 10);
          totalBatches = parseInt(progressMatch[2], 10);

          // Send progress update to renderer
          mainWindow?.webContents.send('annotation-progress', {
            currentBatch,
            totalBatches,
            percentage: Math.round((currentBatch / totalBatches) * 100),
          });
        }

        // Look for completion message
        if (text.includes('Saved annotations to:')) {
          const pathMatch = text.match(/Saved annotations to: (.+)/);
          if (pathMatch) {
            mainWindow?.webContents.send('annotation-complete', {
              success: true,
              annotationPath: pathMatch[1].trim(),
            });
          }
        }
      });

      annotationProcess.stderr?.on('data', (data: Buffer) => {
        const text = data.toString();
        annotationError += text;
        console.error('Annotation stderr:', text);
      });

      annotationProcess.on('close', (code) => {
        const wasKilled = annotationProcess?.killed;
        annotationProcess = null;

        if (wasKilled) {
          mainWindow?.webContents.send('annotation-cancelled');
          return;
        }

        if (code === 0) {
          // Success - find the annotation file
          const outputDir = options?.outputDir || path.join(SET_OPTICS_ROOT, 'tools', 'annotation', 'annotations');
          const annotationPath = path.join(outputDir, videoName, 'rally', `${videoName}_rally_annotations.json`);

          mainWindow?.webContents.send('annotation-complete', {
            success: true,
            annotationPath,
          });
        } else {
          // Extract last batch number for resume capability
          const lastBatchMatch = annotationOutput.match(/Processing batch (\d+)\/\d+/g);
          const lastBatch = lastBatchMatch
            ? parseInt(lastBatchMatch[lastBatchMatch.length - 1].match(/(\d+)\//)![1], 10)
            : 0;

          mainWindow?.webContents.send('annotation-error', {
            code,
            error: annotationError || 'Annotation failed',
            lastBatch,
            output: annotationOutput,
          });
        }
      });

      annotationProcess.on('error', (error) => {
        console.error('Failed to start annotation process:', error);
        mainWindow?.webContents.send('annotation-error', {
          code: -1,
          error: `Failed to start annotation: ${error.message}`,
          lastBatch: 0,
          output: '',
        });
        annotationProcess = null;
      });

      return { success: true, message: 'Annotation started' };
    } catch (error) {
      console.error('Error starting annotation:', error);
      return { success: false, error: String(error) };
    }
  });

  // Stop the running annotation
  ipcMain.handle('stop-annotation', () => {
    if (annotationProcess) {
      annotationProcess.kill();
      annotationProcess = null;
      return { success: true, message: 'Annotation stopped' };
    }
    return { success: false, message: 'No annotation running' };
  });

  // Get annotation process status
  ipcMain.handle('get-annotation-status', () => {
    return {
      running: annotationProcess !== null,
      output: annotationOutput,
      error: annotationError,
    };
  });

  // Read annotation JSON file
  ipcMain.handle('read-annotation-file', async (_, filePath: string) => {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      return { success: true, data: JSON.parse(content) };
    } catch (error) {
      console.error('Error reading annotation file:', error);
      return { success: false, error: String(error) };
    }
  });

  // Save any JSON file — caller provides the full default path
  ipcMain.handle('save-json-file', async (_, defaultPath: string, jsonData: string) => {
    try {
      const result = await dialog.showSaveDialog({
        title: 'Save File',
        defaultPath,
        filters: [
          { name: 'JSON Files', extensions: ['json'] },
          { name: 'All Files', extensions: ['*'] }
        ],
        properties: ['createDirectory', 'showOverwriteConfirmation']
      });

      if (result.canceled || !result.filePath) {
        return { success: false, cancelled: true };
      }

      await fs.writeFile(result.filePath, jsonData, 'utf-8');
      return { success: true, path: result.filePath };
    } catch (error) {
      console.error('Error saving JSON file:', error);
      return { success: false, error: String(error) };
    }
  });

  // Save corrected annotation JSON file
  ipcMain.handle('save-corrected-annotation', async (_, videoPath: string, jsonData: string) => {
    try {
      const videoDir = path.dirname(videoPath);
      const videoName = path.basename(videoPath, path.extname(videoPath));
      const defaultFileName = `${videoName}_annotations_corrected.json`;

      // Pre-save validation. The 2026-04-22 incident showed that a
      // stale raw file from a different video could leave phantom
      // segments past the real video's end in the corrected export —
      // and nobody noticed until F1 was computed. Detect the
      // mismatch and return a warning so the UI can surface it.
      // We do NOT silently drop the segments here; the frontend
      // decides whether to prompt the user or strip them.
      let overrunWarning: { outOfRangeCount: number; videoDurationSec: number; maxEndSec: number } | null = null;
      try {
        const parsed = JSON.parse(jsonData);
        const durSec = parsed?.video_metadata?.duration_seconds;
        const segs: any[] = Array.isArray(parsed?.segments) ? parsed.segments : [];
        if (typeof durSec === 'number' && durSec > 0 && segs.length > 0) {
          const limitMs = durSec * 1000;
          const overrun = segs.filter((s: any) => typeof s?.end_ms === 'number' && s.end_ms > limitMs + 100);
          if (overrun.length > 0) {
            const maxEnd = Math.max(...segs.map((s: any) => (typeof s?.end_ms === 'number' ? s.end_ms : 0)));
            overrunWarning = {
              outOfRangeCount: overrun.length,
              videoDurationSec: Number(durSec.toFixed(3)),
              maxEndSec: Number((maxEnd / 1000).toFixed(3)),
            };
            console.warn(
              `[save-corrected-annotation] ${overrun.length} segment(s) extend past video duration `
              + `(duration=${durSec}s, maxEnd=${(maxEnd / 1000).toFixed(1)}s). Saving anyway — `
              + `frontend should surface the mismatch to the user.`
            );
          }
        }
      } catch (validationErr) {
        // Malformed JSON — let the save still attempt; the read path will complain.
        console.warn('[save-corrected-annotation] validation parse failed:', validationErr);
      }

      const result = await dialog.showSaveDialog({
        title: 'Save Corrected Annotations',
        defaultPath: path.join(videoDir, defaultFileName),
        filters: [
          { name: 'JSON Files', extensions: ['json'] },
          { name: 'All Files', extensions: ['*'] }
        ],
        properties: ['createDirectory', 'showOverwriteConfirmation']
      });

      if (result.canceled || !result.filePath) {
        return { success: false, cancelled: true };
      }

      await fs.writeFile(result.filePath, jsonData, 'utf-8');

      return {
        success: true,
        path: result.filePath,
        warning: overrunWarning,
      };
    } catch (error) {
      console.error('Error saving corrected annotation:', error);
      return { success: false, error: String(error) };
    }
  });
}
