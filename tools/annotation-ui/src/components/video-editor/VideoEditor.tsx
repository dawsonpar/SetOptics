import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Button } from "@/components/ui/button";
import { Upload, FileVideo, ChevronUp, ChevronDown, Download, Crosshair, Trash2 } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import VideoPlayback, { VideoPlaybackRef } from "./VideoPlayback";
import PlaybackControls from "./PlaybackControls";
import { ProcessingOverlay } from "./ProcessingOverlay";
import { AnnotationErrorDialog } from "./AnnotationErrorDialog";
import { SegmentTimeline, SegmentTimelineRef } from "./SegmentTimeline";
import { VideoContextPanel } from "./VideoContextPanel";
import {
  SegmentRegion, SegmentType, AnnotationJSON, loadAnnotationSegments,
  loadSegmentsFromChapterMap, ChapterMapJSON,
  VideoContext, SetMoment,
} from "./types";

// Helper to format time with milliseconds
function formatTimeWithMs(ms: number): string {
  const totalSeconds = ms / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const milliseconds = Math.floor((ms % 1000) / 10);
  return `${minutes}:${seconds.toString().padStart(2, "0")}.${milliseconds.toString().padStart(2, "0")}`;
}

// Source label shown in the status section after loading
type AnnotationSource = 'corrected' | 'raw' | 'rally-edit' | 'file';

type LoadingState =
  | { status: 'idle' }
  | { status: 'detecting' }
  | { status: 'ready'; context: VideoContext }
  | { status: 'loading' }
  | { status: 'running-detection'; currentBatch: number; totalBatches: number; percentage: number }
  | { status: 'loaded'; source: AnnotationSource }
  | { status: 'detection-error'; error: string; lastBatch: number; output: string };

export default function VideoEditor() {
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [rawVideoPath, setRawVideoPath] = useState<string | null>(null); // Original file path (not file:// URL)
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Loading state
  const [loadingState, setLoadingState] = useState<LoadingState>({ status: 'idle' });
  const [showErrorDialog, setShowErrorDialog] = useState(false);

  // Segment state
  const [segments, setSegments] = useState<SegmentRegion[]>([]);
  const blankSeedPendingRef = useRef(false);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [annotationDurationMs, setAnnotationDurationMs] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isReversing, _setIsReversing] = useState(false);
  const isReversingRef = useRef(false);
  const setIsReversing = useCallback((val: boolean | ((prev: boolean) => boolean)) => {
    _setIsReversing(prev => {
      const next = typeof val === 'function' ? val(prev) : val;
      isReversingRef.current = next;
      return next;
    });
  }, []);

  // Set moment state
  const [setMoments, setSetMoments] = useState<SetMoment[]>([]);
  const [selectedSetMomentId, setSelectedSetMomentId] = useState<string | null>(null);

  const [draftSavedAt, setDraftSavedAt] = useState<Date | null>(null);

  const videoPlaybackRef = useRef<VideoPlaybackRef>(null);
  const segmentTimelineRef = useRef<SegmentTimelineRef>(null);

  // Ref so loadSegments can read rawVideoPath without stale closure
  const rawVideoPathRef = useRef<string | null>(null);
  useEffect(() => {
    rawVideoPathRef.current = rawVideoPath;
  }, [rawVideoPath]);

  // Draft auto-save to localStorage
  const DRAFT_PREFIX = 'annotation-draft:';

  useEffect(() => {
    if (!rawVideoPath || segments.length === 0) return;
    const key = `${DRAFT_PREFIX}${rawVideoPath}`;
    const timer = setTimeout(() => {
      localStorage.setItem(key, JSON.stringify({
        savedAt: new Date().toISOString(),
        segments,
      }));
      setDraftSavedAt(new Date());
    }, 1000);
    return () => clearTimeout(timer);
  }, [segments, rawVideoPath]);

  // Helper to convert file path to proper file:// URL
  const toFileUrl = (filePath: string): string => {
    const normalized = filePath.replace(/\\/g, '/');
    if (normalized.match(/^[a-zA-Z]:/)) {
      return `file:///${normalized}`;
    }
    return `file://${normalized}`;
  };

  // Detection process IPC event listeners
  useEffect(() => {
    const unsubProgress = window.electronAPI.onAnnotationProgress((progress) => {
      setLoadingState({
        status: 'running-detection',
        currentBatch: progress.currentBatch,
        totalBatches: progress.totalBatches,
        percentage: progress.percentage,
      });
    });

    const unsubComplete = window.electronAPI.onAnnotationComplete((result) => {
      loadAnnotationFromFile(result.annotationPath, 'raw');
      toast.success('Detection complete');
    });

    const unsubError = window.electronAPI.onAnnotationError((err) => {
      setLoadingState({
        status: 'detection-error',
        error: err.error,
        lastBatch: err.lastBatch,
        output: err.output,
      });
      setShowErrorDialog(true);
    });

    const unsubCancelled = window.electronAPI.onAnnotationCancelled(() => {
      setLoadingState({ status: 'idle' });
      toast.info('Detection cancelled');
    });

    return () => {
      unsubProgress();
      unsubComplete();
      unsubError();
      unsubCancelled();
    };
  }, []);

  // Apply loaded segments to state and check for a saved draft
  const applySegments = useCallback((
    loadedSegments: SegmentRegion[],
    durationMs: number,
    source: AnnotationSource,
  ) => {
    setSegments(loadedSegments);
    setAnnotationDurationMs(durationMs);
    if (loadedSegments.length > 0) {
      setSelectedSegmentId(loadedSegments[0].id);
    }
    setLoadingState({ status: 'loaded', source });

    const filePath = rawVideoPathRef.current;
    if (filePath) {
      const saved = localStorage.getItem(`annotation-draft:${filePath}`);
      if (saved) {
        const draft = JSON.parse(saved) as { savedAt: string; segments: SegmentRegion[] };
        const savedAt = new Date(draft.savedAt);
        toast(`Draft found from ${savedAt.toLocaleTimeString()}`, {
          description: `${draft.segments.length} segments — restore to continue editing`,
          action: {
            label: 'Restore',
            onClick: () => {
              setSegments(draft.segments);
              setSelectedSegmentId(draft.segments[0]?.id ?? null);
              setDraftSavedAt(savedAt);
              toast.success('Draft restored');
            },
          },
          duration: 15000,
        });
        return;
      }
    }
    toast.success(`Loaded ${loadedSegments.length} segments`);
  }, []);

  // Load segments from a JSON annotation file (corrected, raw, or manually chosen)
  const loadAnnotationFromFile = useCallback(async (
    filePath: string,
    source: AnnotationSource,
  ) => {
    setLoadingState({ status: 'loading' });
    try {
      const result = await window.electronAPI.readAnnotationFile(filePath);
      if (result.success && result.data) {
        const { segments: loaded, durationMs } = loadAnnotationSegments(result.data as AnnotationJSON);
        applySegments(loaded, durationMs, source);
      } else {
        throw new Error(result.error || 'Failed to read file');
      }
    } catch (err) {
      console.error('Error loading annotation:', err);
      toast.error('Failed to load annotation');
      setLoadingState({ status: 'idle' });
    }
  }, [applySegments]);

  // Load segments from a rally edit chapter map
  const loadAnnotationFromChapterMap = useCallback(async (mapPath: string) => {
    setLoadingState({ status: 'loading' });
    try {
      const result = await window.electronAPI.readAnnotationFile(mapPath);
      if (result.success && result.data) {
        const { segments: loaded, durationMs } = loadSegmentsFromChapterMap(result.data as ChapterMapJSON);
        applySegments(loaded, durationMs, 'rally-edit');
      } else {
        throw new Error(result.error || 'Failed to read chapter map');
      }
    } catch (err) {
      console.error('Error loading chapter map:', err);
      toast.error('Failed to load chapter map');
      setLoadingState({ status: 'idle' });
    }
  }, [applySegments]);

  // Start a blank timeline (no Python / no API key needed). The video duration
  // is not known until the editor mounts and the <video> reports it, so we mark
  // a pending seed and fill in a single full-length break once duration arrives.
  const handleStartBlank = useCallback(() => {
    blankSeedPendingRef.current = true;
    setSegments([]);
    setSelectedSegmentId(null);
    setAnnotationDurationMs(0);
    setLoadingState({ status: 'loaded', source: 'raw' });
    toast.success('Blank timeline started — press B to split the break, 1/2 to label');
  }, []);

  // Seed the blank timeline with one full-length break once the video reports
  // its duration. Split (B) it and toggle types (1 = in-play, 2 = break) to mark
  // rallies, then export.
  useEffect(() => {
    if (
      blankSeedPendingRef.current &&
      loadingState.status === 'loaded' &&
      duration > 0
    ) {
      const durMs = Math.round(duration * 1000);
      const seed: SegmentRegion = { id: 'segment-1', startMs: 0, endMs: durMs, type: 'break' };
      setSegments([seed]);
      setSelectedSegmentId('segment-1');
      setAnnotationDurationMs(durMs);
      blankSeedPendingRef.current = false;
    }
  }, [duration, loadingState.status]);

  // Get selected segment
  const selectedSegment = segments.find(s => s.id === selectedSegmentId) || null;

  // Handle segment boundary change
  const handleSegmentBoundaryChange = useCallback((segmentId: string, newStartMs: number, newEndMs: number) => {
    setSegments(prev => prev.map(s =>
      s.id === segmentId ? { ...s, startMs: newStartMs, endMs: newEndMs } : s
    ));
  }, []);

  // Mark a set moment — deduplicates within ±200ms
  const handleAddSetMoment = useCallback((timeMs: number) => {
    const MIN_GAP_MS = 200;
    const alreadyExists = setMoments.some(
      m => Math.abs(m.timeMs - timeMs) < MIN_GAP_MS,
    );
    if (alreadyExists) {
      toast.error('A set moment already exists within 200ms of this position');
      return;
    }
    const id = `set-${Date.now()}`;
    setSetMoments(prev => [...prev, { id, timeMs }]);
    setSelectedSetMomentId(id);
    toast.success(`Set marked at ${formatTimeWithMs(timeMs)}`);
  }, [setMoments]);

  const handleDeleteSetMoment = useCallback((id: string) => {
    setSetMoments(prev => prev.filter(m => m.id !== id));
    setSelectedSetMomentId(prev => prev === id ? null : prev);
  }, []);

  // Export set annotations as *_set_annotations.json
  const handleExportSetAnnotations = useCallback(async () => {
    if (!rawVideoPath || setMoments.length === 0) {
      toast.error('No set moments to export');
      return;
    }

    const sorted = [...setMoments].sort((a, b) => a.timeMs - b.timeMs);
    const videoFilename = rawVideoPath.split('/').pop() || rawVideoPath.split('\\').pop() || '';

    const json = {
      video: videoFilename,
      annotated_at: new Date().toISOString().split('T')[0],
      annotator: 'human',
      sets: sorted.map(m => {
        const momentSec = m.timeMs / 1000;
        // Try to scope clip window to the containing rally
        const containingRally = segments.find(
          s => s.type === 'in-play' && m.timeMs >= s.startMs && m.timeMs <= s.endMs,
        );
        const clipStart = containingRally
          ? Math.max(containingRally.startMs / 1000, momentSec - 4)
          : Math.max(0, momentSec - 4);
        const clipEnd = containingRally
          ? Math.min(containingRally.endMs / 1000, momentSec + 4)
          : momentSec + 4;
        return {
          set_moment_seconds: Math.round(momentSec * 1000) / 1000,
          clip_start_seconds: Math.round(clipStart * 1000) / 1000,
          clip_end_seconds: Math.round(clipEnd * 1000) / 1000,
        };
      }),
    };

    const jsonString = JSON.stringify(json, null, 2);

    try {
      const defaultPath = rawVideoPath.replace(/(\.[^.]+)$/, '_set_annotations.json');
      const result = await window.electronAPI.saveJsonFile(defaultPath, jsonString);
      if (result.cancelled) return;
      if (result.success && result.path) {
        toast.success(`Set annotations saved to ${result.path}`);
      } else {
        toast.error(result.error || 'Failed to save set annotations');
      }
    } catch (err) {
      console.error('Error saving set annotations:', err);
      toast.error('Failed to save set annotations');
    }
  }, [rawVideoPath, setMoments, segments]);

  // Handle segment type change
  const handleSegmentTypeChange = useCallback((type: SegmentType) => {
    if (!selectedSegmentId) return;
    setSegments(prev => prev.map(s =>
      s.id === selectedSegmentId ? { ...s, type } : s
    ));
  }, [selectedSegmentId]);

  // Navigate to previous segment
  const goToPrevSegment = useCallback(() => {
    if (segments.length === 0) return;
    const sortedSegments = [...segments].sort((a, b) => a.startMs - b.startMs);
    const currentIndex = selectedSegmentId
      ? sortedSegments.findIndex(s => s.id === selectedSegmentId)
      : -1;
    const newIndex = currentIndex > 0 ? currentIndex - 1 : sortedSegments.length - 1;
    const newSegment = sortedSegments[newIndex];
    setSelectedSegmentId(newSegment.id);
    // Seek to segment start
    handleSeek(newSegment.startMs / 1000);
  }, [segments, selectedSegmentId]);

  // Navigate to next segment
  const goToNextSegment = useCallback(() => {
    if (segments.length === 0) return;
    const sortedSegments = [...segments].sort((a, b) => a.startMs - b.startMs);
    const currentIndex = selectedSegmentId
      ? sortedSegments.findIndex(s => s.id === selectedSegmentId)
      : -1;
    const newIndex = currentIndex < sortedSegments.length - 1 ? currentIndex + 1 : 0;
    const newSegment = sortedSegments[newIndex];
    setSelectedSegmentId(newSegment.id);
    // Seek to segment start
    handleSeek(newSegment.startMs / 1000);
  }, [segments, selectedSegmentId]);

  // Check for gaps between segments
  const gaps = useMemo(() => {
    if (segments.length === 0) return [];
    const sortedSegments = [...segments].sort((a, b) => a.startMs - b.startMs);
    const foundGaps: { startMs: number; endMs: number; afterSegmentId: string }[] = [];

    for (let i = 0; i < sortedSegments.length - 1; i++) {
      const gap = sortedSegments[i + 1].startMs - sortedSegments[i].endMs;
      if (gap > 1) { // More than 1ms gap
        foundGaps.push({
          startMs: sortedSegments[i].endMs,
          endMs: sortedSegments[i + 1].startMs,
          afterSegmentId: sortedSegments[i].id,
        });
      }
    }
    return foundGaps;
  }, [segments]);

  // Check for overlaps between segments (next segment starts before previous ends)
  const overlaps = useMemo(() => {
    if (segments.length === 0) return [];
    const sortedSegments = [...segments].sort((a, b) => a.startMs - b.startMs);
    const foundOverlaps: { overlapMs: number; afterSegmentId: string }[] = [];

    for (let i = 0; i < sortedSegments.length - 1; i++) {
      const gap = sortedSegments[i + 1].startMs - sortedSegments[i].endMs;
      if (gap < -1) { // More than 1ms overlap
        foundOverlaps.push({
          overlapMs: -gap,
          afterSegmentId: sortedSegments[i].id,
        });
      }
    }
    return foundOverlaps;
  }, [segments]);

  // Fix overlaps by truncating the end of the earlier segment
  const handleFixOverlaps = useCallback(() => {
    if (overlaps.length === 0) return;

    const sortedSegments = [...segments].sort((a, b) => a.startMs - b.startMs);
    const updatedById = new Map(segments.map(s => [s.id, { ...s }]));

    for (let i = 0; i < sortedSegments.length - 1; i++) {
      const gap = sortedSegments[i + 1].startMs - sortedSegments[i].endMs;
      if (gap < -1) {
        const seg = updatedById.get(sortedSegments[i].id);
        if (seg) {
          seg.endMs = sortedSegments[i + 1].startMs;
          updatedById.set(seg.id, seg);
        }
      }
    }

    setSegments(Array.from(updatedById.values()));
    toast.success(`Fixed ${overlaps.length} overlap${overlaps.length > 1 ? 's' : ''}`);
  }, [overlaps, segments]);

  // Check for adjacent segments of the same type that can be merged
  const mergeableCount = useMemo(() => {
    if (segments.length < 2) return 0;
    const sortedSegments = [...segments].sort((a, b) => a.startMs - b.startMs);
    let count = 0;
    for (let i = 0; i < sortedSegments.length - 1; i++) {
      if (sortedSegments[i].type === sortedSegments[i + 1].type) {
        count++;
      }
    }
    return count;
  }, [segments]);

  // Split segment at playhead (blade tool)
  const handleSplitSegment = useCallback(() => {
    if (segments.length === 0) return;

    const currentTimeMs = currentTime * 1000;
    const sortedSegments = [...segments].sort((a, b) => a.startMs - b.startMs);

    // Find segment containing the playhead
    const segmentIndex = sortedSegments.findIndex(
      seg => currentTimeMs >= seg.startMs && currentTimeMs < seg.endMs
    );

    if (segmentIndex === -1) {
      toast.error('Playhead is not within any segment');
      return;
    }

    const segment = sortedSegments[segmentIndex];

    // Don't split if too close to edges (minimum 500ms on each side)
    const minDuration = 500;
    if (currentTimeMs - segment.startMs < minDuration || segment.endMs - currentTimeMs < minDuration) {
      toast.error('Cannot split: resulting segments would be too short');
      return;
    }

    // Create two new segments from the split
    const firstHalf: SegmentRegion = {
      ...segment,
      id: `${segment.id}-a`,
      endMs: currentTimeMs,
    };

    const secondHalf: SegmentRegion = {
      ...segment,
      id: `${segment.id}-b`,
      startMs: currentTimeMs,
      rallyNumber: segment.type === 'in-play' ? (segment.rallyNumber || 0) + 1 : undefined,
    };

    // Replace original segment with two halves
    const newSegments = [
      ...sortedSegments.slice(0, segmentIndex),
      firstHalf,
      secondHalf,
      ...sortedSegments.slice(segmentIndex + 1),
    ];

    setSegments(newSegments);
    setSelectedSegmentId(secondHalf.id);
    toast.success('Segment split at playhead');
  }, [segments, currentTime]);

  // Merge adjacent segments of the same type
  const handleMergeAdjacentSegments = useCallback(() => {
    if (segments.length < 2) return;

    const sortedSegments = [...segments].sort((a, b) => a.startMs - b.startMs);
    const mergedSegments: SegmentRegion[] = [];
    let currentMerge: SegmentRegion | null = null;

    for (const segment of sortedSegments) {
      if (!currentMerge) {
        currentMerge = { ...segment };
      } else if (currentMerge.type === segment.type) {
        // Same type - extend the current merge
        currentMerge.endMs = segment.endMs;
        // Keep the rally number from the first segment in the merge
      } else {
        // Different type - save current merge and start new one
        mergedSegments.push(currentMerge);
        currentMerge = { ...segment };
      }
    }

    // Don't forget the last segment
    if (currentMerge) {
      mergedSegments.push(currentMerge);
    }

    // Renumber segments and rally numbers
    let rallyNumber = 1;
    const renumberedSegments = mergedSegments.map((seg, index) => ({
      ...seg,
      id: `segment-${index + 1}`,
      rallyNumber: seg.type === 'in-play' ? rallyNumber++ : undefined,
    }));

    const mergedCount = segments.length - renumberedSegments.length;
    setSegments(renumberedSegments);
    setSelectedSegmentId(null);
    toast.success(`Merged ${mergedCount} segment${mergedCount > 1 ? 's' : ''} (${renumberedSegments.length} remaining)`);
  }, [segments]);

  // Auto-fill gaps using adjacent segment type
  const handleAutoFillGaps = useCallback(() => {
    if (gaps.length === 0) return;

    const sortedSegments = [...segments].sort((a, b) => a.startMs - b.startMs);
    const newSegments: SegmentRegion[] = [];
    let segmentCounter = sortedSegments.length;

    for (let i = 0; i < sortedSegments.length; i++) {
      newSegments.push(sortedSegments[i]);

      // Check if there's a gap after this segment
      const gap = gaps.find(g => g.afterSegmentId === sortedSegments[i].id);
      if (gap) {
        // Create new segment with the preceding segment's type
        segmentCounter++;
        newSegments.push({
          id: `segment-filled-${segmentCounter}`,
          startMs: gap.startMs,
          endMs: gap.endMs,
          type: sortedSegments[i].type, // Use preceding segment's type
          confidence: 1.0,
          rallyNumber: sortedSegments[i].type === 'in-play'
            ? (sortedSegments[i].rallyNumber || 0) + 1
            : undefined,
        });
      }
    }

    setSegments(newSegments);
    toast.success(`Filled ${gaps.length} gap${gaps.length > 1 ? 's' : ''}`);
  }, [gaps, segments]);

  // Export corrected annotations
  const handleExportAnnotations = useCallback(async () => {
    if (!rawVideoPath || segments.length === 0) {
      toast.error('No annotations to export');
      return;
    }

    // Validate segments are contiguous (no gaps)
    const sortedSegments = [...segments].sort((a, b) => a.startMs - b.startMs);
    for (let i = 0; i < sortedSegments.length - 1; i++) {
      const gap = sortedSegments[i + 1].startMs - sortedSegments[i].endMs;
      if (Math.abs(gap) > 1) { // Allow 1ms tolerance
        toast.error(`Gap detected between segments ${i + 1} and ${i + 2}`);
        return;
      }
    }

    // Number in-play segments fresh: 1..N in time order, regardless of how they
    // were created (split, merge, manual blank start). Fixes rally_number coming
    // out null on split-only timelines.
    let rallyCounter = 0;
    const numberedSegments = sortedSegments.map((seg, index) => {
      const isInPlay = seg.type === 'in-play';
      if (isInPlay) rallyCounter += 1;
      return {
        segment_id: index + 1,
        type: seg.type,
        start_ms: Math.round(seg.startMs),
        end_ms: Math.round(seg.endMs),
        start_frame: Math.round((seg.startMs / 1000) * 30),
        end_frame: Math.round((seg.endMs / 1000) * 30),
        rally_number: isInPlay ? rallyCounter : null,
      };
    });

    // Build corrected JSON
    const correctedJson = {
      video_metadata: {
        path: rawVideoPath,
        fps: 0, // Will be filled from original annotation if available
        duration_seconds: duration,
        total_frames: 0,
      },
      segments: numberedSegments,
      correction_metadata: {
        corrected_at: new Date().toISOString(),
        mode: 'rally',
      },
    };

    const jsonString = JSON.stringify(correctedJson, null, 2);

    try {
      const result = await window.electronAPI.saveCorrectedAnnotation(rawVideoPath, jsonString);

      if (result.cancelled) {
        return;
      }

      if (result.success && result.path) {
        localStorage.removeItem(`annotation-draft:${rawVideoPath}`);
        setDraftSavedAt(null);
        if (result.warning) {
          toast.warning(
            `Saved with issues: ${result.warning.outOfRangeCount} segment(s) extend past `
            + `video end (duration ${result.warning.videoDurationSec}s, max end `
            + `${result.warning.maxEndSec}s). Likely loaded from a stale raw file — `
            + `please review.`,
            { duration: 10000 },
          );
        } else {
          toast.success(`Saved to ${result.path}`);
        }
      } else {
        toast.error(result.error || 'Failed to save annotations');
      }
    } catch (err) {
      console.error('Error saving annotations:', err);
      toast.error('Failed to save annotations');
    }
  }, [rawVideoPath, segments, duration]);

  // Check for existing video path on load
  useEffect(() => {
    async function checkExistingVideo() {
      try {
        const result = await window.electronAPI.getCurrentVideoPath();
        if (result.success && result.path) {
          const videoUrl = toFileUrl(result.path);
          setVideoPath(videoUrl);
          setRawVideoPath(result.path);
          await detectVideoContext(result.path);
        }
      } catch (err) {
        console.error('Error checking for existing video:', err);
      } finally {
        setLoading(false);
      }
    }
    checkExistingVideo();
  }, []);

  // Detect what files are available and present options — never auto-decides
  const detectVideoContext = useCallback(async (filePath: string) => {
    setLoadingState({ status: 'detecting' });
    try {
      const context = await window.electronAPI.detectVideoContext(filePath);
      setLoadingState({ status: 'ready', context });
    } catch (err) {
      console.error('Error detecting video context:', err);
      setLoadingState({ status: 'idle' });
    }
  }, []);

  // Run Gemini detection (annotate_fast.py)
  const runDetection = useCallback(async (filePath: string, startBatch?: number) => {
    setLoadingState({
      status: 'running-detection',
      currentBatch: 0,
      totalBatches: 0,
      percentage: 0,
    });
    try {
      const result = await window.electronAPI.runAnnotation(filePath, { startBatch });
      if (!result.success) {
        toast.error(result.error || 'Failed to start detection');
        setLoadingState({ status: 'idle' });
      }
    } catch (err) {
      console.error('Error starting detection:', err);
      toast.error('Failed to start detection');
      setLoadingState({ status: 'idle' });
    }
  }, []);

  // Cancel running detection
  const handleCancelAnnotation = useCallback(async () => {
    try {
      await window.electronAPI.stopAnnotation();
    } catch (err) {
      console.error('Error stopping detection:', err);
    }
  }, []);

  // Retry detection from beginning
  const handleRetryAnnotation = useCallback(() => {
    setShowErrorDialog(false);
    if (rawVideoPath) runDetection(rawVideoPath);
  }, [rawVideoPath, runDetection]);

  // Resume detection from last batch
  const handleResumeAnnotation = useCallback(() => {
    setShowErrorDialog(false);
    if (rawVideoPath && loadingState.status === 'detection-error') {
      runDetection(rawVideoPath, loadingState.lastBatch);
    }
  }, [rawVideoPath, loadingState, runDetection]);

  // Open annotation file picker (escape hatch — any JSON)
  const handleLoadFile = useCallback(async () => {
    const result = await window.electronAPI.openAnnotationFilePicker();
    if (!result.success || result.cancelled || !result.path) return;
    await loadAnnotationFromFile(result.path, 'file');
  }, [loadAnnotationFromFile]);

  const handleSelectVideo = useCallback(async () => {
    try {
      const result = await window.electronAPI.openVideoFilePicker();
      if (result.success && result.path) {
        const videoUrl = toFileUrl(result.path);
        // Reset playback state for new video
        setCurrentTime(0);
        setDuration(0);
        setIsPlaying(false);
        setError(null);
        setLoadingState({ status: 'idle' });
        // Reset segment state
        setSegments([]);
        setSelectedSegmentId(null);
        // Set new video path
        setVideoPath(videoUrl);
        setRawVideoPath(result.path);
        await window.electronAPI.setCurrentVideoPath(result.path);
        await detectVideoContext(result.path);
      }
    } catch (err) {
      toast.error('Failed to select video');
      console.error('Error selecting video:', err);
    }
  }, [detectVideoContext]);

  function togglePlayPause() {
    const playback = videoPlaybackRef.current;
    const video = playback?.video;
    if (!playback || !video) return;

    // If reversing, Space pauses (stops reverse) without starting forward play
    if (isReversingRef.current) {
      setIsReversing(false);
      return;
    }

    if (isPlaying) {
      playback.pause();
    } else {
      playback.play().catch(err => console.error('Video play failed:', err));
    }
  }

  function handleSeek(time: number) {
    const video = videoPlaybackRef.current?.video;
    if (!video) return;
    video.currentTime = time;
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // Space bar to play/pause
      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        togglePlayPause();
        return;
      }

      // Segment navigation (Up/K = next, Down/J = previous)
      if (e.key === 'ArrowUp' || e.key === 'k' || e.key === 'K') {
        e.preventDefault();
        goToNextSegment();
        return;
      }

      if (e.key === 'ArrowDown' || e.key === 'j' || e.key === 'J') {
        e.preventDefault();
        goToPrevSegment();
        return;
      }

      // Set segment type with number keys
      if (e.key === '1' && selectedSegmentId) {
        e.preventDefault();
        handleSegmentTypeChange('in-play');
        return;
      }

      if (e.key === '2' && selectedSegmentId) {
        e.preventDefault();
        handleSegmentTypeChange('break');
        return;
      }

      // Frame-by-frame navigation with arrow keys (when holding Shift)
      if (e.shiftKey && e.key === 'ArrowLeft') {
        e.preventDefault();
        const newTime = Math.max(0, currentTime - 1/30); // ~1 frame at 30fps
        handleSeek(newTime);
        return;
      }

      if (e.shiftKey && e.key === 'ArrowRight') {
        e.preventDefault();
        const newTime = Math.min(duration, currentTime + 1/30);
        handleSeek(newTime);
        return;
      }

      // Z to toggle timeline zoom
      if ((e.key === 'z' || e.key === 'Z') && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        segmentTimelineRef.current?.toggleZoom();
        return;
      }

      // M to merge adjacent segments
      if ((e.key === 'm' || e.key === 'M') && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        if (mergeableCount > 0) {
          handleMergeAdjacentSegments();
        }
        return;
      }

      // L to cycle playback speed: 1x -> 2x -> 4x -> 1x
      if ((e.key === 'l' || e.key === 'L') && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        const video = videoPlaybackRef.current?.video;
        if (video) {
          const nextSpeed = video.playbackRate >= 4 ? 1 : video.playbackRate >= 2 ? 4 : 2;
          video.playbackRate = nextSpeed;
          setPlaybackSpeed(nextSpeed);
        }
        return;
      }

      // H to toggle reverse playback
      if ((e.key === 'h' || e.key === 'H') && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        setIsReversing(prev => !prev);
        return;
      }

      // B to split segment at playhead (blade tool)
      if ((e.key === 'b' || e.key === 'B') && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        handleSplitSegment();
        return;
      }

      // S to mark a set moment at the current playhead
      if ((e.key === 's' || e.key === 'S') && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        handleAddSetMoment(currentTime * 1000);
        return;
      }

      // Delete/Backspace to remove selected set moment
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedSetMomentId) {
        e.preventDefault();
        handleDeleteSetMoment(selectedSetMomentId);
        return;
      }

    };

    window.addEventListener('keydown', handleKeyDown, { capture: true });
    return () => window.removeEventListener('keydown', handleKeyDown, { capture: true });
  }, [isPlaying, goToPrevSegment, goToNextSegment, selectedSegmentId, handleSegmentTypeChange, currentTime, duration, mergeableCount, handleMergeAdjacentSegments, handleSplitSegment, handleAddSetMoment, handleDeleteSetMoment, selectedSetMomentId]);

  // Reverse playback via requestAnimationFrame
  useEffect(() => {
    if (!isReversing) return;

    const video = videoPlaybackRef.current?.video;
    if (!video) {
      setIsReversing(false);
      return;
    }

    video.pause();
    let lastTimestamp: number | null = null;
    let rafId: number;

    const step = (timestamp: number) => {
      if (lastTimestamp !== null) {
        const elapsed = (timestamp - lastTimestamp) / 1000;
        const newTime = video.currentTime - elapsed * playbackSpeed;
        if (newTime <= 0) {
          video.currentTime = 0;
          setIsReversing(false);
          return;
        }
        video.currentTime = newTime;
      }
      lastTimestamp = timestamp;
      rafId = requestAnimationFrame(step);
    };

    rafId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafId);
  }, [isReversing, playbackSpeed]);

  // Stop reversing when user plays forward
  useEffect(() => {
    if (isPlaying && isReversing) {
      setIsReversing(false);
    }
  }, [isPlaying, isReversing]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#09090b]">
        <div className="text-slate-400">Loading...</div>
      </div>
    );
  }

  // Video selection screen
  if (!videoPath) {
    return (
      <div className="flex flex-col h-screen bg-[#09090b] text-slate-200">
        <div
          className="h-10 flex-shrink-0 bg-[#09090b]/80 backdrop-blur-md border-b border-white/5 flex items-center px-6"
          style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
        />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-24 h-24 mx-auto mb-6 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
              <FileVideo className="w-12 h-12 text-slate-500" />
            </div>
            <h2 className="text-xl font-semibold mb-2">Select a Video</h2>
            <p className="text-slate-400 mb-6 max-w-md">
              Choose a volleyball video to annotate. The video will be analyzed to detect rally segments.
            </p>
            <Button
              onClick={handleSelectVideo}
              size="lg"
              className="gap-2 bg-[#34B27B] hover:bg-[#34B27B]/90 text-white"
            >
              <Upload className="w-5 h-5" />
              Select Video
            </Button>
          </div>
        </div>
        <Toaster theme="dark" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col h-screen bg-[#09090b] text-slate-200">
        <div
          className="h-10 flex-shrink-0 bg-[#09090b]/80 backdrop-blur-md border-b border-white/5 flex items-center px-6"
          style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
        />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-red-400 mb-4">{error}</div>
            <Button onClick={handleSelectVideo} variant="outline">
              Select Different Video
            </Button>
          </div>
        </div>
        <Toaster theme="dark" />
      </div>
    );
  }

  const isProcessing = loadingState.status === 'running-detection' || loadingState.status === 'detecting' || loadingState.status === 'loading';

  return (
    <div className="flex flex-col h-screen bg-[#09090b] text-slate-200 overflow-hidden selection:bg-[#34B27B]/30">
      {/* Title bar */}
      <div
        className="h-10 flex-shrink-0 bg-[#09090b]/80 backdrop-blur-md border-b border-white/5 flex items-center justify-between px-6 z-50"
        style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
      >
        <div className="flex-1" />
      </div>

      <div className="flex-1 p-5 gap-4 flex min-h-0 relative">
        {/* Main content area */}
        <div className="flex-1 flex flex-col gap-3 min-w-0 h-full">
          <PanelGroup direction="vertical" className="gap-3">
            {/* Video preview */}
            <Panel defaultSize={70} minSize={40}>
              <div className="w-full h-full flex flex-col items-center justify-center bg-black/40 rounded-2xl border border-white/5 shadow-2xl overflow-hidden relative">
                {/* Processing overlay */}
                {isProcessing && (
                  <ProcessingOverlay
                    mode={loadingState.status === 'detecting' || loadingState.status === 'loading' ? 'checking' : 'processing'}
                    currentBatch={loadingState.status === 'running-detection' ? loadingState.currentBatch : 0}
                    totalBatches={loadingState.status === 'running-detection' ? loadingState.totalBatches : 0}
                    percentage={loadingState.status === 'running-detection' ? loadingState.percentage : 0}
                    onCancel={handleCancelAnnotation}
                  />
                )}

                <div className="w-full flex justify-center items-center" style={{ flex: '1 1 auto', margin: '6px 0 0' }}>
                  <div className="relative" style={{ width: 'auto', height: '100%', aspectRatio: '16/9', maxWidth: '100%', margin: '0 auto' }}>
                    <VideoPlayback
                      key={videoPath}
                      ref={videoPlaybackRef}
                      videoPath={videoPath}
                      onDurationChange={setDuration}
                      onTimeUpdate={setCurrentTime}
                      currentTime={currentTime}
                      onPlayStateChange={setIsPlaying}
                      onError={setError}
                      isPlaying={isPlaying}
                    />
                  </div>
                </div>
                {/* Playback controls */}
                <div className="w-full flex justify-center items-center" style={{ height: '48px', flexShrink: 0, padding: '6px 12px', margin: '6px 0' }}>
                  <div className="relative" style={{ width: '100%', maxWidth: '700px' }}>
                    <PlaybackControls
                      isPlaying={isPlaying}
                      currentTime={currentTime}
                      duration={duration}
                      onTogglePlayPause={togglePlayPause}
                      onSeek={handleSeek}
                    />
                    {(playbackSpeed !== 1 || isReversing) && (
                      <span className="absolute -right-14 top-1/2 -translate-y-1/2 text-[10px] font-mono text-[#34B27B] font-semibold">
                        {isReversing ? `◀ ${playbackSpeed}x` : `${playbackSpeed}x`}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </Panel>

            <PanelResizeHandle className="h-3 bg-[#09090b]/80 hover:bg-[#09090b] transition-colors rounded-full mx-4 flex items-center justify-center">
              <div className="w-8 h-1 bg-white/20 rounded-full"></div>
            </PanelResizeHandle>

            {/* Timeline section */}
            <Panel defaultSize={30} minSize={20}>
              <div className="h-full bg-[#09090b] rounded-2xl border border-white/5 shadow-lg overflow-hidden flex flex-col p-4">
                {segments.length > 0 ? (
                  <SegmentTimeline
                    ref={segmentTimelineRef}
                    segments={segments}
                    videoDurationMs={duration * 1000 || annotationDurationMs}
                    currentTimeMs={currentTime * 1000}
                    selectedSegmentId={selectedSegmentId}
                    onSelectSegment={(id) => {
                      setSelectedSegmentId(id);
                      if (id) setSelectedSetMomentId(null);
                    }}
                    onSegmentBoundaryChange={handleSegmentBoundaryChange}
                    onSeek={(timeMs) => handleSeek(timeMs / 1000)}
                    setMoments={setMoments}
                    selectedSetMomentId={selectedSetMomentId}
                    onSelectSetMoment={(id) => {
                      setSelectedSetMomentId(id);
                      if (id) setSelectedSegmentId(null);
                    }}
                    onAddSetMoment={handleAddSetMoment}
                    onDeleteSetMoment={handleDeleteSetMoment}
                  />
                ) : loadingState.status === 'idle' || loadingState.status === 'ready' || loadingState.status === 'detection-error' ? (
                  <div className="flex-1 flex flex-col items-center justify-center">
                    <p className="text-slate-500 text-sm">No annotation loaded</p>
                  </div>
                ) : (
                  <div className="flex-1 flex items-center justify-center">
                    <p className="text-slate-500 text-sm">Analyzing video...</p>
                  </div>
                )}
              </div>
            </Panel>
          </PanelGroup>
        </div>

        {/* Right sidebar */}
        <div className="w-64 min-w-0 bg-[#09090b] border border-white/5 rounded-2xl flex flex-col shadow-xl h-full overflow-hidden">
          <div className="flex-1 p-4 flex flex-col overflow-y-auto">
            {/* Video context panel — shows when context detected, awaiting user action */}
            {loadingState.status === 'ready' && (
              <VideoContextPanel
                context={loadingState.context}
                onLoadRallyEdit={() => {
                  const mapPath = loadingState.context.rallyEditMap?.path;
                  if (mapPath) loadAnnotationFromChapterMap(mapPath);
                }}
                onLoadCorrected={() => {
                  const p = loadingState.context.correctedAnnotation?.path;
                  if (p) loadAnnotationFromFile(p, 'corrected');
                }}
                onLoadRaw={() => {
                  const p = loadingState.context.rawAnnotation?.path;
                  if (p) loadAnnotationFromFile(p, 'raw');
                }}
                onRunDetection={() => {
                  if (rawVideoPath) runDetection(rawVideoPath);
                }}
                onLoadFile={handleLoadFile}
                onStartBlank={handleStartBlank}
              />
            )}

            {/* Status — shows after loading or while detecting */}
            {loadingState.status !== 'ready' && (
              <div className="mb-4 p-3 rounded-lg bg-white/5 border border-white/5">
                <p className="text-xs text-slate-400 mb-1">Status</p>
                {loadingState.status === 'idle' && (
                  <p className="text-sm text-slate-300">No video loaded</p>
                )}
                {(loadingState.status === 'detecting' || loadingState.status === 'loading') && (
                  <p className="text-sm text-yellow-400">Loading…</p>
                )}
                {loadingState.status === 'running-detection' && (
                  <p className="text-sm text-[#34B27B]">Running detection…</p>
                )}
                {loadingState.status === 'loaded' && (
                  <p className="text-sm text-[#34B27B]">
                    {segments.length} segments loaded
                    <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-[#34B27B]/20 text-[#34B27B] border border-[#34B27B]/30">
                      {loadingState.source}
                    </span>
                  </p>
                )}
                {draftSavedAt && (
                  <p className="text-xs text-slate-500 mt-1">
                    Draft saved {draftSavedAt.toLocaleTimeString()}
                  </p>
                )}
                {loadingState.status === 'detection-error' && (
                  <p className="text-sm text-red-400">Detection failed</p>
                )}
              </div>
            )}

            {/* Gap warning */}
            {gaps.length > 0 && (
              <div className="mb-4 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                <p className="text-xs text-yellow-400 mb-2">
                  {gaps.length} gap{gaps.length > 1 ? 's' : ''} detected between segments
                </p>
                <Button
                  onClick={handleAutoFillGaps}
                  size="sm"
                  variant="outline"
                  className="w-full text-xs border-yellow-500/30 text-yellow-400 hover:bg-yellow-500/10"
                >
                  Auto-fill gaps
                </Button>
              </div>
            )}

            {/* Overlap warning */}
            {overlaps.length > 0 && (
              <div className="mb-4 p-3 rounded-lg bg-orange-500/10 border border-orange-500/20">
                <p className="text-xs text-orange-400 mb-2">
                  {overlaps.length} overlap{overlaps.length > 1 ? 's' : ''} detected between segments
                </p>
                <Button
                  onClick={handleFixOverlaps}
                  size="sm"
                  variant="outline"
                  className="w-full text-xs border-orange-500/30 text-orange-400 hover:bg-orange-500/10"
                >
                  Fix overlaps
                </Button>
              </div>
            )}

            {/* Merge adjacent segments */}
            {mergeableCount > 0 && (
              <div className="mb-4 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                <p className="text-xs text-blue-400 mb-2">
                  {mergeableCount} adjacent segment{mergeableCount > 1 ? 's' : ''} can be merged
                </p>
                <Button
                  onClick={handleMergeAdjacentSegments}
                  size="sm"
                  variant="outline"
                  className="w-full text-xs border-blue-500/30 text-blue-400 hover:bg-blue-500/10"
                >
                  Merge adjacent segments
                </Button>
              </div>
            )}

            {/* Selected segment details */}
            {selectedSegment && (
              <div className="mb-4 p-3 rounded-lg bg-white/5 border border-white/5">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-slate-400">Selected Segment</p>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={goToPrevSegment}
                      className="p-1 rounded hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                      title="Previous segment (↑ or K)"
                    >
                      <ChevronUp className="w-4 h-4" />
                    </button>
                    <button
                      onClick={goToNextSegment}
                      className="p-1 rounded hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                      title="Next segment (↓ or J)"
                    >
                      <ChevronDown className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Segment type selector */}
                <div className="mb-3">
                  <p className="text-xs text-slate-500 mb-1">Type</p>
                  <Select
                    value={selectedSegment.type}
                    onValueChange={(value) => handleSegmentTypeChange(value as SegmentType)}
                  >
                    <SelectTrigger className="w-full bg-white/5 border-white/10 text-slate-200">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-[#1a1a1a] border-white/10">
                      <SelectItem value="in-play" className="text-slate-200 hover:bg-white/10">
                        In-Play (Rally)
                      </SelectItem>
                      <SelectItem value="break" className="text-slate-200 hover:bg-white/10">
                        Break
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Time info */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <p className="text-slate-500">Start</p>
                    <p className="text-slate-300 font-mono">
                      {formatTimeWithMs(selectedSegment.startMs)}
                    </p>
                  </div>
                  <div>
                    <p className="text-slate-500">End</p>
                    <p className="text-slate-300 font-mono">
                      {formatTimeWithMs(selectedSegment.endMs)}
                    </p>
                  </div>
                  <div>
                    <p className="text-slate-500">Duration</p>
                    <p className="text-slate-300 font-mono">
                      {((selectedSegment.endMs - selectedSegment.startMs) / 1000).toFixed(1)}s
                    </p>
                  </div>
                  {selectedSegment.confidence !== undefined && (
                    <div>
                      <p className="text-slate-500">Confidence</p>
                      <p className="text-slate-300 font-mono">
                        {Math.round(selectedSegment.confidence * 100)}%
                      </p>
                    </div>
                  )}
                </div>

                {selectedSegment.rallyNumber && (
                  <div className="mt-2 text-xs">
                    <p className="text-slate-500">Rally Number</p>
                    <p className="text-slate-300">{selectedSegment.rallyNumber}</p>
                  </div>
                )}
              </div>
            )}

            {/* Set moments list */}
            {setMoments.length > 0 && (
              <div className="mb-4 p-3 rounded-lg bg-white/5 border border-white/5">
                <p className="text-xs text-slate-400 mb-2">
                  Set Moments ({setMoments.length})
                </p>
                <div className="space-y-1 max-h-40 overflow-y-auto">
                  {[...setMoments]
                    .sort((a, b) => a.timeMs - b.timeMs)
                    .map(m => (
                      <div
                        key={m.id}
                        className={`flex items-center justify-between px-2 py-1 rounded cursor-pointer text-xs transition-colors ${
                          selectedSetMomentId === m.id
                            ? 'bg-[#f59e0b]/20 border border-[#f59e0b]/40'
                            : 'hover:bg-white/5'
                        }`}
                        onClick={() => {
                          setSelectedSetMomentId(m.id);
                          setSelectedSegmentId(null);
                          handleSeek(m.timeMs / 1000);
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <span
                            className="w-2 h-2 rotate-45 flex-shrink-0"
                            style={{ background: '#f59e0b' }}
                          />
                          <span className="font-mono text-slate-300">
                            {formatTimeWithMs(m.timeMs)}
                          </span>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteSetMoment(m.id);
                          }}
                          className="p-0.5 rounded hover:bg-red-500/20 text-slate-500 hover:text-red-400 transition-colors"
                          title="Delete set moment"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Keyboard shortcuts */}
            {segments.length > 0 && (
              <div className="mb-4 p-3 rounded-lg bg-white/5 border border-white/5">
                <p className="text-xs text-slate-400 mb-2">Shortcuts</p>
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between text-slate-500">
                    <span>Play/Pause</span>
                    <span className="font-mono text-slate-400">Space</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Next/Prev segment</span>
                    <span className="font-mono text-slate-400">↑/↓ or K/J</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Set In-Play</span>
                    <span className="font-mono text-slate-400">1</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Set Break</span>
                    <span className="font-mono text-slate-400">2</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Frame step</span>
                    <span className="font-mono text-slate-400">Shift+←→</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Toggle zoom</span>
                    <span className="font-mono text-slate-400">Z</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Merge adjacent</span>
                    <span className="font-mono text-slate-400">M</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Split at playhead</span>
                    <span className="font-mono text-slate-400">B</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Mark set moment</span>
                    <span className="font-mono text-slate-400">S</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Delete set moment</span>
                    <span className="font-mono text-slate-400">Del/⌫</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Cycle speed{playbackSpeed !== 1 ? ` (${playbackSpeed}x)` : ""}</span>
                    <span className="font-mono text-slate-400">L</span>
                  </div>
                  <div className="flex justify-between text-slate-500">
                    <span>Reverse{isReversing ? " (on)" : ""}</span>
                    <span className="font-mono text-slate-400">H</span>
                  </div>
                </div>
              </div>
            )}

            <div className="flex-1" />

            {/* Actions */}
            <div className="space-y-2">
              {segments.length > 0 && (
                <Button
                  onClick={handleExportAnnotations}
                  className="w-full gap-2 bg-[#34B27B] hover:bg-[#34B27B]/90 text-white"
                >
                  <Download className="w-4 h-4" />
                  Export JSON
                </Button>
              )}

              {setMoments.length > 0 && (
                <Button
                  onClick={handleExportSetAnnotations}
                  className="w-full gap-2 bg-[#f59e0b] hover:bg-[#f59e0b]/90 text-white"
                >
                  <Crosshair className="w-4 h-4" />
                  Save Set Annotations
                </Button>
              )}

              {loadingState.status === 'detection-error' && (
                <Button
                  onClick={() => setShowErrorDialog(true)}
                  variant="outline"
                  className="w-full border-red-500/30 text-red-400 hover:bg-red-500/10"
                >
                  View Error
                </Button>
              )}

              <Button
                onClick={handleSelectVideo}
                variant="outline"
                className="w-full gap-2"
                disabled={isProcessing}
              >
                <Upload className="w-4 h-4" />
                Change Video
              </Button>
            </div>
          </div>
        </div>
      </div>

      <Toaster theme="dark" className="pointer-events-auto" />

      {/* Error dialog */}
      {loadingState.status === 'detection-error' && (
        <AnnotationErrorDialog
          isOpen={showErrorDialog}
          onClose={() => setShowErrorDialog(false)}
          error={loadingState.error}
          lastBatch={loadingState.lastBatch}
          output={loadingState.output}
          onRetry={handleRetryAnnotation}
          onResume={handleResumeAnnotation}
        />
      )}
    </div>
  );
}
