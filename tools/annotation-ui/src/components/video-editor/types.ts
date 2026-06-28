// Segment types for annotation correction
export type SegmentType = 'in-play' | 'break';

export interface SegmentRegion {
  id: string;
  startMs: number;
  endMs: number;
  type: SegmentType;
  confidence?: number;  // From LLM, for display only
  rallyNumber?: number; // For in-play segments
}

export const SEGMENT_COLORS: Record<SegmentType, string> = {
  'in-play': '#22c55e',  // Green
  'break': '#6b7280',    // Gray
};

// Annotation JSON format from Python scripts.
// Supports both frame-based (legacy) and ms-based (video pipeline) formats.
export interface AnnotationJSON {
  video_metadata?: {
    filename?: string;
    fps?: number;
    total_frames?: number;
    duration_seconds?: number;
    width?: number;
    height?: number;
    extraction_fps?: number;
  };
  annotation_metadata?: {
    pipeline?: string;
    [key: string]: unknown;
  };
  segments: Array<{
    segment_id?: number;
    type: string;
    // Frame-based format (legacy)
    frame_start?: number;
    frame_end?: number;
    timestamp_start?: string;
    timestamp_end?: string;
    // Ms-based format (video pipeline)
    start_ms?: number;
    end_ms?: number;
    rally_number: number | null;
    confidence?: number;
    description?: string;
  }>;
}

// Result of loading annotation segments
export interface LoadAnnotationResult {
  segments: SegmentRegion[];
  durationMs: number;
}

// Convert annotation JSON to SegmentRegion array.
// Handles both frame-based and ms-based segment formats.
export function loadAnnotationSegments(json: AnnotationJSON): LoadAnnotationResult {
  const extractionFps = json.video_metadata?.extraction_fps || 1;

  const segments = json.segments.map((seg, idx) => {
    let startMs: number;
    let endMs: number;

    if (seg.start_ms !== undefined && seg.end_ms !== undefined) {
      // Video pipeline format: start_ms/end_ms already in milliseconds
      startMs = seg.start_ms;
      endMs = seg.end_ms;
    } else if (seg.frame_start !== undefined && seg.frame_end !== undefined) {
      // Legacy frame-based format: convert using extraction_fps
      startMs = (seg.frame_start / extractionFps) * 1000;
      endMs = ((seg.frame_end + 1) / extractionFps) * 1000;
    } else {
      startMs = 0;
      endMs = 0;
    }

    return {
      id: `segment-${seg.segment_id ?? idx + 1}`,
      startMs,
      endMs,
      type: seg.type as SegmentType,
      confidence: seg.confidence,
      rallyNumber: seg.rally_number ?? undefined,
    };
  });

  const durationMs = json.video_metadata?.duration_seconds
    ? json.video_metadata.duration_seconds * 1000
    : segments.length > 0
      ? segments[segments.length - 1].endMs
      : 0;

  return { segments, durationMs };
}

// ---------------------------------------------------------------------------
// Video context — what files are available for a loaded video
// ---------------------------------------------------------------------------

export type VideoType = 'rally-edit' | 'raw-footage' | 'unknown';

export interface VideoContext {
  videoType: VideoType;
  rallyEditMap: { path: string; segmentCount: number; totalDurationSec: number } | null;
  correctedAnnotation: { path: string } | null;
  rawAnnotation: { path: string } | null;
}

// ---------------------------------------------------------------------------
// Chapter map — concat-time to original-time mapping for rally edits
// ---------------------------------------------------------------------------

export interface ChapterMapSegment {
  rally_number: number | null;
  segment_id: number | null;
  original_start_sec: number;
  original_end_sec: number;
  concat_start_sec: number;
  concat_end_sec: number;
}

export interface ChapterMapJSON {
  source_video: string;
  total_concat_duration_sec: number;
  segment_count: number;
  segments: ChapterMapSegment[];
}

// Load SegmentRegion[] from a rally edit chapter map.
// Each chapter entry is an in-play segment in concat-video time.
export function loadSegmentsFromChapterMap(map: ChapterMapJSON): LoadAnnotationResult {
  const segments: SegmentRegion[] = map.segments.map((seg, idx) => ({
    id: `segment-${idx + 1}`,
    startMs: Math.round(seg.concat_start_sec * 1000),
    endMs: Math.round(seg.concat_end_sec * 1000),
    type: 'in-play' as SegmentType,
    rallyNumber: seg.rally_number ?? idx + 1,
  }));
  return {
    segments,
    durationMs: Math.round(map.total_concat_duration_sec * 1000),
  };
}

// ---------------------------------------------------------------------------
// Set Moments — point-event annotations for set detection ground truth
// ---------------------------------------------------------------------------

export interface SetMoment {
  id: string;
  timeMs: number; // Exact ball-contact timestamp
}

export const SET_MOMENT_COLOR = '#f59e0b'; // Amber

// ---------------------------------------------------------------------------

export type ZoomDepth = 1 | 2 | 3 | 4 | 5 | 6;

export interface ZoomFocus {
  cx: number; // normalized horizontal center (0-1)
  cy: number; // normalized vertical center (0-1)
}

export interface ZoomRegion {
  id: string;
  startMs: number;
  endMs: number;
  depth: ZoomDepth;
  focus: ZoomFocus;
}

export interface TrimRegion {
  id: string;
  startMs: number;
  endMs: number;
}

export type AnnotationType = 'text' | 'image' | 'figure';

export type ArrowDirection = 'up' | 'down' | 'left' | 'right' | 'up-right' | 'up-left' | 'down-right' | 'down-left';

export interface FigureData {
  arrowDirection: ArrowDirection;
  color: string;
  strokeWidth: number;
}

export interface AnnotationPosition {
  x: number;
  y: number;
}

export interface AnnotationSize {
  width: number;
  height: number;
}

export interface AnnotationTextStyle {
  color: string;
  backgroundColor: string;
  fontSize: number; // pixels
  fontFamily: string;
  fontWeight: 'normal' | 'bold';
  fontStyle: 'normal' | 'italic';
  textDecoration: 'none' | 'underline';
  textAlign: 'left' | 'center' | 'right';
}

export interface AnnotationRegion {
  id: string;
  startMs: number;
  endMs: number;
  type: AnnotationType;
  content: string; // Legacy - still used for current type
  textContent?: string; // Separate storage for text
  imageContent?: string; // Separate storage for image data URL
  position: AnnotationPosition;
  size: AnnotationSize;
  style: AnnotationTextStyle;
  zIndex: number;
  figureData?: FigureData;
}

export const DEFAULT_ANNOTATION_POSITION: AnnotationPosition = {
  x: 50,
  y: 50,
};

export const DEFAULT_ANNOTATION_SIZE: AnnotationSize = {
  width: 30,
  height: 20,
};

export const DEFAULT_ANNOTATION_STYLE: AnnotationTextStyle = {
  color: '#ffffff',
  backgroundColor: 'transparent',
  fontSize: 32,
  fontFamily: 'Inter',
  fontWeight: 'bold',
  fontStyle: 'normal',
  textDecoration: 'none',
  textAlign: 'center',
};

export const DEFAULT_FIGURE_DATA: FigureData = {
  arrowDirection: 'right',
  color: '#34B27B',
  strokeWidth: 4,
};



export interface CropRegion {
  x: number; 
  y: number; 
  width: number; 
  height: number; 
}

export const DEFAULT_CROP_REGION: CropRegion = {
  x: 0,
  y: 0,
  width: 1,
  height: 1,
};

export const ZOOM_DEPTH_SCALES: Record<ZoomDepth, number> = {
  1: 1.25,
  2: 1.5,
  3: 1.8,
  4: 2.2,
  5: 3.5,
  6: 5.0,
};

export const DEFAULT_ZOOM_DEPTH: ZoomDepth = 3;

export function clampFocusToDepth(focus: ZoomFocus, _depth: ZoomDepth): ZoomFocus {
  return {
    cx: clamp(focus.cx, 0, 1),
    cy: clamp(focus.cy, 0, 1),
  };
}

function clamp(value: number, min: number, max: number) {
  if (Number.isNaN(value)) return (min + max) / 2;
  return Math.min(max, Math.max(min, value));
}
