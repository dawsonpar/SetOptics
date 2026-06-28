import { useCallback, useEffect, useMemo, useRef, useState, useImperativeHandle, forwardRef } from "react";
import { cn } from "@/lib/utils";
import { SegmentRegion, SEGMENT_COLORS, SetMoment, SET_MOMENT_COLOR } from "./types";

const MIN_SEGMENT_DURATION_MS = 500;
const MIN_VISIBLE_RANGE_MS = 5000; // Minimum 5 seconds visible
const ZOOM_FACTOR = 1.2; // Zoom speed

interface SegmentTimelineProps {
  segments: SegmentRegion[];
  videoDurationMs: number;
  currentTimeMs: number;
  selectedSegmentId: string | null;
  onSelectSegment: (id: string | null) => void;
  onSegmentBoundaryChange: (segmentId: string, newStartMs: number, newEndMs: number) => void;
  onSeek: (timeMs: number) => void;
  onZoomToggle?: () => void;
  // Set moment props
  setMoments?: SetMoment[];
  selectedSetMomentId?: string | null;
  onSelectSetMoment?: (id: string | null) => void;
  onAddSetMoment?: (timeMs: number) => void;
  onDeleteSetMoment?: (id: string) => void;
}

// Export ref type for external control
export interface SegmentTimelineRef {
  toggleZoom: () => void;
  fitToTimeline: () => void;
}

function formatTime(ms: number): string {
  const totalSeconds = ms / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export const SegmentTimeline = forwardRef<SegmentTimelineRef, SegmentTimelineProps>(({
  segments,
  videoDurationMs,
  currentTimeMs,
  selectedSegmentId,
  onSelectSegment,
  onSegmentBoundaryChange,
  onSeek,
  setMoments = [],
  selectedSetMomentId,
  onSelectSetMoment,
  onAddSetMoment,
  onDeleteSetMoment,
}, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [draggingBoundary, setDraggingBoundary] = useState<{
    segmentIndex: number;
    edge: "start" | "end";
  } | null>(null);

  // Zoom state: visibleRange defines what portion of the video is visible
  const [visibleRange, setVisibleRange] = useState<{ start: number; end: number }>({
    start: 0,
    end: videoDurationMs,
  });
  const [savedZoom, setSavedZoom] = useState<{ start: number; end: number } | null>(null);

  // Update visible range when video duration changes
  useEffect(() => {
    if (videoDurationMs > 0) {
      setVisibleRange({ start: 0, end: videoDurationMs });
    }
  }, [videoDurationMs]);

  // Sort segments by start time
  const sortedSegments = useMemo(
    () => [...segments].sort((a, b) => a.startMs - b.startMs),
    [segments]
  );

  // Calculate visible duration
  const visibleDuration = visibleRange.end - visibleRange.start;

  // Convert pixel position to time (within visible range)
  const pixelToTime = useCallback(
    (pixelX: number): number => {
      if (!containerRef.current || visibleDuration <= 0) return visibleRange.start;
      const rect = containerRef.current.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, pixelX / rect.width));
      return visibleRange.start + ratio * visibleDuration;
    },
    [visibleRange, visibleDuration]
  );

  // Fit all segments to timeline
  const fitToTimeline = useCallback(() => {
    setVisibleRange({ start: 0, end: videoDurationMs });
  }, [videoDurationMs]);

  // Toggle between saved zoom and fit-all
  const toggleZoom = useCallback(() => {
    const isFittedToAll = visibleRange.start === 0 && visibleRange.end === videoDurationMs;

    if (isFittedToAll && savedZoom) {
      // Restore saved zoom
      setVisibleRange(savedZoom);
    } else if (!isFittedToAll) {
      // Save current zoom and fit to all
      setSavedZoom({ ...visibleRange });
      fitToTimeline();
    }
  }, [visibleRange, savedZoom, videoDurationMs, fitToTimeline]);

  // Expose methods via ref
  useImperativeHandle(ref, () => ({
    toggleZoom,
    fitToTimeline,
  }), [toggleZoom, fitToTimeline]);

  // Track if mouse is over the timeline for wheel zoom
  const [isMouseOver, setIsMouseOver] = useState(false);

  // Handle zoom (Cmd+scroll) and horizontal pan (Shift+scroll / horizontal swipe)
  useEffect(() => {
    if (!isMouseOver) return;

    const handleWheel = (e: WheelEvent) => {
      const isZoomGesture = e.metaKey || e.ctrlKey;
      // On macOS, Shift+scroll is converted to deltaX by the OS,
      // so detect horizontal scroll directly instead of checking shiftKey.
      const hasHorizontalScroll = Math.abs(e.deltaX) > Math.abs(e.deltaY);

      if (!isZoomGesture && !hasHorizontalScroll) return;

      const container = containerRef.current;
      if (!container) return;

      e.preventDefault();

      const currentVisibleDuration = visibleRange.end - visibleRange.start;

      if (hasHorizontalScroll && !isZoomGesture) {
        // Horizontal scroll: pan timeline
        const panAmount = (e.deltaX / 1000) * currentVisibleDuration;
        let newStart = visibleRange.start + panAmount;
        let newEnd = visibleRange.end + panAmount;

        // Clamp to video bounds
        if (newStart < 0) {
          newStart = 0;
          newEnd = currentVisibleDuration;
        } else if (newEnd > videoDurationMs) {
          newEnd = videoDurationMs;
          newStart = videoDurationMs - currentVisibleDuration;
        }

        setVisibleRange({ start: newStart, end: newEnd });
        return;
      }

      // Cmd/Ctrl+scroll: zoom
      const rect = container.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseRatio = Math.max(0, Math.min(1, mouseX / rect.width));
      const mouseTime = visibleRange.start + mouseRatio * currentVisibleDuration;

      const zoomIn = e.deltaY < 0;
      const factor = zoomIn ? 1 / ZOOM_FACTOR : ZOOM_FACTOR;
      let newDuration = currentVisibleDuration * factor;

      // Clamp duration
      newDuration = Math.max(MIN_VISIBLE_RANGE_MS, Math.min(videoDurationMs, newDuration));

      // Calculate new range centered on mouse position
      const newStart = mouseTime - mouseRatio * newDuration;
      const newEnd = newStart + newDuration;

      // Clamp to video bounds
      if (newStart < 0) {
        setVisibleRange({ start: 0, end: Math.min(newDuration, videoDurationMs) });
      } else if (newEnd > videoDurationMs) {
        setVisibleRange({ start: Math.max(0, videoDurationMs - newDuration), end: videoDurationMs });
      } else {
        setVisibleRange({ start: newStart, end: newEnd });
      }
    };

    window.addEventListener("wheel", handleWheel, { passive: false });
    return () => window.removeEventListener("wheel", handleWheel);
  }, [isMouseOver, visibleRange, videoDurationMs]);

  // Playhead dragging state
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);

  // Handle playhead drag
  useEffect(() => {
    if (!isDraggingPlayhead) return;

    const handleMouseMove = (e: MouseEvent) => {
      const container = containerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const localX = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, localX / rect.width));
      const timeMs = visibleRange.start + ratio * (visibleRange.end - visibleRange.start);
      onSeek(timeMs);
    };

    const handleMouseUp = () => {
      setIsDraggingPlayhead(false);
      document.body.style.cursor = "";
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    document.body.style.cursor = "ew-resize";

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
    };
  }, [isDraggingPlayhead, visibleRange, onSeek]);


  // Handle boundary drag
  useEffect(() => {
    if (!draggingBoundary) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const localX = e.clientX - rect.left;
      const newTimeMs = pixelToTime(localX);

      const { segmentIndex, edge } = draggingBoundary;
      const segment = sortedSegments[segmentIndex];
      if (!segment) return;

      if (edge === "start") {
        // Dragging the start edge - affects this segment and the previous one
        const prevSegment = sortedSegments[segmentIndex - 1];
        if (!prevSegment) return;

        // Clamp: can't go before prev segment's start + min duration
        // and can't go after this segment's end - min duration
        const minTime = prevSegment.startMs + MIN_SEGMENT_DURATION_MS;
        const maxTime = segment.endMs - MIN_SEGMENT_DURATION_MS;
        const clampedTime = Math.max(minTime, Math.min(maxTime, newTimeMs));

        // Update both segments
        onSegmentBoundaryChange(prevSegment.id, prevSegment.startMs, clampedTime);
        onSegmentBoundaryChange(segment.id, clampedTime, segment.endMs);
      } else {
        // Dragging the end edge - affects this segment and the next one
        const nextSegment = sortedSegments[segmentIndex + 1];
        if (!nextSegment) return;

        // Clamp: can't go before this segment's start + min duration
        // and can't go after next segment's end - min duration
        const minTime = segment.startMs + MIN_SEGMENT_DURATION_MS;
        const maxTime = nextSegment.endMs - MIN_SEGMENT_DURATION_MS;
        const clampedTime = Math.max(minTime, Math.min(maxTime, newTimeMs));

        // Update both segments
        onSegmentBoundaryChange(segment.id, segment.startMs, clampedTime);
        onSegmentBoundaryChange(nextSegment.id, clampedTime, nextSegment.endMs);
      }
    };

    const handleMouseUp = () => {
      setDraggingBoundary(null);
      document.body.style.cursor = "";
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    document.body.style.cursor = "col-resize";

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
    };
  }, [draggingBoundary, sortedSegments, pixelToTime, onSegmentBoundaryChange]);

  // Handle click on timeline (seek)
  const handleTimelineClick = useCallback(
    (e: React.MouseEvent) => {
      if (draggingBoundary) return;
      if (!containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const localX = e.clientX - rect.left;
      const timeMs = pixelToTime(localX);
      onSeek(timeMs);
    },
    [draggingBoundary, pixelToTime, onSeek]
  );

  // Calculate playhead position within visible range
  const playheadPosition = useMemo(() => {
    if (visibleDuration <= 0) return -100; // Off screen
    if (currentTimeMs < visibleRange.start || currentTimeMs > visibleRange.end) {
      return -100; // Off screen
    }
    return ((currentTimeMs - visibleRange.start) / visibleDuration) * 100;
  }, [currentTimeMs, visibleRange, visibleDuration]);

  // Generate time markers for visible range
  const timeMarkers = useMemo(() => {
    const markerCount = 5;
    const markers = [];
    for (let i = 0; i <= markerCount; i++) {
      const time = visibleRange.start + (visibleDuration * i) / markerCount;
      markers.push(formatTime(time));
    }
    return markers;
  }, [visibleRange, visibleDuration]);

  // Check if zoomed in
  const isZoomed = visibleRange.start > 0 || visibleRange.end < videoDurationMs;

  if (videoDurationMs <= 0 || segments.length === 0) {
    return (
      <div className="h-16 bg-[#0a0a0a] rounded-lg flex items-center justify-center text-slate-500 text-sm">
        No segments loaded
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {/* Timeline header + segment track wrapper (shared for playhead) */}
      <div className="relative">
        {/* Timeline header with time markers */}
        <div className="flex justify-between text-[10px] text-slate-500 px-1 pb-1">
          {timeMarkers.map((marker, i) => (
            <span key={i}>{marker}</span>
          ))}
        </div>

        {/* Segment track */}
        <div
          ref={containerRef}
          className="relative h-12 bg-[#0a0a0a] rounded-lg overflow-hidden cursor-pointer"
          onClick={handleTimelineClick}
          onMouseEnter={() => setIsMouseOver(true)}
          onMouseLeave={() => setIsMouseOver(false)}
        >
          {/* Segments */}
          {sortedSegments.map((segment, index) => {
            // Calculate position relative to visible range
            const segmentStart = Math.max(segment.startMs, visibleRange.start);
            const segmentEnd = Math.min(segment.endMs, visibleRange.end);

            // Skip if segment is outside visible range
            if (segmentEnd <= visibleRange.start || segmentStart >= visibleRange.end) {
              return null;
            }

            const leftPercent = ((segmentStart - visibleRange.start) / visibleDuration) * 100;
            const widthPercent = ((segmentEnd - segmentStart) / visibleDuration) * 100;

            const isSelected = segment.id === selectedSegmentId;
            const color = SEGMENT_COLORS[segment.type];
            const isFirst = index === 0;
            const isLast = index === sortedSegments.length - 1;

            return (
              <div
                key={segment.id}
                className={cn(
                  "absolute top-0 h-full flex items-center justify-center transition-opacity",
                  isSelected ? "opacity-100" : "opacity-80 hover:opacity-100"
                )}
                style={{
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                  backgroundColor: color,
                }}
                tabIndex={-1}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectSegment(segment.id);
                  (document.activeElement as HTMLElement)?.blur?.();
                }}
              >
                {/* Segment label */}
                <span
                  className={cn(
                    "text-[11px] font-medium text-white/90 truncate px-2 select-none",
                    widthPercent < 8 && "hidden"
                  )}
                >
                  {segment.type === "in-play"
                    ? `Rally ${segment.rallyNumber || ""}`
                    : "Break"}
                </span>

                {/* Selection indicator */}
                {isSelected && (
                  <div className="absolute inset-0 border-2 border-white rounded-sm pointer-events-none" />
                )}

                {/* Draggable left edge (not for first segment) */}
                {!isFirst && segment.startMs >= visibleRange.start && (
                  <div
                    className="absolute left-0 top-0 bottom-0 w-2 cursor-col-resize hover:bg-white/20 z-10"
                    tabIndex={-1}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                      setDraggingBoundary({ segmentIndex: index, edge: "start" });
                      (document.activeElement as HTMLElement)?.blur?.();
                    }}
                  />
                )}

                {/* Draggable right edge (not for last segment) */}
                {!isLast && segment.endMs <= visibleRange.end && (
                  <div
                    className="absolute right-0 top-0 bottom-0 w-2 cursor-col-resize hover:bg-white/20 z-10"
                    tabIndex={-1}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                      setDraggingBoundary({ segmentIndex: index, edge: "end" });
                      (document.activeElement as HTMLElement)?.blur?.();
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>

        {/* Playhead - spans from timecodes to bottom of segment track */}
        {playheadPosition >= 0 && playheadPosition <= 100 && (
          <div
            className="absolute top-0 bottom-0 w-4 z-20 cursor-ew-resize flex justify-center group"
            style={{ left: `calc(${playheadPosition}% - 8px)` }}
            tabIndex={-1}
            onMouseDown={(e) => {
              e.stopPropagation();
              e.preventDefault();
              setIsDraggingPlayhead(true);
              (document.activeElement as HTMLElement)?.blur?.();
            }}
          >
            <div className="w-0.5 h-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.5)] group-hover:bg-[#34B27B] group-hover:shadow-[0_0_12px_rgba(52,178,123,0.7)] transition-all" />
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2.5 h-2.5 bg-white rotate-45 rounded-sm group-hover:bg-[#34B27B] group-hover:scale-125 transition-all" />
          </div>
        )}
      </div>

      {/* Set moment track */}
      <div
        className="relative h-8 bg-[#0d0d0d] rounded-lg border border-white/5 cursor-crosshair overflow-hidden"
        title="Click to mark a set moment here (or press S at playhead)"
        onClick={(e) => {
          if (!onAddSetMoment || !containerRef.current) return;
          const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
          const localX = e.clientX - rect.left;
          const ratio = Math.max(0, Math.min(1, localX / rect.width));
          const timeMs = visibleRange.start + ratio * visibleDuration;
          onAddSetMoment(timeMs);
          onSelectSetMoment?.(null); // deselect after adding
        }}
      >
        {/* Row label */}
        <span className="absolute left-2 top-1/2 -translate-y-1/2 text-[9px] text-slate-600 font-semibold uppercase tracking-wider select-none pointer-events-none z-10">
          Sets
        </span>

        {/* Pins */}
        {setMoments.map((moment) => {
          if (moment.timeMs < visibleRange.start || moment.timeMs > visibleRange.end) return null;
          const pos = ((moment.timeMs - visibleRange.start) / visibleDuration) * 100;
          const isSelected = moment.id === selectedSetMomentId;
          return (
            <div
              key={moment.id}
              className="absolute top-0 bottom-0 flex flex-col items-center"
              style={{ left: `${pos}%`, transform: 'translateX(-50%)', zIndex: 10 }}
              onClick={(e) => {
                e.stopPropagation();
                onSelectSetMoment?.(isSelected ? null : moment.id);
              }}
              onDoubleClick={(e) => {
                e.stopPropagation();
                onDeleteSetMoment?.(moment.id);
              }}
              title={`Set at ${formatTime(moment.timeMs)} — click to select, double-click to delete`}
            >
              {/* Vertical stem */}
              <div
                className="w-[2px] flex-1 transition-colors"
                style={{ backgroundColor: isSelected ? SET_MOMENT_COLOR : `${SET_MOMENT_COLOR}80` }}
              />
              {/* Diamond cap */}
              <div
                className="w-3 h-3 rotate-45 rounded-[2px] flex-shrink-0 -mt-1.5 transition-all cursor-pointer"
                style={{
                  backgroundColor: SET_MOMENT_COLOR,
                  opacity: isSelected ? 1 : 0.7,
                  transform: `rotate(45deg) scale(${isSelected ? 1.3 : 1})`,
                }}
              />
            </div>
          );
        })}

        {/* Empty hint */}
        {setMoments.length === 0 && (
          <span className="absolute inset-0 flex items-center justify-center text-[9px] text-slate-700 select-none pointer-events-none">
            Press S at playhead or click here to mark a set
          </span>
        )}
      </div>

      {/* Legend and zoom info */}
      <div className="flex items-center justify-between text-[10px] text-slate-400">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: SEGMENT_COLORS["in-play"] }}
            />
            <span>In-Play (Rally)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: SEGMENT_COLORS["break"] }}
            />
            <span>Break</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div
              className="w-2.5 h-2.5 rotate-45 rounded-[2px]"
              style={{ backgroundColor: SET_MOMENT_COLOR }}
            />
            <span>Set Moment</span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-slate-500">
          <span>
            <kbd className="px-1 py-0.5 bg-white/5 border border-white/10 rounded text-[9px]">⌘</kbd>
            <span className="mx-0.5">+</span>
            <span>scroll to zoom</span>
          </span>
          <span className="text-white/20">|</span>
          <span>
            <kbd className="px-1 py-0.5 bg-white/5 border border-white/10 rounded text-[9px]">⇧</kbd>
            <span className="mx-0.5">+</span>
            <span>scroll to pan</span>
          </span>
          <span className="text-white/20">|</span>
          <span>
            <kbd className="px-1 py-0.5 bg-white/5 border border-white/10 rounded text-[9px]">Z</kbd>
            <span className="ml-1">{isZoomed ? "fit all" : "restore zoom"}</span>
          </span>
        </div>
      </div>
    </div>
  );
});
