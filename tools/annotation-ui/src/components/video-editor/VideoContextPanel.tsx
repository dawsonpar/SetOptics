import { Film, Video, FileQuestion, CheckCircle2, Circle, FolderOpen, Play, FileText, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { VideoContext } from "./types";

interface VideoContextPanelProps {
  context: VideoContext;
  onLoadRallyEdit: () => void;
  onLoadCorrected: () => void;
  onLoadRaw: () => void;
  onRunDetection: () => void;
  onLoadFile: () => void;
  onStartBlank: () => void;
}

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}m ${s}s`;
}

export function VideoContextPanel({
  context,
  onLoadRallyEdit,
  onLoadCorrected,
  onLoadRaw,
  onRunDetection,
  onLoadFile,
  onStartBlank,
}: VideoContextPanelProps) {
  const { videoType, rallyEditMap, correctedAnnotation, rawAnnotation } = context;

  const TypeBadge = () => {
    if (videoType === 'rally-edit') {
      return (
        <div className="flex items-center gap-2 mb-3">
          <Film className="w-4 h-4 text-[#34B27B]" />
          <span className="text-sm font-medium text-[#34B27B]">Rally Edit</span>
        </div>
      );
    }
    if (videoType === 'raw-footage') {
      return (
        <div className="flex items-center gap-2 mb-3">
          <Video className="w-4 h-4 text-slate-300" />
          <span className="text-sm font-medium text-slate-300">Game Footage</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2 mb-3">
        <FileQuestion className="w-4 h-4 text-slate-400" />
        <span className="text-sm font-medium text-slate-400">Unknown</span>
      </div>
    );
  };

  return (
    <div className="mb-4 p-3 rounded-lg bg-white/5 border border-white/5 space-y-3">
      <div>
        <p className="text-xs text-slate-400 mb-2">Detected</p>
        <TypeBadge />

        {/* Available files */}
        <div className="space-y-1.5 mb-3">
          {rallyEditMap && (
            <div className="flex items-start gap-2 text-xs">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#34B27B] mt-0.5 flex-shrink-0" />
              <span className="text-slate-300">
                Chapter map · {rallyEditMap.segmentCount} rallies · {formatDuration(rallyEditMap.totalDurationSec)}
              </span>
            </div>
          )}
          {correctedAnnotation && (
            <div className="flex items-start gap-2 text-xs">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#34B27B] mt-0.5 flex-shrink-0" />
              <span className="text-slate-300">Corrected annotation</span>
            </div>
          )}
          {rawAnnotation && (
            <div className="flex items-start gap-2 text-xs">
              <Circle className="w-3.5 h-3.5 text-slate-500 mt-0.5 flex-shrink-0" />
              <span className="text-slate-400">Raw annotation</span>
            </div>
          )}
          {!rallyEditMap && !correctedAnnotation && !rawAnnotation && (
            <p className="text-xs text-slate-500">No annotation files found</p>
          )}
        </div>
      </div>

      <div className="border-t border-white/5 pt-3 space-y-1.5">
        {/* Primary action */}
        {videoType === 'rally-edit' && rallyEditMap && (
          <Button
            onClick={onLoadRallyEdit}
            size="sm"
            className="w-full gap-2 bg-[#34B27B] hover:bg-[#34B27B]/90 text-white"
          >
            <Film className="w-3.5 h-3.5" />
            Load Rally Segments
          </Button>
        )}

        {videoType !== 'rally-edit' && correctedAnnotation && (
          <Button
            onClick={onLoadCorrected}
            size="sm"
            className="w-full gap-2 bg-[#34B27B] hover:bg-[#34B27B]/90 text-white"
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            Load Corrected
          </Button>
        )}

        {/* Secondary actions */}
        {videoType !== 'rally-edit' && rawAnnotation && (
          <Button
            onClick={onLoadRaw}
            size="sm"
            variant="outline"
            className="w-full gap-2 border-white/10 text-slate-300 hover:bg-white/5"
          >
            <FileText className="w-3.5 h-3.5" />
            Load Raw{correctedAnnotation ? " (unreviewed)" : ""}
          </Button>
        )}

        {videoType !== 'rally-edit' && (
          <Button
            onClick={onRunDetection}
            size="sm"
            variant="outline"
            className="w-full gap-2 border-white/10 text-slate-300 hover:bg-white/5"
          >
            <Play className="w-3.5 h-3.5" />
            Run Detection
          </Button>
        )}

        <Button
          onClick={onStartBlank}
          size="sm"
          variant="outline"
          className="w-full gap-2 border-white/10 text-slate-300 hover:bg-white/5"
        >
          <Plus className="w-3.5 h-3.5" />
          Start from scratch
        </Button>

        <Button
          onClick={onLoadFile}
          size="sm"
          variant="ghost"
          className="w-full gap-2 text-slate-400 hover:text-slate-200 hover:bg-white/5"
        >
          <FolderOpen className="w-3.5 h-3.5" />
          Load File…
        </Button>
      </div>
    </div>
  );
}
