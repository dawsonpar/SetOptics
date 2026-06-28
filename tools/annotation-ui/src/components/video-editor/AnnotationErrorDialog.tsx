import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { AlertTriangle, RotateCcw, Play, FileText, X } from "lucide-react";

interface AnnotationErrorDialogProps {
  isOpen: boolean;
  onClose: () => void;
  error: string;
  lastBatch: number;
  output: string;
  onRetry: () => void;
  onResume: () => void;
}

export function AnnotationErrorDialog({
  isOpen,
  onClose,
  error,
  lastBatch,
  output,
  onRetry,
  onResume,
}: AnnotationErrorDialogProps) {
  const [showOutput, setShowOutput] = useState(false);

  const canResume = lastBatch > 0;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-[#09090b] border-white/10 text-slate-200 max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-red-400">
            <AlertTriangle className="w-5 h-5" />
            Annotation Failed
          </DialogTitle>
          <DialogDescription className="text-slate-400">
            An error occurred while analyzing the video.
          </DialogDescription>
        </DialogHeader>

        <div className="py-4">
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 mb-4">
            <p className="text-red-400 text-sm font-mono break-words select-text cursor-text">
              {error}
            </p>
          </div>

          {canResume && (
            <p className="text-slate-400 text-sm">
              Progress was saved at batch {lastBatch}. You can resume from where it stopped.
            </p>
          )}
        </div>

        {showOutput && (
          <div className="mb-4">
            <p className="text-sm text-slate-400 mb-2">Full Output:</p>
            <div className="bg-black/50 border border-white/10 rounded-lg p-3 max-h-48 overflow-y-auto">
              <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap select-text cursor-text">
                {output || "No output captured"}
              </pre>
            </div>
          </div>
        )}

        <DialogFooter className="flex-col sm:flex-row gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowOutput(!showOutput)}
            className="text-slate-400 hover:text-slate-200"
          >
            <FileText className="w-4 h-4 mr-2" />
            {showOutput ? "Hide" : "Show"} Output
          </Button>

          <div className="flex gap-2 ml-auto">
            <Button
              variant="outline"
              onClick={onClose}
              className="border-white/20 text-slate-300 hover:bg-white/10"
            >
              <X className="w-4 h-4 mr-2" />
              Cancel
            </Button>

            {canResume && (
              <Button
                variant="outline"
                onClick={onResume}
                className="border-[#34B27B]/50 text-[#34B27B] hover:bg-[#34B27B]/10"
              >
                <Play className="w-4 h-4 mr-2" />
                Resume from Batch {lastBatch}
              </Button>
            )}

            <Button
              onClick={onRetry}
              className="bg-[#34B27B] hover:bg-[#34B27B]/90 text-white"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Retry
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
