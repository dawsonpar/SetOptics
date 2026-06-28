import { Button } from "@/components/ui/button";
import { Loader2, X, Search } from "lucide-react";

interface ProcessingOverlayProps {
  mode: 'checking' | 'processing';
  currentBatch: number;
  totalBatches: number;
  percentage: number;
  onCancel: () => void;
}

export function ProcessingOverlay({
  mode,
  currentBatch,
  totalBatches,
  percentage,
  onCancel,
}: ProcessingOverlayProps) {
  const isChecking = mode === 'checking';

  return (
    <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-10 rounded-2xl">
      <div className="text-center">
        <div className="w-20 h-20 mx-auto mb-6 relative">
          {isChecking ? (
            <Search className="w-20 h-20 text-[#34B27B] animate-pulse" />
          ) : (
            <>
              <Loader2 className="w-20 h-20 text-[#34B27B] animate-spin" />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-lg font-bold text-white">{percentage}%</span>
              </div>
            </>
          )}
        </div>

        <h3 className="text-lg font-semibold text-white mb-2">
          {isChecking ? 'Checking for Annotations' : 'Analyzing Video'}
        </h3>

        <p className="text-slate-400 mb-1">
          {isChecking
            ? 'Looking for existing analysis...'
            : 'Detecting rally segments with AI...'}
        </p>

        {!isChecking && (
          <p className="text-slate-500 text-sm mb-6">
            {totalBatches > 0 ? (
              <>Batch {currentBatch} of {totalBatches}</>
            ) : (
              <>Preparing batches...</>
            )}
          </p>
        )}

        {isChecking && <div className="mb-6" />}

        {/* Progress bar - only show during processing */}
        {!isChecking && (
          <div className="w-64 mx-auto mb-6">
            <div className="h-2 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-[#34B27B] transition-all duration-300 ease-out"
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
        )}

        <Button
          onClick={onCancel}
          variant="outline"
          size="sm"
          className="gap-2 border-white/20 text-slate-300 hover:bg-white/10 hover:text-white"
        >
          <X className="w-4 h-4" />
          Cancel
        </Button>
      </div>
    </div>
  );
}
