import type { ScanProgress } from "../types";
import { PHASE_LABELS } from "../types";

interface ProgressBarProps {
  progress: ScanProgress;
}

export function ProgressBar({ progress }: ProgressBarProps) {
  const pct =
    progress.total > 0
      ? Math.min(100, Math.round((progress.current / progress.total) * 100))
      : 0;

  return (
    <div className="progress-bar">
      <div className="progress-head">
        <span className="progress-phase">
          {PHASE_LABELS[progress.phase]}
          {progress.detail ? ` · ${progress.detail}` : ""}
        </span>
        <span className="progress-count">
          {progress.current} / {progress.total}
        </span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="progress-foot">
        <span className="progress-backend">{progress.backend}</span>
        <span className="progress-file">
          {progress.current_file ?? ""}
        </span>
        <span className="progress-pct">{pct}%</span>
      </div>
    </div>
  );
}
