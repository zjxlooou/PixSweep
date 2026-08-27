import type { ScanResult } from "../types";
import { formatBytes } from "../types";

interface StatsBarProps {
  result: ScanResult | null;
  onDeleteAll: () => void;
  onExportReport: () => void;
  disabled: boolean;
}

export function StatsBar({
  result,
  onDeleteAll,
  onExportReport,
  disabled,
}: StatsBarProps) {
  if (!result) return null;

  const totalDeleteCount = result.groups.reduce(
    (sum, g) => sum + g.images.filter((i) => !i.recommended).length,
    0,
  );

  return (
    <footer className="stats-bar">
      <div className="stats-left">
        <div className="stat">
          <span className="stat-label">扫描图片</span>
          <span className="stat-value">{result.total_images}</span>
        </div>
        <div className="stat">
          <span className="stat-label">相似分组</span>
          <span className="stat-value">{result.groups.length}</span>
        </div>
        <div className="stat">
          <span className="stat-label">可删图片</span>
          <span className="stat-value">{totalDeleteCount}</span>
        </div>
        <div className="stat">
          <span className="stat-label">可节省空间</span>
          <span className="stat-value highlight">
            {formatBytes(result.total_reclaimable_bytes)}
          </span>
        </div>
      </div>
      <div className="stats-actions">
        <button className="btn" onClick={onExportReport} disabled={disabled}>
          导出报告
        </button>
        <button
          className="btn btn-danger"
          onClick={onDeleteAll}
          disabled={disabled || totalDeleteCount === 0}
        >
          一键删除全部
        </button>
      </div>
    </footer>
  );
}
