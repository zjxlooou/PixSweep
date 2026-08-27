import { open } from "@tauri-apps/plugin-dialog";

interface ToolbarProps {
  folders: string[];
  onAddFolders: (folders: string[]) => void;
  onRemoveFolder: (folder: string) => void;
  onStartScan: () => void;
  imageCount?: number | null;
  onOpenSettings: () => void;
  onOpenScoreHelp: () => void;
  onOpenTrashBin: () => void;
  scanning: boolean;
  aiEnabled: boolean;
}

export function Toolbar({
  folders,
  onAddFolders,
  onRemoveFolder,
  onStartScan,
  imageCount,
  onOpenSettings,
  onOpenScoreHelp,
  onOpenTrashBin,
  scanning,
  aiEnabled,
}: ToolbarProps) {
  const handleAdd = async () => {
    const selected = await open({
      directory: true,
      multiple: true,
      title: "选择要扫描的文件夹",
    });
    if (selected) {
      const dirs = Array.isArray(selected) ? selected : [selected];
      onAddFolders(dirs.filter((d): d is string => !!d));
    }
  };

  return (
    <header className="toolbar">
      <div className="brand">
        <div className="brand-logo">PS</div>
        <div className="brand-text">
          <span className="brand-name">PixSweep</span>
          <span className="brand-sub">智能图片去重</span>
        </div>
      </div>

      <div className="folder-pills">
        {folders.length === 0 && (
          <span className="folder-empty">尚未选择文件夹</span>
        )}
        {folders.map((f) => (
          <span key={f} className="folder-pill" title={f}>
            {shortenPath(f)}
            <button
              className="pill-remove"
              onClick={() => onRemoveFolder(f)}
              title="移除"
              disabled={scanning}
            >
              ×
            </button>
          </span>
        ))}
      </div>

      <div className="toolbar-actions">
        {imageCount !== null && imageCount !== undefined && (
          <span className="folder-count">{imageCount} 张</span>
        )}
        <button className="btn" onClick={handleAdd} disabled={scanning}>
          + 添加文件夹
        </button>
        <button
          className="btn btn-primary"
          onClick={onStartScan}
          disabled={scanning || folders.length === 0}
        >
          {scanning ? "扫描中…" : "开始扫描"}
        </button>
        <button className="btn btn-ghost" onClick={onOpenTrashBin} title="查看临时文件夹并恢复">
          临时文件夹
        </button>
        <button className="btn btn-ghost" onClick={onOpenScoreHelp} title="查看评分标准说明">
          评分标准
        </button>
        <button className="btn btn-ghost" onClick={onOpenSettings} title="设置">
          设置
        </button>
        <span
          className={`ai-badge ${aiEnabled ? "on" : "off"}`}
          title={aiEnabled ? "AI 推理已启用" : "AI 推理未启用"}
        >
          AI
        </span>
      </div>
    </header>
  );
}

function shortenPath(path: string): string {
  const parts = path.split(/[\\/]/);
  if (parts.length <= 2) return path;
  return `.../${parts[parts.length - 1]}`;
}
