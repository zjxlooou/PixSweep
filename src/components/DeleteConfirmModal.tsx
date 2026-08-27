import { useEffect } from "react";
import { formatBytes } from "../types";

interface DeleteConfirmModalProps {
  paths: string[];
  totalBytes: number;
  permanent: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function DeleteConfirmModal({
  paths,
  totalBytes,
  permanent,
  onConfirm,
  onCancel,
}: DeleteConfirmModalProps) {
  const modeText = permanent ? "永久删除" : "移至临时文件夹";

  // 键盘操作：Y/Enter = 确认，N/Esc = 取消
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key === "y" || key === "enter") {
        e.preventDefault();
        onConfirm();
      } else if (key === "n" || key === "escape") {
        e.preventDefault();
        onCancel();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onConfirm, onCancel]);

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2 className="modal-title">确认{modeText}</h2>
        <p className="modal-desc">
          即将{modeText} <strong>{paths.length}</strong> 个文件，共{" "}
          <strong>{formatBytes(totalBytes)}</strong>。
          {!permanent && "文件将进入临时文件夹，可随时恢复。"}
          {permanent && "此操作不可逆！"}
        </p>
        <div className="modal-list">
          {paths.slice(0, 8).map((p) => (
            <div key={p} className="modal-list-item" title={p}>
              {p}
            </div>
          ))}
          {paths.length > 8 && (
            <div className="modal-list-more">
              ... 以及另外 {paths.length - 8} 个文件
            </div>
          )}
        </div>
        <div className="modal-actions">
          <button className="btn" onClick={onCancel}>
            取消 <kbd>N</kbd>
          </button>
          <button className="btn btn-danger" onClick={onConfirm}>
            确认{modeText} <kbd>Y</kbd>
          </button>
        </div>
        <div className="modal-kbd-hint">
          <kbd>Y</kbd> 确认 · <kbd>N</kbd> 取消 · <kbd>Enter</kbd> 确认 ·{" "}
          <kbd>Esc</kbd> 取消
        </div>
      </div>
    </div>
  );
}
