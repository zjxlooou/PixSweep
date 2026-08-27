/**
 * 临时文件夹面板（应用内置隔离区）。
 * - 列出已删除的图片
 * - 单张恢复 / 批量恢复
 * - 清空临时文件夹
 * - 在系统文件管理器中打开隔离区目录
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { api } from "../api";
import type { TrashImage } from "../types";
import { formatBytes } from "../types";

interface TrashBinModalProps {
  onClose: () => void;
  /** 恢复成功时回调，传入恢复的原始路径列表 */
  onRestored: (paths: string[]) => void;
}

export function TrashBinModal({ onClose, onRestored }: TrashBinModalProps) {
  const [items, setItems] = useState<TrashImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [restoring, setRestoring] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scanCount, setScanCount] = useState(0);
  const inflightRef = useRef(false);

  // 触发异步扫描：立即返回，通过 trashbin-done 事件回调结果
  const refresh = useCallback(() => {
    if (inflightRef.current) return;
    inflightRef.current = true;
    setLoading(true);
    setError(null);
    setScanCount(0);
    api.listTrashImages().catch((e) => {
      inflightRef.current = false;
      setError(String(e));
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    let mounted = true;
    let unProgress: UnlistenFn | undefined;
    let unDone: UnlistenFn | undefined;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    // 先注册监听器，再触发扫描，避免 race（emit 比 listen 早到达）
    (async () => {
      unProgress = await listen<number>("trashbin-progress", (event) => {
        if (mounted) setScanCount(event.payload);
      });
      unDone = await listen<TrashImage[]>("trashbin-done", (event) => {
        if (!mounted) return;
        inflightRef.current = false;
        setItems(event.payload);
        setLoading(false);
        if (timeoutId) clearTimeout(timeoutId);
      });
      if (mounted) refresh();
    })();

    // 兜底超时：8 秒内没收到 trashbin-done 则强制重置状态
    timeoutId = setTimeout(() => {
      if (!mounted) return;
      if (inflightRef.current) {
        inflightRef.current = false;
        setLoading(false);
        setError("扫描超时（后端未响应），请重试");
      }
    }, 8000);

    return () => {
      mounted = false;
      if (timeoutId) clearTimeout(timeoutId);
      unProgress?.();
      unDone?.();
    };
  }, [refresh]);

  // 单张恢复：立刻从列表移除（不等 toast 反馈），失败则放回
  const handleRestore = useCallback(
    async (id: string, originalPath: string) => {
      setBusyId(id);
      setError(null);
      // 乐观更新：立即从列表中移除（同步显示）
      const removed = items.find((it) => it.id === id);
      setItems((prev) => prev.filter((it) => it.id !== id));
      try {
        const restoredPath = await api.restoreTrashItem(id);
        onRestored([restoredPath]);
      } catch (e) {
        // 失败则放回列表
        if (removed) {
          setItems((prev) => [removed, ...prev]);
        }
        setError(`恢复失败: ${originalPath}\n${e}`);
      } finally {
        setBusyId(null);
      }
    },
    [items, onRestored],
  );

  // 全部恢复
  const handleRestoreAll = useCallback(async () => {
    setRestoring(true);
    setError(null);
    const snapshot = items;
    setItems([]); // 乐观清空
    try {
      const paths = await api.restoreAllTrashImages();
      onRestored(paths);
    } catch (e) {
      // 失败则恢复列表
      setItems(snapshot);
      setError(String(e));
    } finally {
      setRestoring(false);
    }
  }, [items, onRestored]);

  // 清空临时文件夹
  const handleClear = useCallback(async () => {
    if (items.length === 0) return;
    if (
      !confirm(
        `确定清空临时文件夹？\n\n将永久删除 ${items.length} 张照片，无法恢复。\n（这不是系统回收站，清空后无法找回）`,
      )
    ) {
      return;
    }
    setClearing(true);
    setError(null);
    try {
      await api.clearTrashBin();
      setItems([]);
    } catch (e) {
      setError(String(e));
    } finally {
      setClearing(false);
    }
  }, [items]);

  // 在系统文件管理器中打开隔离区目录
  const handleOpenDir = useCallback(async () => {
    try {
      await api.openTrashBinInExplorer();
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const totalBytes = items.reduce((s, it) => s + it.size, 0);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal modal-wide trashbin-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-title">
          临时文件夹（{items.length} 张 · 共 {formatBytes(totalBytes)}）
          <div className="modal-title-actions">
            <button
              className="btn btn-sm"
              onClick={handleOpenDir}
              title="在文件管理器中打开隔离区目录"
            >
              打开目录
            </button>
            <button
              className="btn btn-sm"
              onClick={handleRestoreAll}
              disabled={items.length === 0 || restoring}
              title="恢复所有图片到原位置"
            >
              {restoring ? "恢复中…" : "全部恢复"}
            </button>
            <button
              className="btn btn-sm btn-danger"
              onClick={handleClear}
              disabled={items.length === 0 || clearing}
              title="永久删除隔离区所有文件"
            >
              {clearing ? "清空中…" : "清空"}
            </button>
            <button
              className="score-help-close"
              onClick={onClose}
              title="关闭"
              aria-label="关闭"
            >
              ×
            </button>
          </div>
        </div>

        {error && <div className="error-banner">{error}</div>}

        {loading ? (
          <div className="trashbin-loading">
            扫描隔离区…
            {scanCount > 0 && (
              <span className="trashbin-scan-count">已发现 {scanCount} 条</span>
            )}
          </div>
        ) : items.length === 0 ? (
          <div className="trashbin-empty">
            临时文件夹中没有图片。
            <br />
            在 PixSweep 中"一键删除"的图片会保留在这里，可以随时恢复。
          </div>
        ) : (
          <div className="trashbin-list">
            {items.map((it) => (
              <div key={it.id} className="trashbin-item">
                <div className="trashbin-info">
                  <div className="trashbin-name" title={it.original_path}>
                    {it.original_path.split(/[\\/]/).pop()}
                  </div>
                  <div className="trashbin-path" title={it.original_path}>
                    {it.original_path}
                  </div>
                  <div className="trashbin-meta">
                    {formatBytes(it.size)} · 删除于{" "}
                    {new Date(it.deleted_at * 1000).toLocaleString("zh-CN")}
                  </div>
                </div>
                <button
                  className="btn btn-sm"
                  onClick={() => handleRestore(it.id, it.original_path)}
                  disabled={busyId === it.id}
                >
                  {busyId === it.id ? "恢复中…" : "恢复"}
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="modal-actions">
          <button className="btn" onClick={onClose}>
            关闭
          </button>
        </div>
      </div>
    </div>
  );
}