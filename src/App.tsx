import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { useScan } from "./hooks/useScan";
import { useDelete } from "./hooks/useDelete";
import type { AppSettings, McpStatus, ScanResult, SystemInfo } from "./types";
import { formatBytes } from "./types";
import { Toolbar } from "./components/Toolbar";
import { ProgressBar } from "./components/ProgressBar";
import { GroupCard } from "./components/GroupCard";
import { StatsBar } from "./components/StatsBar";
import { DeleteConfirmModal } from "./components/DeleteConfirmModal";
import { SettingsPanel } from "./components/SettingsPanel";
import { PreviewModal } from "./components/PreviewModal";
import { ScoreHelpModal } from "./components/ScoreHelpModal";
import { TrashBinModal } from "./components/TrashBinModal";
import { save } from "@tauri-apps/plugin-dialog";
import { writeTextFile } from "@tauri-apps/plugin-fs";

interface PendingDelete {
  paths: string[];
  totalBytes: number;
}

// 预览状态：记录从哪个组、哪张图进入（用 path 定位图片，避免排序后索引错位）
interface PreviewState {
  groupIndex: number;
  imagePath: string;
}

// 滑动窗口参数（避免爆内存：DOM 中始终保持固定数量的分组）
const PAGE_SIZE = 30;
/** 窗口内最多同时渲染的分组数（上限，防止 DOM 膨胀） */
const RENDER_WINDOW = 120;

export default function App() {
  const [folders, setFolders] = useState<string[]>([]);
  const [imageCount, setImageCount] = useState<number | null>(null);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [mcpStatus, setMcpStatus] = useState<McpStatus | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showScoreHelp, setShowScoreHelp] = useState(false);
  const [showTrashBin, setShowTrashBin] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null);
  const [preview, setPreview] = useState<PreviewState | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  // 滑动窗口起点：只渲染 [windowStart, windowStart+RENDER_WINDOW) 的分组，
  // 向上/向下滚动时窗口平移，DOM 总数恒定（避免爆内存）
  const [windowStart, setWindowStart] = useState(0);
  // 滚动容器（main.content）与哨兵引用
  const contentRef = useRef<HTMLElement | null>(null);
  const topSentinelRef = useRef<HTMLDivElement | null>(null);
  const bottomSentinelRef = useRef<HTMLDivElement | null>(null);
  // 窗口平移前的锚点补偿量（滚动位置保持）
  const scrollAnchorRef = useRef<number | null>(null);

  const { state: scanState, startScan } = useScan();
  const { state: deleteState, deleteFiles, consumeResult } = useDelete();

  // 加载设置与系统信息
  useEffect(() => {
    api.getSettings().then(setSettings).catch(console.error);
    api.getSystemInfo().then(setSystemInfo).catch(console.error);
    api.getMcpStatus().then(setMcpStatus).catch(console.error);
  }, []);

  // 同步扫描结果
  useEffect(() => {
    if (scanState.result) {
      setResult(scanState.result);
      setWindowStart(0); // 新扫描结果重置窗口
    }
  }, [scanState.result]);

  // 双哨兵滑动窗口：顶部/底部哨兵进入视口时窗口平移，
  // 保持 DOM 中分组数恒定（≤ RENDER_WINDOW），避免爆内存
  useEffect(() => {
    if (!result) return;
    const total = result.groups.length;
    if (total === 0) return;

    const doScroll = (dir: -1 | 1) => {
      const el = contentRef.current;
      // 记录平移前第一个可见组的偏移（用于滚动补偿）
      if (el) {
        const visible = el.querySelector<HTMLElement>("[data-group-index]");
        if (visible) {
          const rect = visible.getBoundingClientRect();
          const contentRect = el.getBoundingClientRect();
          scrollAnchorRef.current = rect.top - contentRect.top + el.scrollTop;
        }
      }
      setWindowStart((prev) => {
        const maxStart = Math.max(0, total - RENDER_WINDOW);
        if (dir === 1) return Math.min(prev + PAGE_SIZE, maxStart);
        return Math.max(prev - PAGE_SIZE, 0);
      });
    };

    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          if (e.target === bottomSentinelRef.current) doScroll(1);
          else if (e.target === topSentinelRef.current) doScroll(-1);
        }
      },
      { root: contentRef.current, rootMargin: "240px" }, // 提前 240px 预加载
    );

    if (topSentinelRef.current) observer.observe(topSentinelRef.current);
    if (bottomSentinelRef.current) observer.observe(bottomSentinelRef.current);
    return () => observer.disconnect();
  }, [result, windowStart]);

  // 窗口平移后滚动位置补偿：把"平移前的第一个可见组"恢复到原视口位置。
  useEffect(() => {
    const el = contentRef.current;
    if (!el || scrollAnchorRef.current === null) return;
    const targetTop = scrollAnchorRef.current;
    scrollAnchorRef.current = null;
    // 在 DOM 更新后（下一帧）找到同索引分组，滚动到目标位置
    requestAnimationFrame(() => {
      // 新窗口第一组可能是旧的第 30 组（向下滚）——用 data-group-index 精确匹配
      const prevStart = windowStart - PAGE_SIZE;
      const anchor = el.querySelector<HTMLElement>(
        `[data-group-index="${prevStart}"]`,
      );
      if (!anchor) return;
      const rect = anchor.getBoundingClientRect();
      const contentRect = el.getBoundingClientRect();
      const currentTop = rect.top - contentRect.top + el.scrollTop;
      el.scrollTop += targetTop - currentTop;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [windowStart]);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  }, []);

  const addFolders = useCallback((dirs: string[]) => {
    setFolders((prev) => {
      const merged = new Set([...prev, ...dirs]);
      return Array.from(merged);
    });
  }, []);

  const removeFolder = useCallback((folder: string) => {
    setFolders((prev) => prev.filter((f) => f !== folder));
  }, []);

  // 全局键盘操作：Esc 逐层关闭弹窗（按"最上层优先"顺序）
  // 删除确认框 / 预览内部的 Esc 由组件自身处理（优先级更高），
  // 这里兜底处理其他弹窗，避免遗漏。
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // 删除确认框在预览之上时，预览组件已处理 Esc——这里只关非删除确认的弹窗
      setPreview((p) => {
        if (p) return null; // 预览自己也有 Esc，双保险
        return p;
      });
      setShowSettings((v) => {
        if (v) return false;
        return v;
      });
      setShowScoreHelp((v) => {
        if (v) return false;
        return v;
      });
      setShowTrashBin((v) => {
        if (v) return false;
        return v;
      });
      setPendingDelete((p) => {
        if (p) return null;
        return p;
      });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // 切换 MCP 服务（热启停 + 持久化设置）
  const handleToggleMcp = useCallback(
    async (enabled: boolean) => {
      try {
        const status = await api.setMcpEnabled(enabled);
        setMcpStatus(status);
        setSettings((s) => (s ? { ...s, mcp_enabled: enabled } : s));
        showToast(enabled ? "MCP 服务已启动" : "MCP 服务已停止");
      } catch (e) {
        showToast(`切换 MCP 失败: ${e}`);
      }
    },
    [showToast],
  );

  // 选中的文件夹变化 → 快速统计图片总数（显示在"添加文件夹"按钮左侧）
  useEffect(() => {
    let cancelled = false;
    if (folders.length === 0) {
      setImageCount(null);
      return;
    }
    api
      .countImages(folders)
      .then((n) => !cancelled && setImageCount(n))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [folders]);

  const handleStartScan = useCallback(async () => {
    if (folders.length === 0) return;
    await startScan(
      folders,
      settings?.similarity_threshold,
      settings?.incremental ?? true,
    );
  }, [folders, settings, startScan]);

  const handleSaveSettings = useCallback(async (s: AppSettings) => {
    setSettings(s);
    try {
      await api.setSettings(s);
    } catch (e) {
      console.error(e);
    }
  }, []);

  // 组级批量删除（直接删除，无确认弹窗）—— 主界面"删除 N 张" / 右键"删除其他"
  const handleGroupDeleteDirect = useCallback(
    async (paths: string[]) => {
      try {
        await deleteFiles(paths, settings?.permanent_delete ?? false);
      } catch (e) {
        showToast(`删除失败: ${e}`);
      }
    },
    [deleteFiles, settings, showToast],
  );

  // 打开预览（imagePath 为被双击图片的绝对路径，唯一标识）
  const handleOpenPreview = useCallback(
    (groupIndex: number, imagePath: string) => {
      setPreview({ groupIndex, imagePath });
    },
    [],
  );

  // 预览中删除（从预览的组删除非推荐照片，带确认弹窗）
  const handlePreviewDelete = useCallback(
    (paths: string[]) => {
      const bytes = collectBytes(result, paths);
      setPendingDelete({ paths, totalBytes: bytes });
    },
    [result],
  );

  // 单条删除（无确认）— 主界面 X 按钮 + 右键菜单 + 预览按钮 都走这里
  const handleDeleteOne = useCallback(
    async (path: string) => {
      try {
        await deleteFiles([path], settings?.permanent_delete ?? false);
        // 实际删除结果通过 deleteState.result effect 处理
      } catch (e) {
        showToast(`删除失败: ${e}`);
      }
    },
    [deleteFiles, settings, showToast],
  );

  // 一键删除全部
  const handleDeleteAll = useCallback(() => {
    if (!result) return;
    const paths = result.groups.flatMap((g) =>
      g.images.filter((i) => !i.recommended).map((i) => i.info.path),
    );
    const bytes = collectBytes(result, paths);
    setPendingDelete({ paths, totalBytes: bytes });
  }, [result]);

  // 确认删除（异步执行，进度通过 delete-progress 事件展示）
  const handleConfirmDelete = useCallback(async () => {
    if (!pendingDelete) return;
    const { paths } = pendingDelete;
    setPendingDelete(null);
    setPreview(null); // 关闭预览（如果是从预览发起的删除）
    try {
      await deleteFiles(paths, settings?.permanent_delete ?? false);
    } catch (e) {
      showToast(`删除失败: ${e}`);
    }
  }, [pendingDelete, settings, deleteFiles, showToast]);

  // 监听删除完成事件，更新结果并提示
  useEffect(() => {
    if (!deleteState.result) return;
    const res = deleteState.result;
    setResult((prev) => prev && removePaths(prev, res.deleted));
    showToast(
      `已删除 ${res.deleted.length} 个文件${
        res.failed.length > 0 ? `，${res.failed.length} 个失败` : ""
      }`,
    );
    consumeResult();
  }, [deleteState.result, showToast, consumeResult]);

  // 导出报告
  const handleExportReport = useCallback(async () => {
    if (!result) return;
    const csv = buildReportCsv(result);
    const path = await save({
      defaultPath: "pixsweep-report.csv",
      filters: [{ name: "CSV", extensions: ["csv"] }],
    });
    if (!path) return;
    try {
      await writeTextFile(path, csv);
      showToast(`报告已导出到 ${path}`);
    } catch (e) {
      showToast(`导出失败: ${e}`);
    }
  }, [result, showToast]);

  const aiEnabled = settings?.ai_enabled ?? true;
  const totalGroups = result?.groups.length ?? 0;

  return (
    <div className="app">
      <Toolbar
        folders={folders}
        onAddFolders={addFolders}
        onRemoveFolder={removeFolder}
        onStartScan={handleStartScan}
        imageCount={imageCount}
        onOpenSettings={() => setShowSettings(true)}
        onOpenScoreHelp={() => setShowScoreHelp(true)}
        onOpenTrashBin={() => setShowTrashBin(true)}
        scanning={scanState.scanning}
        aiEnabled={aiEnabled}
      />

      {scanState.scanning && scanState.progress && (
        <ProgressBar progress={scanState.progress} />
      )}

      {/* 删除进度条（一键删除时的实时反馈） */}
      {deleteState.deleting && deleteState.progress && (
        <div className="delete-progress">
          <div className="delete-progress-head">
            <span>正在删除照片…</span>
            <span>
              {deleteState.progress.current}/{deleteState.progress.total}
            </span>
          </div>
          <div className="delete-progress-track">
            <div
              className="delete-progress-fill"
              style={{
                width: `${Math.round(
                  (deleteState.progress.current / deleteState.progress.total) * 100,
                )}%`,
              }}
            />
          </div>
        </div>
      )}

      {scanState.error && (
        <div className="error-banner">扫描失败：{scanState.error}</div>
      )}

      <main className="content" ref={contentRef}>
        {!result && !scanState.scanning && (
          <div className="empty-state">
            <div className="empty-icon">PS</div>
            <h2>开始清理重复图片</h2>
            <p>添加一个或多个文件夹，PixSweep 会扫描其中的图片，</p>
            <p>找出相似图片并用 AI 推荐最佳的一张保留。</p>
          </div>
        )}

        {result && (
          <div className="result-header">
            <h2>
              相似图片分组（{totalGroups} 组 · 可节省{" "}
              {formatBytes(result.total_reclaimable_bytes)}）
            </h2>
            <div className="batch-id-label" title="批次号：可用于日志追溯">
              批次号：<code>{result.batch_id}</code>
            </div>
          </div>
        )}

        <div className="groups">
          {/* 顶部哨兵：滚到接近顶部时窗口向上平移 */}
          {windowStart > 0 && (
            <div ref={topSentinelRef} className="window-sentinel" aria-hidden="true" />
          )}
          {result?.groups
            .slice(windowStart, windowStart + RENDER_WINDOW)
            .map((group, i) => (
              <div key={group.group_id} data-group-index={windowStart + i}>
                <GroupCard
                  group={group}
                  onDeleteOne={handleDeleteOne}
                  onBatchDelete={handleGroupDeleteDirect}
                  onPreview={(imagePath) =>
                    handleOpenPreview(windowStart + i, imagePath)
                  }
                  disabled={scanState.scanning}
                />
              </div>
            ))}
          {/* 底部哨兵：滚到底部时窗口向下平移 */}
          {result && windowStart + RENDER_WINDOW < result.groups.length && (
            <div
              ref={bottomSentinelRef}
              className="window-sentinel"
              aria-hidden="true"
            />
          )}
        </div>

        {result && result.groups.length > RENDER_WINDOW && (
          <div className="load-more">
            <div className="load-more-hint">
              共 {result.groups.length} 组 · 当前显示{" "}
              {windowStart + 1}–{Math.min(windowStart + RENDER_WINDOW, result.groups.length)}
            </div>
          </div>
        )}

        {result && result.groups.length === 0 && (
          <div className="no-groups">未发现相似图片，你的相册很干净！</div>
        )}
      </main>

      <StatsBar
        result={result}
        onDeleteAll={handleDeleteAll}
        onExportReport={handleExportReport}
        disabled={scanState.scanning}
      />

      {showSettings && settings && (
        <SettingsPanel
          settings={settings}
          systemInfo={systemInfo}
          mcpStatus={mcpStatus}
          onChange={handleSaveSettings}
          onToggleMcp={handleToggleMcp}
          onClose={() => setShowSettings(false)}
        />
      )}

      {pendingDelete && (
        <DeleteConfirmModal
          paths={pendingDelete.paths}
          totalBytes={pendingDelete.totalBytes}
          permanent={settings?.permanent_delete ?? false}
          onConfirm={handleConfirmDelete}
          onCancel={() => setPendingDelete(null)}
        />
      )}

      {showScoreHelp && <ScoreHelpModal onClose={() => setShowScoreHelp(false)} />}

      {showTrashBin && (
        <TrashBinModal
          onClose={() => setShowTrashBin(false)}
          onRestored={(paths) => {
            if (paths.length === 0) return;
            const preview =
              paths.length === 1
                ? paths[0].split(/[\\/]/).pop()
                : `${paths[0].split(/[\\/]/).pop()} 等 ${paths.length} 张`;
            // 恢复后主界面不再显示这些照片：它们已回到原文件夹，从扫描结果组中移除
            setResult((prev) => prev && removePaths(prev, paths));
            showToast(`已从临时回收站恢复：${preview}`);
            setShowTrashBin(false);
          }}
        />
      )}

      {preview && result && (
        <PreviewModal
          result={result}
          initialGroupIndex={preview.groupIndex}
          initialImagePath={preview.imagePath}
          onClose={() => setPreview(null)}
          onDelete={handlePreviewDelete}
          onDeleteOne={handleDeleteOne}
          keyboardLocked={!!pendingDelete}
        />
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

/** 计算指定路径集合的总字节数。 */
function collectBytes(result: ScanResult | null, paths: string[]): number {
  if (!result) return 0;
  const pathSet = new Set(paths);
  return result.groups.reduce(
    (sum, g) =>
      sum +
      g.images
        .filter((i) => pathSet.has(i.info.path))
        .reduce((s, i) => s + i.info.size, 0),
    0,
  );
}

/** 从结果中移除已删除的图片路径。 */
function removePaths(result: ScanResult, deleted: string[]): ScanResult {
  const deletedSet = new Set(deleted);
  const groups = result.groups
    .map((g) => {
      const images = g.images.filter((i) => !deletedSet.has(i.info.path));
      return { ...g, images };
    })
    .filter((g) => g.images.length > 0);
  const totalReclaimable = groups.reduce(
    (sum, g) =>
      sum +
      g.images.filter((i) => !i.recommended).reduce((s, i) => s + i.info.size, 0),
    0,
  );
  return { ...result, groups, total_reclaimable_bytes: totalReclaimable };
}

/** 生成 CSV 报告。 */
function buildReportCsv(result: ScanResult): string {
  const lines = [
    "分组,推荐保留,评分,文件路径,大小(字节)",
  ];
  result.groups.forEach((g) => {
    g.images.forEach((img) => {
      lines.push(
        [
          g.group_id,
          img.recommended ? "是" : "否",
          img.score != null ? img.score.toFixed(1) : "",
          `"${img.info.path.replace(/"/g, '""')}"`,
          img.info.size,
        ].join(","),
      );
    });
  });
  return "\uFEFF" + lines.join("\n");
}
