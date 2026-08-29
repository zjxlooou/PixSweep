import { useEffect, useState } from "react";
import { api } from "../api";
import type { AppSettings, CacheSummary, CacheType, McpStatus, SystemInfo } from "../types";
import { formatBytes } from "../types";

const CACHE_LABEL: Record<CacheType, string> = {
  proxy: "AI 代理图",
  thumbnails: "缩略图",
  ai_cache: "AI 评分缓存",
  logs: "日志",
  quarantine: "临时文件夹",
};

interface SettingsPanelProps {
  settings: AppSettings;
  systemInfo: SystemInfo | null;
  mcpStatus: McpStatus | null;
  onChange: (settings: AppSettings) => void;
  onToggleMcp: (enabled: boolean) => void;
  onClose: () => void;
  /** 缓存清理完成后回调（App 刷新工具栏临时文件夹占用） */
  onAfterCleanup?: () => void;
}

export function SettingsPanel({
  settings,
  systemInfo,
  mcpStatus,
  onChange,
  onToggleMcp,
  onClose,
  onAfterCleanup,
}: SettingsPanelProps) {
  const update = (patch: Partial<AppSettings>) => {
    onChange({ ...settings, ...patch });
  };

  // ---- 缓存清理（勾选类型 → 移入系统回收站） ----
  const [cacheSummary, setCacheSummary] = useState<CacheSummary[] | null>(null);
  const [selected, setSelected] = useState<Set<CacheType>>(new Set());
  const [cleanMsg, setCleanMsg] = useState<string | null>(null);
  const [cleaning, setCleaning] = useState(false);

  useEffect(() => {
    api
      .getCacheSummary()
      .then(setCacheSummary)
      .catch(() => setCacheSummary([]));
  }, []);

  const toggleType = (t: CacheType) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });
  };

  const runCleanup = async () => {
    if (selected.size === 0 || cleaning) return;
    setCleaning(true);
    setCleanMsg(null);
    try {
      const r = await api.cleanupCache(Array.from(selected));
      setCleanMsg(
        `已移入系统回收站 ${r.moved} 个文件${r.failed ? `，失败 ${r.failed} 个` : ""}`,
      );
      setCacheSummary(await api.getCacheSummary());
      setSelected(new Set());
      onAfterCleanup?.();
    } catch (e) {
      setCleanMsg(`清理失败：${String(e)}`);
    } finally {
      setCleaning(false);
    }
  };

  const allSelected =
    cacheSummary !== null &&
    cacheSummary.length > 0 &&
    cacheSummary.every((s) => selected.has(s.cache_type));
  const toggleSelectAll = () => {
    if (allSelected) setSelected(new Set());
    else setSelected(new Set((cacheSummary ?? []).map((s) => s.cache_type)));
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal modal-wide modal-fixed-title" onClick={(e) => e.stopPropagation()}>
        <div className="modal-titlebar">
          <h2 className="modal-title">设置</h2>
          <button className="modal-close-btn" onClick={onClose} aria-label="关闭">
            ✕
          </button>
        </div>

        <div className="modal-body">
        <div className="settings-group">
          <label className="settings-label">
            相似度阈值：{(settings.similarity_threshold * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min={0.85}
            max={0.98}
            step={0.01}
            value={settings.similarity_threshold}
            onChange={(e) =>
              update({ similarity_threshold: Number(e.target.value) })
            }
          />
          <p className="settings-hint">
            阈值越高越严格，仅找出几乎相同的图片。
          </p>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.ai_enabled}
              onChange={(e) => update({ ai_enabled: e.target.checked })}
            />
            <span>
              启用 AI 质量评分
              {settings.ai_enabled &&
                systemInfo &&
                !systemInfo.technical_model_available && (
                  <span className="settings-warn">（未检测到模型文件）</span>
                )}
            </span>
          </label>
          <p className="settings-hint">
            使用本地 GPU 的 TOPIQ 模型（技术 + 美学）为每组推荐最佳图片。
          </p>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.permanent_delete}
              onChange={(e) => update({ permanent_delete: e.target.checked })}
            />
            <span>永久删除（不进入临时文件夹）</span>
          </label>
          <p className="settings-hint">
            默认关闭：删除的照片移入「临时文件夹」，可随时恢复。开启后：直接永久删除，
            不进入任何可恢复位置（不可找回）。
          </p>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.incremental}
              onChange={(e) => update({ incremental: e.target.checked })}
            />
            <span>增量扫描</span>
          </label>
          <p className="settings-hint">
            跳过已处理过的图片，大幅加速重复扫描。
          </p>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.mcp_enabled}
              onChange={(e) => onToggleMcp(e.target.checked)}
            />
            <span>
              MCP 服务
              {mcpStatus?.running ? (
                <span className="mcp-status mcp-on">● 运行中</span>
              ) : (
                <span className="mcp-status mcp-off">● 已停止</span>
              )}
            </span>
          </label>
          <p className="settings-hint">
            允许外部 AI Agent 通过 MCP 协议操作本应用（扫描 / 删除 / 临时文件夹 /
            设置），用于自动化测试。仅监听本机{" "}
            <code>{mcpStatus?.url ?? "http://127.0.0.1:18765/mcp"}</code>
            ，不对外网开放。
          </p>
        </div>

        {systemInfo && (
          <div className="settings-system">
            <div className="sys-row">
              <span>GPU 加速</span>
              <span className={systemInfo.gpu_available ? "ok" : "off"}>
                {systemInfo.gpu_available
                  ? `可用：${systemInfo.gpu_name ?? "GPU"}`
                  : "不可用（回退 CPU）"}
              </span>
            </div>
            <div className="sys-row">
            </div>
            <div className="sys-row">
              <span>技术质量模型</span>
              <span className={systemInfo.technical_model_available ? "ok" : "off"}>
                {systemInfo.technical_model_available ? "已就绪" : "缺失"}
              </span>
            </div>
            <div className="sys-row">
              <span>数据目录</span>
              <span className="sys-path">{systemInfo.data_dir}</span>
            </div>
          </div>
        )}

        <div className="settings-group settings-cache-box">
          <div className="settings-cache-head">
            <div className="settings-label">清理缓存</div>
          </div>
          <p className="settings-hint">
            勾选要清理的缓存类型，清理后的文件将移入<strong>系统回收站</strong>
            （可恢复），下次扫描会自动重建。不会影响你的源照片。
          </p>
          <div className="settings-cache-list">
            {cacheSummary === null ? (
              <p className="settings-hint">加载缓存摘要中…</p>
            ) : cacheSummary.length === 0 ? (
              <p className="settings-hint">暂无可清理缓存。</p>
            ) : (
              cacheSummary.map((s) => (
                <label className="settings-cache-row" key={s.cache_type}>
                  <input
                    type="checkbox"
                    checked={selected.has(s.cache_type)}
                    onChange={() => toggleType(s.cache_type)}
                  />
                  <span className="settings-cache-name">{CACHE_LABEL[s.cache_type]}</span>
                  <span className="settings-cache-size">
                    {s.count} 个 · {formatBytes(s.bytes)}
                  </span>
                </label>
              ))
            )}
          </div>
          <div className="settings-cache-actions">
            <button
              className="btn btn-ghost btn-sm"
              type="button"
              onClick={toggleSelectAll}
            >
              {allSelected ? "取消全选" : "全选"}
            </button>
            <button
              className="btn"
              onClick={runCleanup}
              disabled={selected.size === 0 || cleaning}
            >
              {cleaning ? "清理中…" : `清理选中（${selected.size} 项）`}
            </button>
            <span className="settings-cache-msg">{cleanMsg}</span>
          </div>
        </div>
        </div>
      </div>
    </div>
  );
}
