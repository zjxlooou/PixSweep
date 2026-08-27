// 后端 IPC 调用封装
import { invoke } from "@tauri-apps/api/core";
import type {
  AppSettings,
  CacheCleanupResult,
  CacheSummary,
  CacheType,
  McpStatus,
  ScanResult,
  SystemInfo,
} from "./types";

export const api = {
  getSystemInfo: () => invoke<SystemInfo>("get_system_info"),

  getSettings: () => invoke<AppSettings>("get_settings"),

  setSettings: (settings: AppSettings) =>
    invoke<void>("set_settings", { settings }),

  startScan: (
    folders: string[],
    similarityThreshold?: number,
    incremental?: boolean,
  ) =>
    invoke<string>("start_scan", {
      folders,
      similarityThreshold,
      incremental,
    }),

  getThumbnail: (path: string, fileHash: string) =>
    invoke<string>("get_thumbnail", { path, fileHash }),

  getFullImage: (path: string) => invoke<string>("get_full_image", { path }),

  deleteFiles: (paths: string[], permanent: boolean) =>
    invoke<void>("delete_files", { paths, permanent }),

  // 临时回收站管理
  listTrashImages: () => invoke<void>("list_trash_images"),
  restoreTrashItem: (id: string) =>
    invoke<string>("restore_trash_item", { id }),
  restoreAllTrashImages: () => invoke<string[]>("restore_all_trash_images"),
  clearTrashBin: () => invoke<number>("clear_trash_bin"),
  /** 在系统文件管理器中打开临时回收站目录（Windows: explorer.exe） */
  openTrashBinInExplorer: () =>
    invoke<void>("open_trash_bin_in_explorer"),

  getScanResult: () => invoke<ScanResult | null>("get_scan_result"),

  // MCP server 管理
  getMcpStatus: () => invoke<McpStatus>("get_mcp_status"),
  setMcpEnabled: (enabled: boolean) =>
    invoke<McpStatus>("set_mcp_enabled", { enabled }),

  // 缓存清理（移入系统回收站）
  getCacheSummary: () => invoke<CacheSummary[]>("get_cache_summary"),
  cleanupCache: (types: CacheType[]) =>
    invoke<CacheCleanupResult>("cleanup_cache", { types }),

  // 文件夹图片总数
  countImages: (folders: string[]) => invoke<number>("count_images", { folders }),
};