// 与 Rust 后端对应的数据类型定义

export interface ImageInfo {
  path: string;
  name: string;
  size: number;
  modified: number;
  width: number;
  height: number;
  format: string;
  file_hash: string;
}

export type ScanPhase =
  | "scanning"
  | "hashing"
  | "clustering"
  | "quality"
  | "done"
  | "error";

export interface ScanProgress {
  session_id: string;
  phase: ScanPhase;
  current: number;
  total: number;
  current_file: string | null;
  ai_enabled: boolean;
  /** 实际使用的推理后端（如 "CUDA (NVIDIA GPU)" / "DirectML (DirectX 12)" / "CPU"） */
  backend: string;
  /** 更细粒度的当前子阶段（如 "识别内容 / 识别眼部 / 对焦判断 / 美学评分"） */
  detail: string;
}

export interface GroupImage {
  info: ImageInfo;
  score: number | null;
  aesthetic_score: number | null;
  technical_score: number | null;
  /** TOPIQ-NR-Face 人脸专评（1.0 ~ 10.0），无人脸或未启用时为 null */
  face_score: number | null;
  /** 是否检测到人脸（用于前端显示"人像"图标） */
  has_face: boolean;
  /** 场景分类（0=其他 1=人像 2=宠物 3=风景） */
  scene: number;
  /** 双眼都闭（OCEC 检测，`max(open_l,open_r) <= 0.5`，前端显示"闭眼"标签） */
  is_eye_closed: boolean;
  /** 对焦分（1.0 ~ 10.0）：人像/宠物为眼部对焦，其余为整图对焦；未启用时为 null */
  focus_score: number | null;
  /** 是否失焦（focus_score 低于阈值；前端显示"失焦"标签） */
  is_out_of_focus: boolean;
  recommended: boolean;
  reason: string;
}

export interface ImageGroup {
  group_id: string; // 格式 "{batch_id}-{6位序号}"，如 "20260818093107-000001"
  images: GroupImage[];
  similarity: number;
  reclaimable_bytes: number;
}

export interface ScanResult {
  session_id: string;
  /** 批次号 yyyyMMddHHmmSS，用于日志追溯 */
  batch_id: string;
  total_images: number;
  groups: ImageGroup[];
  total_reclaimable_bytes: number;
  ai_enabled: boolean;
}

/** 扫描完成摘要（事件小 payload）。 */
export interface ScanSummary {
  session_id: string;
  batch_id: string;
  total_images: number;
  total_groups: number;
  total_reclaimable_bytes: number;
  ai_enabled: boolean;
}

export interface DeleteFailure {
  path: string;
  reason: string;
}

export interface DeleteResult {
  deleted: string[];
  failed: DeleteFailure[];
}

/** 临时回收站（隔离区）中的图片条目 */
export interface TrashImage {
  id: string;
  original_path: string;
  quarantine_filename: string;
  deleted_at: number;
  size: number;
}

export interface AppSettings {
  similarity_threshold: number;
  ai_enabled: boolean;
  permanent_delete: boolean;
  incremental: boolean;
  /** MCP server 开关（供外部 AI Agent 操作应用） */
  mcp_enabled: boolean;
}

/** MCP server 状态 */
export interface McpStatus {
  running: boolean;
  port: number;
  url: string;
}

export interface SystemInfo {
  gpu_available: boolean;
  gpu_name: string | null;
  clip_model_available: boolean;
  /** 主技术质量模型（TOPIQ-NR）是否可用 */
  technical_model_available: boolean;
  data_dir: string;
}

export const PHASE_LABELS: Record<ScanPhase, string> = {
  scanning: "扫描文件夹",
  hashing: "计算图片指纹",
  clustering: "聚类相似图片",
  quality: "AI 质量评分",
  done: "完成",
  error: "出错",
};

/** 将字节数格式化为人类可读字符串。 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = bytes;
  let i = -1;
  do {
    v /= 1024;
    i++;
  } while (v >= 1024 && i < units.length - 1);
  return `${v.toFixed(1)} ${units[i]}`;
}

/** 可清理的缓存类型（对应后端 [serde(rename_all = "snake_case")]）。 */
export type CacheType = "proxy" | "thumbnails" | "ai_cache" | "logs" | "quarantine";

/** 某类缓存的体积摘要（供"清理缓存"面板勾选）。 */
export interface CacheSummary {
  cache_type: CacheType;
  count: number;
  bytes: number;
}

/** 缓存清理结果（移入系统回收站）。 */
export interface CacheCleanupResult {
  moved: number;
  failed: number;
}
