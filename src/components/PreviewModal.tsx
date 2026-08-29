/**
 * 全屏预览对比组件。
 *
 * 布局参考 QQ 空间相册：左侧大图主区（全屏自适应）+ 右侧信息栏 + 底部缩略图导航。
 *
 * 操作：
 *  - ↑/↓：切换上一组 / 下一组
 *  - ←/→：切换组内上一张 / 下一张照片
 *  - Enter：删除组内除推荐外的所有照片（保留推荐）
 *  - Esc：退出预览
 *  - 滚轮：缩放（以鼠标位置为中心）
 *  - 左键拖动图片：平移（仅缩放后有效）
 *  - 右键图片 / 缩略图：弹出菜单（删除当前 / 删除其他）
 */
import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import type { ScanResult } from "../types";
import { formatBytes } from "../types";
import { api } from "../api";
import { ContextMenu, type ContextMenuItem } from "./ContextMenu";

interface PreviewModalProps {
  result: ScanResult;
  initialGroupIndex: number;
  /** 进入预览时被双击图片的绝对路径（唯一标识，避免索引错位） */
  initialImagePath: string;
  onClose: () => void;
  /** 批量删除（带确认弹窗）：删一组里的若干张 */
  onDelete: (paths: string[]) => void;
  /** 单条删除（无确认）：仅删 1 张 */
  onDeleteOne: (path: string) => void;
  /** 删除确认框打开时锁定预览键盘（避免 Esc/Enter 冲突） */
  keyboardLocked?: boolean;
}

interface MenuPos {
  x: number;
  y: number;
  /** 右键点击的 path（用于菜单操作的目标） */
  targetPath: string;
}

/** 从完整组号 "20260818094357-000001" 提取 6 位序号。 */
function extractSeq(groupId: string): number {
  const i = groupId.lastIndexOf("-");
  if (i < 0) return 0;
  return parseInt(groupId.slice(i + 1), 10) || 0;
}

export function PreviewModal({
  result,
  initialGroupIndex,
  initialImagePath,
  onClose,
  onDelete,
  onDeleteOne,
  keyboardLocked = false,
}: PreviewModalProps) {
  const [groupIndex, setGroupIndex] = useState(initialGroupIndex);
  // 【关键】用 path（唯一标识）定位当前图片，不用索引：
  // 组内图片会按推荐排序（orderedImages），若用"进入时的原始索引"定位会错位（双击 A 显示 B）。
  const [imagePath, setImagePath] = useState(initialImagePath);
  // 【关键修复】原图与缩略图分开存储，互不污染：
  // - fullImages 只被 getFullImage 写入（主区大图）
  // - thumbs 只被 getThumbnail 写入（底部导航缩略图）
  // 之前共用一个 images state，缩略图先完成→小图，原图后完成→大图，导致"时大时小"
  const [fullImages, setFullImages] = useState<Record<string, string>>({});
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [loadState, setLoadState] = useState<"loading" | "loaded" | "error">("loading");
  const [loadError, setLoadError] = useState<string>("");
  const abortRef = useRef<AbortController | null>(null);
  const loadedFullRef = useRef<Set<string>>(new Set());
  const loadedThumbRef = useRef<Set<string>>(new Set());
  // 右键菜单
  const [menu, setMenu] = useState<MenuPos | null>(null);

  // 缩放/拖动状态（scale=1 表示 CSS 填满容器，Ctrl+滚轮在基础上叠加）
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  // 缩放/平移的 ref 镜像（滚轮以鼠标为锚点计算需要当前值，避免 setState 闭包陈旧）
  const scaleRef = useRef(1);
  const panRef = useRef({ x: 0, y: 0 });
  const stageRef = useRef<HTMLDivElement | null>(null);

  const groups = useMemo(() => result.groups, [result]);
  const group = useMemo(
    () => groups[Math.max(0, Math.min(groupIndex, groups.length - 1))],
    [groups, groupIndex],
  );

  // 组内按推荐排序（推荐在前）
  const orderedImages = useMemo(() => {
    if (!group) return [];
    return [...group.images].sort((a, b) => Number(b.recommended) - Number(a.recommended));
  }, [group]);

  // 当前图片索引：由 path 派生（从排序后的数组中定位）。path 不在组内时回退到第 0 张。
  const imageIndex = Math.max(
    0,
    orderedImages.findIndex((img) => img.info.path === imagePath),
  );
  const currentImage = orderedImages[imageIndex];
  const currentSrc = currentImage ? fullImages[currentImage.info.path] : undefined;

  // 加载当前图片原图（含缓存、错误处理、陈旧请求取消）
  useEffect(() => {
    if (!currentImage) {
      setLoadState("loaded");
      setLoadError("");
      return;
    }
    const path = currentImage.info.path;
    if (loadedFullRef.current.has(path)) {
      setLoadState("loaded");
      setLoadError("");
      return;
    }
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoadState("loading");
    setLoadError("");
    api
      .getFullImage(path)
      .then((dataUrl) => {
        if (ctrl.signal.aborted) return;
        loadedFullRef.current.add(path);
        setFullImages((prev) => ({ ...prev, [path]: dataUrl }));
        setLoadState("loaded");
      })
      .catch((e) => {
        if (ctrl.signal.aborted) return;
        console.error("加载原图失败:", path, e);
        setLoadState("error");
        setLoadError(String(e?.message ?? e));
      });
  }, [currentImage]);

  // 加载当前组所有缩略图（只用于底部导航条，独立 state 不与原图冲突）
  useEffect(() => {
    if (!group) return;
    for (const img of group.images) {
      const path = img.info.path;
      if (loadedThumbRef.current.has(path)) continue;
      api
        .getThumbnail(path, img.info.file_hash)
        .then((dataUrl) => {
          loadedThumbRef.current.add(path);
          setThumbs((prev) => ({ ...prev, [path]: dataUrl }));
        })
        .catch((e) => console.warn("缩略图加载失败:", path, e));
    }
  }, [group]);

  // 组件卸载时取消请求
  useEffect(() => () => abortRef.current?.abort(), []);

  // 切换组：定位到新组"推荐优先"的第一张
  const goToGroup = useCallback(
    (delta: number) => {
      setGroupIndex((prev) => {
        const next = Math.max(0, Math.min(prev + delta, groups.length - 1));
        const newGroup = groups[next];
        if (newGroup?.images.length) {
          const first = [...newGroup.images].sort(
            (a, b) => Number(b.recommended) - Number(a.recommended),
          )[0];
          setImagePath(first.info.path);
        }
        return next;
      });
    },
    [groups],
  );

  // 切换照片：按 path 切换（先算目标索引，再设置对应的 path）
  const goToImage = useCallback(
    (delta: number) => {
      const next = Math.max(0, Math.min(imageIndex + delta, orderedImages.length - 1));
      const target = orderedImages[next];
      if (target) setImagePath(target.info.path);
    },
    [imageIndex, orderedImages],
  );

  // 重置缩放/位移
  const resetView = useCallback(() => {
    scaleRef.current = 1;
    panRef.current = { x: 0, y: 0 };
    setScale(1);
    setPan({ x: 0, y: 0 });
  }, []);

  // 切换图片时自动重置视图
  useEffect(() => {
    resetView();
  }, [currentImage, resetView]);

  // 滚轮缩放（以鼠标位置为中心）。
  // React 的 onWheel 是 passive 监听，preventDefault 无效——用原生非 passive 监听。
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const rect = el.getBoundingClientRect();
      // 鼠标相对图片元素中心（transform 原点）的坐标
      const px = e.clientX - (rect.left + rect.width / 2);
      const py = e.clientY - (rect.top + rect.height / 2);
      const prev = scaleRef.current;
      const next = Math.max(1, Math.min(10, prev + (e.deltaY > 0 ? -0.15 : 0.15)));
      if (next === prev) return;
      // 锚点不动：pan' + s'*p = pan + s*p  =>  pan' = pan + (s - s')*p
      const nextPan = {
        x: panRef.current.x + (prev - next) * px,
        y: panRef.current.y + (prev - next) * py,
      };
      scaleRef.current = next;
      panRef.current = next <= 1 ? { x: 0, y: 0 } : nextPan;
      setScale(next);
      setPan(panRef.current);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // 左键拖动图片
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      if (!(e.target as HTMLElement).classList.contains("preview-image")) return;
      e.preventDefault();
      setIsDragging(true);
      dragStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
    },
    [pan.x, pan.y],
  );

  const handleMouseMove = useCallback(
    (e: globalThis.MouseEvent) => {
      if (!isDragging || !dragStart.current) return;
      setPan({
        x: dragStart.current.panX + (e.clientX - dragStart.current.x),
        y: dragStart.current.panY + (e.clientY - dragStart.current.y),
      });
    },
    [isDragging],
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    dragStart.current = null;
  }, []);

  useEffect(() => {
    if (!isDragging) return;
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // 删除组内除推荐外的所有照片（定义在键盘 effect 之前，供 Enter 键调用）
  const handleDelete = useCallback(() => {
    if (!group) return;
    const toDelete = group.images
      .filter((img) => !img.recommended)
      .map((img) => img.info.path);
    if (toDelete.length === 0) return;
    onDelete(toDelete);
  }, [group, onDelete]);

  // 右键菜单 handler（主图或缩略图）：阻止默认 + 记录坐标 + 目标 path
  const handleContextMenu = useCallback(
    (path: string, e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setMenu({ x: e.clientX, y: e.clientY, targetPath: path });
    },
    [],
  );

  // 右键菜单项
  const menuItems: ContextMenuItem[] = useMemo(() => {
    if (!menu) return [];
    const others = group
      ? group.images.filter((i) => i.info.path !== menu.targetPath)
      : [];
    return [
      {
        label: "删除当前图片",
        danger: true,
        onClick: () => onDeleteOne(menu.targetPath),
      },
      {
        label: others.length > 0 ? `删除其他 ${others.length} 张` : "删除其他图片（无）",
        onClick: () =>
          others.length > 0 ? onDelete(others.map((i) => i.info.path)) : undefined,
        disabled: others.length === 0,
      },
    ];
  }, [menu, group, onDelete, onDeleteOne]);

  // 键盘控制
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // 删除确认框打开时锁定预览键盘（避免 Esc/Enter 冲突）
      if (keyboardLocked) return;
      switch (e.key) {
        case "ArrowUp":
          e.preventDefault();
          goToGroup(-1);
          break;
        case "ArrowDown":
          e.preventDefault();
          goToGroup(1);
          break;
        case "ArrowLeft":
          e.preventDefault();
          goToImage(-1);
          break;
        case "ArrowRight":
          e.preventDefault();
          goToImage(1);
          break;
        case "Enter":
          e.preventDefault();
          handleDelete();
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goToGroup, goToImage, onClose, handleDelete]);

  if (!group || !currentImage) return null;

  const isRecommended = currentImage.recommended;
  const deleteCount = group.images.filter((img) => !img.recommended).length;

  return (
    <div className="preview-overlay" onClick={onClose}>
      <div className="preview-shell" onClick={(e) => e.stopPropagation()}>
        {/* ===== 左侧：大图主区 ===== */}
        <div
          ref={stageRef}
          className={`preview-main ${isDragging ? "dragging" : ""}`}
          onMouseDown={handleMouseDown}
        >
          {/* 顶栏：组号 + 计数 + 缩放/重置 */}
          <div className="preview-topbar">
            <div className="preview-title">
              <span className="preview-group-label">{group.group_id}</span>
              <span className="preview-group-count">
                {extractSeq(group.group_id)}/{groups.length}
              </span>
              <span className="preview-sim">相似度 {(group.similarity * 100).toFixed(1)}%</span>
              <span className="preview-photo-count">
                照片 {imageIndex + 1}/{orderedImages.length}
              </span>
            </div>
            <div className="preview-topbar-actions">
              {loadState === "loaded" && (
                <div className="preview-zoom-bar">
                  <button
                    className="preview-zoom-btn"
                    onClick={(e) => { e.stopPropagation(); setScale((s) => Math.max(0.1, s - 0.2)); }}
                    title="缩小"
                  >−</button>
                  <span className="preview-zoom-val" title="100% = 图片填满容器，滚轮缩放（以鼠标为中心）">
                    {Math.round(scale * 100)}%
                  </span>
                  <button
                    className="preview-zoom-btn"
                    onClick={(e) => { e.stopPropagation(); setScale((s) => Math.min(10, s + 0.2)); }}
                    title="放大"
                  >+</button>
                  <button
                    className="preview-zoom-btn"
                    onClick={(e) => { e.stopPropagation(); resetView(); }}
                    title="重置"
                  >⟲</button>
                </div>
              )}
              <button
                className="btn btn-sm"
                onClick={(e) => { e.stopPropagation(); onClose(); }}
                title="Esc 退出"
              >✕ 关闭</button>
            </div>
          </div>

          {/* 图片舞台（占满剩余空间） */}
          <div className="preview-stage">
            <button
              className="preview-arrow prev"
              onClick={(e) => { e.stopPropagation(); goToImage(-1); }}
              disabled={imageIndex === 0}
              title="← 上一张"
            >‹</button>

            <div
              className="preview-image-wrap"
              onContextMenu={(e) => handleContextMenu(currentImage.info.path, e)}
            >
              {loadState === "loaded" && currentSrc ? (
                <img
                  src={currentSrc}
                  alt={currentImage.info.name}
                  className={`preview-image ${isDragging ? "dragging" : ""}`}
                  style={{
                    // scale=1 时不加 transform：由 CSS object-fit: contain 填满容器（稳定）
                    // scale>1 时叠加平移+缩放（用户 Ctrl+滚轮）
                    transform:
                      scale > 1
                        ? `translate(${pan.x}px, ${pan.y}px) scale(${scale})`
                        : undefined,
                    cursor: scale > 1 ? (isDragging ? "grabbing" : "grab") : "default",
                    transition: isDragging ? "none" : undefined,
                    pointerEvents: isDragging ? "none" : "auto",
                  }}
                  draggable={false}
                />
              ) : loadState === "error" ? (
                <div className="preview-error">
                  <div className="preview-error-icon">!</div>
                  <div className="preview-error-text">图片加载失败</div>
                  <div className="preview-error-detail">{loadError}</div>
                  <button
                    className="btn btn-sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      loadedFullRef.current.delete(currentImage.info.path);
                      setFullImages((prev) => {
                        const next = { ...prev };
                        delete next[currentImage.info.path];
                        return next;
                      });
                      setLoadState("loading");
                      api
                        .getFullImage(currentImage.info.path)
                        .then((d) => {
                          loadedFullRef.current.add(currentImage.info.path);
                          setFullImages((prev) => ({ ...prev, [currentImage.info.path]: d }));
                          setLoadState("loaded");
                        })
                        .catch((err) => {
                          console.error("重试仍失败:", err);
                          setLoadState("error");
                          setLoadError(String((err as Error)?.message ?? err));
                        });
                    }}
                  >重试</button>
                </div>
              ) : (
                <div className="preview-loading">
                  <div className="preview-spinner" />
                  <div>加载中...</div>
                </div>
              )}
              {/* 推荐/删除角标（图片左上角小角标，不挡图） */}
              <div className={`preview-flag ${isRecommended ? "keep" : "del"}`}>
                {isRecommended ? "★ 推荐保留" : "建议删除"}
              </div>
            </div>

            <button
              className="preview-arrow next"
              onClick={(e) => { e.stopPropagation(); goToImage(1); }}
              disabled={imageIndex >= orderedImages.length - 1}
              title="→ 下一张"
            >›</button>
          </div>

          {/* 底部缩略图导航条（独立条，不重叠图片） */}
          <div className="preview-thumbs">
            {orderedImages.map((img, idx) => {
              const thumbSrc = thumbs[img.info.path];
              const isActive = idx === imageIndex;
              return (
                <div
                  key={img.info.path}
                  className={`preview-thumb ${isActive ? "active" : ""} ${img.recommended ? "keep" : "del"}`}
                  onClick={(e) => { e.stopPropagation(); setImagePath(img.info.path); }}
                  onContextMenu={(e) => handleContextMenu(img.info.path, e)}
                  title={`${img.info.path}\n${img.info.width}×${img.info.height} · ${formatBytes(img.info.size)}${img.recommended ? "\n★ 推荐保留" : "\n建议删除"}`}
                >
                  {thumbSrc ? (
                    <img src={thumbSrc} alt={img.info.name} />
                  ) : (
                    <div className="thumb-placeholder">...</div>
                  )}
                  <div className="preview-thumb-meta">
                    {img.info.width > 0 ? `${img.info.width}×${img.info.height}` : formatBytes(img.info.size)}
                  </div>
                  <div className="preview-thumb-tag">{img.recommended ? "★" : "删"}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ===== 右侧：信息栏 ===== */}
        <aside className="preview-sidebar">
          <div className="preview-sidebar-section">
            <div className="preview-sidebar-label">文件名</div>
            <div className="preview-sidebar-value preview-sidebar-name">
              {currentImage.info.name}
            </div>
          </div>

          <div className="preview-sidebar-section">
            <div className="preview-sidebar-label">完整路径</div>
            <div className="preview-sidebar-value preview-sidebar-path" title={currentImage.info.path}>
              {currentImage.info.path}
            </div>
          </div>

          <div className="preview-sidebar-section preview-sidebar-info-grid">
            <div>
              <div className="preview-sidebar-label">尺寸</div>
              <div className="preview-sidebar-value">
                {currentImage.info.width > 0
                  ? `${currentImage.info.width} × ${currentImage.info.height}`
                  : "未知"}
              </div>
            </div>
            <div>
              <div className="preview-sidebar-label">大小</div>
              <div className="preview-sidebar-value">{formatBytes(currentImage.info.size)}</div>
            </div>
            <div>
              <div className="preview-sidebar-label">格式</div>
              <div className="preview-sidebar-value">{currentImage.info.format.toUpperCase()}</div>
            </div>
          </div>

          <div className="preview-sidebar-section">
            <div className="preview-sidebar-label">评分</div>
            <div className="preview-score-grid">
              <div className="preview-score-cell">
                <div className="preview-score-cell-label">类型</div>
                <div className="preview-score-cell-value">
                  {currentImage.has_face ? "人像" : currentImage.scene === 2 ? "宠物" : currentImage.scene === 3 ? "风景" : "其他"}
                </div>
              </div>
              <div className="preview-score-cell">
                <div className="preview-score-cell-label">综合</div>
                <div className="preview-score-cell-value">
                  {currentImage.score != null ? currentImage.score.toFixed(1) : "—"}
                </div>
              </div>
              <div className="preview-score-cell">
                <div className="preview-score-cell-label">对焦</div>
                <div className="preview-score-cell-value">
                  {currentImage.focus_score != null ? currentImage.focus_score.toFixed(1) : "—"}
                </div>
              </div>
              <div className="preview-score-cell">
                <div className="preview-score-cell-label">闭眼</div>
                <div className="preview-score-cell-value">
                  {(currentImage.has_face || currentImage.scene === 2) ? (currentImage.is_eye_closed ? "是" : "否") : "—"}
                </div>
              </div>
              <div className="preview-score-cell">
                <div className="preview-score-cell-label">失焦</div>
                <div className="preview-score-cell-value">
                  {currentImage.is_out_of_focus ? "是" : "否"}
                </div>
              </div>
            </div>
          </div>

          <div className="preview-sidebar-section">
            <div className="preview-sidebar-label">
              {isRecommended ? "推荐理由" : "删除理由"}
            </div>
            <div className={`preview-sidebar-reason ${isRecommended ? "keep" : "del"}`}>
              {currentImage.reason}
            </div>
          </div>

          <div className="preview-sidebar-spacer" />

          {/* 删除操作 */}
          <div className="preview-sidebar-actions">
            <button
              className="btn btn-danger btn-block"
              onClick={() => onDeleteOne(currentImage.info.path)}
              title="仅删除当前查看的图片（无确认）"
            >
              删除当前图片
            </button>
            <button
              className="btn btn-secondary btn-block"
              onClick={handleDelete}
              disabled={deleteCount === 0}
              title="删除组内除推荐外的所有照片（带确认）"
            >
              删除其他 {deleteCount} 张（保留推荐）
            </button>
            <div className="preview-sidebar-hint">
              ←/→ 切换照片 · 滚轮缩放 · 右键图片呼出菜单
            </div>
          </div>
        </aside>
      </div>
      <ContextMenu position={menu} items={menuItems} onClose={() => setMenu(null)} />
    </div>
  );
}