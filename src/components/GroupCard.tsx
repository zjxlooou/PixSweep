import { useMemo, useState, type MouseEvent } from "react";
import type { GroupImage, ImageGroup } from "../types";
import { formatBytes } from "../types";
import { ImageThumbnail } from "./ImageThumbnail";
import { ContextMenu, type ContextMenuItem } from "./ContextMenu";

interface GroupCardProps {
  group: ImageGroup;
  /** 单条删除（无确认）：从组中移除指定 path */
  onDeleteOne: (path: string) => void;
  /** 组级批量删除（直接删除，不弹确认）：传入待删路径 */
  onBatchDelete: (paths: string[]) => void;
  /** 双击图片进预览：传入被点击图片的绝对路径（path 是唯一标识，避免索引错位） */
  onPreview: (path: string) => void;
  disabled: boolean;
}

interface MenuPos {
  x: number;
  y: number;
  /** 右键点击的 path（用于菜单操作的目标） */
  targetPath: string;
}

export function GroupCard({
  group,
  onDeleteOne,
  onBatchDelete,
  onPreview,
  disabled,
}: GroupCardProps) {
  const [manualMode, setManualMode] = useState(false);
  // 手动模式下，被红框标记的图片路径集合（再点一下取消）
  const [marked, setMarked] = useState<Set<string>>(new Set());
  // 右键菜单位置
  const [menu, setMenu] = useState<MenuPos | null>(null);

  // 默认标记：所有"建议删除"的图片（即非推荐）
  const defaultMarked = useMemo(
    () => new Set(
      group.images.filter((img) => !img.recommended).map((img) => img.info.path),
    ),
    [group],
  );

  // 手动模式：marked；推荐模式：默认标记
  const toDelete = manualMode ? marked : defaultMarked;

  const deleteCount = toDelete.size;
  const deleteBytes = group.images
    .filter((img) => toDelete.has(img.info.path))
    .reduce((sum, img) => sum + img.info.size, 0);

  // 切换某张图片的标记状态
  const toggleMark = (path: string) => {
    setMarked((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  // 在手动模式下，进入时把默认标记作为初始值（方便反向操作：取消某张默认要删的）
  // 这里不主动同步，避免覆盖用户已有的选择

  const handleImageClick = (img: GroupImage) => {
    if (manualMode) {
      toggleMark(img.info.path);
    }
    // 推荐模式：单击不响应，避免和"双击预览"冲突
  };

  // 双击进预览（仅推荐模式有效；手动模式双击也响应以方便调整后再预览）
  const handleImageDoubleClick = (path: string) => {
    onPreview(path);
  };

  const handleImageContextMenu = (img: GroupImage, e: MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setMenu({ x: e.clientX, y: e.clientY, targetPath: img.info.path });
  };

  // 右键菜单项
  const menuItems: ContextMenuItem[] = useMemo(() => {
    if (!menu) return [];
    const targetImg = group.images.find((i) => i.info.path === menu.targetPath);
    if (!targetImg) return [];
    const others = group.images.filter((i) => i.info.path !== menu.targetPath);
    const othersPaths = others.map((i) => i.info.path);
    return [
      {
        label: "删除当前图片",
        danger: true,
        onClick: () => onDeleteOne(menu.targetPath),
      },
      {
        label: others.length > 0 ? `删除其他 ${others.length} 张` : "删除其他图片（无）",
        onClick: () => othersPaths.length > 0 && onBatchDelete(othersPaths),
        disabled: othersPaths.length === 0,
      },
    ];
  }, [menu, group, onDeleteOne, onBatchDelete]);

  const handleBatchDelete = () => {
    if (deleteCount === 0) return;
    onBatchDelete(Array.from(toDelete));
  };

  const displayId = group.group_id;

  return (
    <div className="group-card">
      <div className="group-head">
        <span className="group-no">{displayId}</span>
        <span className="group-sim">相似度 {(group.similarity * 100).toFixed(1)}%</span>
        <span className="group-count">{group.images.length} 张</span>
      </div>
      <div className="group-images">
        {group.images.map((img) => {
          const isRecommended = img.recommended;
          const isMarked = toDelete.has(img.info.path);
          return (
            <div
              key={img.info.path}
              className={`group-image ${isRecommended ? "recommended" : ""} ${
                isMarked ? "marked-del" : ""
              } ${manualMode ? "manual-mode" : ""}`}
              onClick={() => handleImageClick(img)}
              onDoubleClick={() => handleImageDoubleClick(img.info.path)}
              onContextMenu={(e) => handleImageContextMenu(img, e)}
              title={img.info.path}
            >
              <ImageThumbnail
                path={img.info.path}
                fileHash={img.info.file_hash}
                name={img.info.name}
              />
              {/* 单条删除按钮（hover 显示）— 无确认 */}
              <button
                type="button"
                className="image-x-btn"
                title="删除此图片（无确认）"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteOne(img.info.path);
                }}
              >
                ✕
              </button>
              <div className="image-meta">
                <span className="image-name">{img.info.name}</span>
                <span className="image-size">{formatBytes(img.info.size)}</span>
              </div>
              <div className="image-badges">
                {isRecommended && <span className="badge keep">推荐</span>}
                <span className="badge type">
                  {img.has_face ? "人像" : img.scene === 2 ? "宠物" : img.scene === 3 ? "风景" : "其他"}
                </span>
                {(img.has_face || img.scene === 2) && img.is_eye_closed && (
                  <span className="badge eye-closed">闭眼</span>
                )}
                {img.is_out_of_focus && <span className="badge out-of-focus">失焦</span>}
                {img.score != null && (
                  <span className="badge score">综合 {img.score.toFixed(1)}</span>
                )}
                {isMarked && <span className="badge del">删除</span>}
              </div>
              <div className={`image-reason ${isRecommended ? "keep" : "del"}`}>
                {img.reason}
              </div>
              <div className="preview-tip">
                {manualMode ? "点击切换红框（取消删除）" : "双击放大预览 · 右键菜单"}
              </div>
            </div>
          );
        })}
      </div>

      <div className="group-footer">
        <div className="group-info">
          <span className="group-save">
            {manualMode
              ? `已标记 ${deleteCount} 张（可节省 ${formatBytes(deleteBytes)}）`
              : `可节省 ${formatBytes(deleteBytes)}（${deleteCount} 张）`}
          </span>
        </div>
        <div className="group-actions">
          <button
            className="btn btn-sm btn-ghost"
            onClick={() => setManualMode((m) => !m)}
            disabled={disabled}
          >
            {manualMode ? "推荐模式" : "手动选择"}
          </button>
          <button
            className="btn btn-sm btn-danger"
            onClick={handleBatchDelete}
            disabled={disabled || deleteCount === 0}
          >
            删除 {deleteCount} 张
          </button>
        </div>
      </div>

      <ContextMenu position={menu} items={menuItems} onClose={() => setMenu(null)} />
    </div>
  );
}