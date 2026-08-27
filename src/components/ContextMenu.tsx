import { useEffect, useRef, useState } from "react";

export interface ContextMenuItem {
  label: string;
  onClick: () => void;
  danger?: boolean;
  disabled?: boolean;
  separatorAfter?: boolean;
}

interface ContextMenuProps {
  /** 鼠标坐标（页面坐标）。传 null 时菜单关闭。 */
  position: { x: number; y: number } | null;
  items: ContextMenuItem[];
  onClose: () => void;
}

/**
 * 右键菜单：受控组件，由父组件提供 position 和 items。
 *
 * 使用方式：
 * - onContextMenu={(e) => { e.preventDefault(); setMenuPos({x:e.clientX,y:e.clientY}); }}
 * - <ContextMenu position={menuPos} items={[...]} onClose={() => setMenuPos(null)} />
 *
 */
export function ContextMenu({ position, items, onClose }: ContextMenuProps) {
  const ref = useRef<HTMLDivElement>(null);
  // 修正菜单超出屏幕右下边界的情况
  const [adjusted, setAdjusted] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (!position) return;
    // 下一帧读 DOM 尺寸，避免 mount 前 width=0
    const id = requestAnimationFrame(() => {
      const el = ref.current;
      if (!el) return;
      const w = el.offsetWidth;
      const h = el.offsetHeight;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      let x = position.x;
      let y = position.y;
      if (x + w > vw) x = Math.max(0, vw - w - 4);
      if (y + h > vh) y = Math.max(0, vh - h - 4);
      setAdjusted({ x, y });
    });
    return () => cancelAnimationFrame(id);
  }, [position]);

  // 点击外部关闭
  useEffect(() => {
    if (!position) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    // mousedown 早于 click 触发，避免点击菜单项时菜单先关闭再点击
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [position, onClose]);

  if (!position) return null;
  const style = {
    left: adjusted?.x ?? position.x,
    top: adjusted?.y ?? position.y,
  };

  return (
    <div ref={ref} className="context-menu" style={style} role="menu">
      {items.map((item, i) => (
        <div key={i}>
          <button
            type="button"
            className={`context-menu-item ${item.danger ? "danger" : ""}`}
            disabled={item.disabled}
            onClick={() => {
              item.onClick();
              onClose();
            }}
            role="menuitem"
          >
            {item.label}
          </button>
          {item.separatorAfter && <div className="context-menu-separator" />}
        </div>
      ))}
    </div>
  );
}