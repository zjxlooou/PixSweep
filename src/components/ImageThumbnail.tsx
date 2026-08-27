import { useEffect, useRef, useState } from "react";
import { api } from "../api";

interface ImageThumbnailProps {
  path: string;
  fileHash: string;
  name: string;
}

// 共享的 IntersectionObserver（单例），避免每个缩略图创建独立 observer 导致性能问题。
let sharedObserver: IntersectionObserver | null = null;
const pendingCallbacks = new Map<Element, () => void>();

function getSharedObserver(): IntersectionObserver {
  if (!sharedObserver) {
    sharedObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const cb = pendingCallbacks.get(entry.target);
            if (cb) {
              pendingCallbacks.delete(entry.target);
              sharedObserver?.unobserve(entry.target);
              cb();
            }
          }
        }
      },
      { rootMargin: "200px" },
    );
  }
  return sharedObserver;
}

export function ImageThumbnail({ path, fileHash, name }: ImageThumbnailProps) {
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const loadedRef = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || loadedRef.current) return;

    const loadThumbnail = async () => {
      try {
        const dataUrl = await api.getThumbnail(path, fileHash);
        setSrc(dataUrl);
      } catch {
        setError(true);
      }
    };

    // 使用共享 observer 懒加载
    const observer = getSharedObserver();
    pendingCallbacks.set(el, () => {
      loadedRef.current = true;
      loadThumbnail();
    });
    observer.observe(el);

    return () => {
      pendingCallbacks.delete(el);
      observer.unobserve(el);
    };
  }, [path, fileHash]);

  return (
    <div ref={ref} className="thumb" title={path} style={{ position: "relative" }}>
      {src ? (
        <img src={src} alt={name} loading="lazy" />
      ) : (
        <div className="thumb-placeholder">{error ? "!" : "..."}</div>
      )}
    </div>
  );
}
