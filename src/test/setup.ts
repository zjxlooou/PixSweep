// Vitest 全局 setup：mock Tauri API，使组件测试无需真实后端即可运行
import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

// jsdom 没有 IntersectionObserver，提供最小实现（立即触发回调）
class MockIntersectionObserver {
  private callback: IntersectionObserverCallback;
  private targets = new Set<Element>();
  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback;
  }
  observe(target: Element) {
    this.targets.add(target);
    // 立即触发一次（视为可见），模拟懒加载立即加载
    this.callback(
      [{ target, isIntersecting: true } as IntersectionObserverEntry],
      this as unknown as IntersectionObserver,
    );
  }
  unobserve(target: Element) {
    this.targets.delete(target);
  }
  disconnect() {
    this.targets.clear();
  }
}
vi.stubGlobal(
  "IntersectionObserver",
  MockIntersectionObserver as unknown as typeof IntersectionObserver,
);

// Mock @tauri-apps/api/core.invoke
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

// Mock @tauri-apps/api/event（listen/unlisten）
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

// Mock @tauri-apps/plugin-dialog
vi.mock("@tauri-apps/plugin-dialog", () => ({
  save: vi.fn(() => Promise.resolve(null)),
  open: vi.fn(() => Promise.resolve(null)),
}));

// Mock @tauri-apps/plugin-fs
vi.mock("@tauri-apps/plugin-fs", () => ({
  writeTextFile: vi.fn(() => Promise.resolve()),
}));
