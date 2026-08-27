import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  // 锁定到 PixSweep 项目目录，避免 vitest 向上找到外层 workspace 根
  // (vitest 4 默认 root: process.cwd()，但 workspace 模式下会向上找到最近的 package.json)
  root: __dirname,
  test: {
    // 组件测试文件用 `// @vitest-environment jsdom` 显式标记（vitest 4 已弃用 environmentMatchGlobs）
    // 默认 node 环境适合纯函数测试（formatBytes 等）
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    css: false,
  },
});