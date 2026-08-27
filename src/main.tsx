//! 前端入口：挂载 React 根组件。
//!
//! 用 ErrorBoundary 包裹 App，捕获渲染错误避免白屏（详情见 components/ErrorBoundary.tsx）。
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
