import React from "react";

interface Props {
  children: React.ReactNode;
}

interface State {
  error: Error | null;
}

/** 错误边界：捕获渲染异常，显示错误信息而不是白屏。 */
export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("PixSweep 渲染错误:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            padding: 24,
            color: "#b91c1c",
            fontFamily: "sans-serif",
            whiteSpace: "pre-wrap",
          }}
        >
          <h2>界面渲染出错</h2>
          <p>{String(this.state.error?.message ?? this.state.error)}</p>
          <button
            onClick={() => this.setState({ error: null })}
            style={{ padding: "6px 14px", cursor: "pointer" }}
          >
            重试
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
