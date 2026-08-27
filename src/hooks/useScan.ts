// 扫描相关的 React hook
import { useCallback, useEffect, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { api } from "../api";
import type { ScanProgress, ScanResult, ScanSummary } from "../types";

export interface ScanState {
  scanning: boolean;
  progress: ScanProgress | null;
  result: ScanResult | null;
  error: string | null;
}

export function useScan() {
  const [state, setState] = useState<ScanState>({
    scanning: false,
    progress: null,
    result: null,
    error: null,
  });

  useEffect(() => {
    let unlistenProgress: UnlistenFn | undefined;
    let unlistenComplete: UnlistenFn | undefined;

    const setup = async () => {
      unlistenProgress = await listen<ScanProgress>("scan-progress", (event) => {
        setState((s) => ({ ...s, progress: event.payload, scanning: true }));
      });
      unlistenComplete = await listen<ScanSummary>("scan-complete", async () => {
        // 事件只带摘要（小 payload），完整结果通过 invoke 拉取，避免大 payload 白屏
        try {
          const result = await api.getScanResult();
          setState({
            scanning: false,
            progress: null,
            result: result ?? null,
            error: null,
          });
        } catch (e) {
          setState({
            scanning: false,
            progress: null,
            result: null,
            error: String(e),
          });
        }
      });
    };

    setup();

    return () => {
      unlistenProgress?.();
      unlistenComplete?.();
    };
  }, []);

  const startScan = useCallback(
    async (folders: string[], threshold?: number, incremental?: boolean) => {
      setState((s) => ({ ...s, scanning: true, error: null, result: null }));
      try {
        await api.startScan(folders, threshold, incremental);
      } catch (e) {
        setState((s) => ({
          ...s,
          scanning: false,
          error: String(e),
        }));
      }
    },
    [],
  );

  return { state, startScan };
}
