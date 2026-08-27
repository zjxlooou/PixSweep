// 删除文件相关的 React hook：监听 delete-progress / delete-done 事件
import { useCallback, useEffect, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { api } from "../api";
import type { DeleteResult, ScanProgress } from "../types";

export interface DeleteState {
  deleting: boolean;
  progress: ScanProgress | null;
  result: DeleteResult | null;
  error: string | null;
}

/** 一键删除的进度状态（供删除按钮/进度条展示） */
export function useDelete() {
  const [state, setState] = useState<DeleteState>({
    deleting: false,
    progress: null,
    result: null,
    error: null,
  });

  useEffect(() => {
    let unlistenProgress: UnlistenFn | undefined;
    let unlistenDone: UnlistenFn | undefined;

    const setup = async () => {
      unlistenProgress = await listen<ScanProgress>("delete-progress", (event) => {
        setState((s) => ({ ...s, deleting: true, progress: event.payload, error: null }));
      });
      unlistenDone = await listen<DeleteResult>("delete-done", (event) => {
        setState((s) => ({ ...s, deleting: false, progress: null, result: event.payload }));
      });
    };

    setup();

    return () => {
      unlistenProgress?.();
      unlistenDone?.();
    };
  }, []);

  const deleteFiles = useCallback(async (paths: string[], permanent: boolean) => {
    setState((s) => ({ ...s, deleting: true, progress: null, result: null, error: null }));
    try {
      await api.deleteFiles(paths, permanent);
      // 结果通过 delete-done 事件回调（异步），这里不等待
    } catch (e) {
      setState((s) => ({ ...s, deleting: false, error: String(e) }));
      throw e;
    }
  }, []);

  // 消费掉结果（App 用完后置空，避免重复 toast）
  const consumeResult = useCallback(() => {
    setState((s) => ({ ...s, result: null }));
  }, []);

  return { state, deleteFiles, consumeResult };
}
