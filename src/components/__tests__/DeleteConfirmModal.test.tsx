/**
 * DeleteConfirmModal 测试：键盘操作（Y/N/Enter/Esc）与渲染
 */
// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DeleteConfirmModal } from "../DeleteConfirmModal";

const paths = ["E:/photos/a.jpg", "E:/photos/b.jpg"];

function renderModal(overrides?: Partial<React.ComponentProps<typeof DeleteConfirmModal>>) {
  return render(
    <DeleteConfirmModal
      paths={paths}
      totalBytes={2048}
      permanent={false}
      onConfirm={vi.fn()}
      onCancel={vi.fn()}
      {...overrides}
    />,
  );
}

describe("DeleteConfirmModal", () => {
  it("显示待删除文件数与大小", () => {
    renderModal();
    expect(screen.getByText(/即将移至临时文件夹/)).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("按 Y 触发确认", () => {
    const onConfirm = vi.fn();
    renderModal({ onConfirm });
    fireEvent.keyDown(window, { key: "y" });
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("按 Enter 触发确认", () => {
    const onConfirm = vi.fn();
    renderModal({ onConfirm });
    fireEvent.keyDown(window, { key: "Enter" });
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("按 N 触发取消", () => {
    const onCancel = vi.fn();
    renderModal({ onCancel });
    fireEvent.keyDown(window, { key: "n" });
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("按 Esc 触发取消", () => {
    const onCancel = vi.fn();
    renderModal({ onCancel });
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("点击确认按钮触发 onConfirm", () => {
    const onConfirm = vi.fn();
    renderModal({ onConfirm });
    fireEvent.click(screen.getByRole("button", { name: /确认移至临时文件夹/ }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("点击取消按钮触发 onCancel", () => {
    const onCancel = vi.fn();
    renderModal({ onCancel });
    fireEvent.click(screen.getByRole("button", { name: /取消/ }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("永久删除模式显示不可逆提示", () => {
    renderModal({ permanent: true });
    expect(screen.getByText(/此操作不可逆/)).toBeInTheDocument();
  });

  it("超过 8 个文件显示省略提示", () => {
    const manyPaths = Array.from({ length: 10 }, (_, i) => `E:/p/${i}.jpg`);
    renderModal({ paths: manyPaths });
    expect(screen.getByText(/另外 2 个文件/)).toBeInTheDocument();
  });
});
