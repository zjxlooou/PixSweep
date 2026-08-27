/**
 * GroupCard 测试：分组渲染、推荐标记、单条删除、手动模式红框、右键菜单
 */
// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { GroupCard } from "../GroupCard";
import type { ImageGroup } from "../../types";

// mock invoke：get_thumbnail 返回空 data URL
const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/core", () => ({ invoke }));

beforeEach(() => {
  invoke.mockReset();
  invoke.mockResolvedValue("data:image/jpeg;base64,AA==");
});

const group: ImageGroup = {
  group_id: "20260819000000-000001",
  images: [
    {
      info: { path: "E:/p/a.jpg", name: "a.jpg", size: 1024, modified: 0, width: 100, height: 100, format: "jpg", file_hash: "h1" },
      score: 8.5,
      aesthetic_score: 9.0,
      technical_score: 8.0,
      has_face: false,
      scene: 0,
      is_eye_closed: false,
      focus_score: 7.5,
      is_out_of_focus: false,
      recommended: true,
      reason: "综合评分最高",
    },
    {
      info: { path: "E:/p/b.jpg", name: "b.jpg", size: 2048, modified: 0, width: 100, height: 100, format: "jpg", file_hash: "h2" },
      score: 6.0,
      aesthetic_score: 6.0,
      technical_score: 6.0,
      has_face: false,
      scene: 0,
      is_eye_closed: false,
      focus_score: 4.2,
      is_out_of_focus: false,
      recommended: false,
      reason: "评分较低",
    },
  ],
  similarity: 0.98,
  reclaimable_bytes: 2048,
};

// 最小渲染：避免在每个测试重复
const renderCard = (
  overrides: Partial<React.ComponentProps<typeof GroupCard>> = {},
) => {
  const props = {
    group,
    onDeleteOne: vi.fn(),
    onBatchDelete: vi.fn(),
    onPreview: vi.fn(),
    disabled: false,
    ...overrides,
  };
  const utils = render(<GroupCard {...props} />);
  return { ...utils, props };
};

describe("GroupCard", () => {
  it("渲染组信息：相似度、图片数", () => {
    renderCard();
    expect(screen.getByText(/相似度 98.0%/)).toBeInTheDocument();
    expect(screen.getByText("2 张")).toBeInTheDocument();
  });

  it("默认模式：批量删除按钮显示非推荐图片数", () => {
    renderCard();
    // 显示 "删除 1 张" 按钮（默认标记 b.jpg）
    expect(screen.getByRole("button", { name: /删除\s*1\s*张/ })).toBeInTheDocument();
  });

  it("点击批量删除按钮传出非推荐图片路径", () => {
    const onBatchDelete = vi.fn();
    renderCard({ onBatchDelete });
    fireEvent.click(screen.getByRole("button", { name: /删除\s*1\s*张/ }));
    expect(onBatchDelete).toHaveBeenCalledWith(["E:/p/b.jpg"]);
  });

  it("disabled 时批量删除按钮不可用", () => {
    renderCard({ disabled: true });
    expect(screen.getByRole("button", { name: /删除/ })).toBeDisabled();
  });

  it("推荐图片标记可见", () => {
    renderCard();
    expect(screen.getAllByText(/推荐/).length).toBeGreaterThan(0);
  });

  it("单条 X 按钮调用 onDeleteOne", () => {
    const onDeleteOne = vi.fn();
    renderCard({ onDeleteOne });
    const xBtns = screen.getAllByTitle("删除此图片（无确认）");
    expect(xBtns.length).toBe(2); // 每张图一个 X 按钮
    fireEvent.click(xBtns[0]); // 点击 a.jpg 的 X
    expect(onDeleteOne).toHaveBeenCalledWith("E:/p/a.jpg");
  });

  it("手动模式：点击图片切换红框（默认无标记 → 标记 → 取消）", () => {
    renderCard();
    // 进入手动模式
    fireEvent.click(screen.getByRole("button", { name: /手动选择/ }));
    // 推荐图片 a.jpg 现在没有 marked-del 类（默认标记只针对非推荐）
    const images = screen.getAllByTitle("E:/p/a.jpg");
    expect(images[0]).not.toHaveClass("marked-del");
    // 点击 a.jpg 触发红框
    fireEvent.click(images[0]);
    expect(images[0]).toHaveClass("marked-del");
    // 再点取消
    fireEvent.click(images[0]);
    expect(images[0]).not.toHaveClass("marked-del");
  });

  it("右键图片弹出菜单：删除当前 / 删除其他", () => {
    const onDeleteOne = vi.fn();
    const onBatchDelete = vi.fn();
    renderCard({ onDeleteOne, onBatchDelete });
    // 右键 a.jpg
    const img = screen.getAllByTitle("E:/p/a.jpg")[0];
    fireEvent.contextMenu(img, { clientX: 100, clientY: 200 });
    // 菜单出现
    expect(screen.getByText("删除当前图片")).toBeInTheDocument();
    expect(screen.getByText(/删除其他\s*1\s*张/)).toBeInTheDocument();
    // 点击删除当前
    fireEvent.click(screen.getByText("删除当前图片"));
    expect(onDeleteOne).toHaveBeenCalledWith("E:/p/a.jpg");
  });

  it("右键菜单：删除其他传非目标图片路径", () => {
    const onDeleteOne = vi.fn();
    const onBatchDelete = vi.fn();
    renderCard({ onDeleteOne, onBatchDelete });
    fireEvent.contextMenu(screen.getAllByTitle("E:/p/a.jpg")[0], { clientX: 100, clientY: 200 });
    fireEvent.click(screen.getByText(/删除其他/));
    expect(onBatchDelete).toHaveBeenCalledWith(["E:/p/b.jpg"]);
  });

  it("渲染类型/综合/对焦相关标签", () => {
    renderCard();
    // 无脸 + scene=0 → 类型"其他"（两张图各一个）
    expect(screen.getAllByText("其他").length).toBe(2);
    // 综合评分
    expect(screen.getAllByText(/综合 8.5/).length).toBeGreaterThan(0);
  });

  it("人像且闭眼 → 显示 人像/闭眼 标签", () => {
    const g: ImageGroup = {
      ...group,
      images: [
        { ...group.images[0], has_face: true, scene: 1, is_eye_closed: true },
        group.images[1],
      ],
    };
    renderCard({ group: g });
    expect(screen.getAllByText("人像").length).toBeGreaterThan(0);
    expect(screen.getAllByText("闭眼").length).toBeGreaterThan(0);
  });

  it("宠物且失焦 → 显示 宠物/失焦 标签", () => {
    const g: ImageGroup = {
      ...group,
      images: [
        { ...group.images[0], scene: 2, is_out_of_focus: true },
        group.images[1],
      ],
    };
    renderCard({ group: g });
    expect(screen.getAllByText("宠物").length).toBeGreaterThan(0);
    expect(screen.getAllByText("失焦").length).toBeGreaterThan(0);
  });
});