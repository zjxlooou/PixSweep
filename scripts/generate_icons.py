"""生成 PixSweep 应用图标（纯 Python 标准库，无需 PIL）。

生成：
- icons/32x32.png
- icons/128x128.png
- icons/128x128@2x.png (256x256)
- icons/icon.ico (内嵌 256x256 PNG)

图标为蓝色到紫色的对角渐变 + 白色 "PS" 字样（简化为圆角方块图案）。
"""
import os
import struct
import zlib

ICON_DIR = os.path.join(os.path.dirname(__file__), "..", "src-tauri", "icons")
ICON_DIR = os.path.abspath(ICON_DIR)


def make_png(width, height, rgba_rows):
    """从 RGBA 像素行构造 PNG 字节。rgba_rows 为 height x width 的 RGBA 列表。"""
    raw = b"".join(
        b"\x00" + b"".join(struct.pack("4B", *px) for px in row)
        for row in rgba_rows
    )

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8bit RGBA
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def lerp(a, b, t):
    return int(a + (b - a) * t)


def render(size):
    """渲染 size x size 的图标，返回 RGBA 像素行列表。"""
    # 对角渐变：左上蓝 #2563eb -> 右下紫 #7c3aed
    c1 = (0x25, 0x63, 0xEB)
    c2 = (0x7C, 0x3A, 0xED)
    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            r = lerp(c1[0], c2[0], t)
            g = lerp(c1[1], c2[1], t)
            b = lerp(c1[2], c2[2], t)
            row.append((r, g, b, 255))
        rows.append(row)

    # 绘制白色圆角方块（代表 "PS" 的简化图案：一个圆形 + 一个方块）
    half = size // 2
    margin = size // 6
    for y in range(margin, size - margin):
        for x in range(margin, size - margin):
            # 圆形（左边）
            cx, cy = size * 2 // 5, size // 2
            r = size // 5
            dx, dy = x - cx, y - cy
            if dx * dx + dy * dy <= r * r:
                rows[y][x] = (255, 255, 255, 255)
            # 方块（右边）
            bx0, bx1 = size * 3 // 5, size * 4 // 5
            by0, by1 = size // 3, size * 2 // 3
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                rows[y][x] = (255, 255, 255, 255)
    return rows


def write_ico(png_bytes_256):
    """构造包含单个 256x256 PNG 的 ICO 文件。"""
    # ICO 头
    header = struct.pack("<HHH", 0, 1, 1)  # reserved, type=icon, count=1
    # 目录项（256 尺寸用 0 表示）
    entry = struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(png_bytes_256), 6 + 16)
    return header + entry + png_bytes_256


def main():
    os.makedirs(ICON_DIR, exist_ok=True)
    sizes = {32: "32x32.png", 128: "128x128.png", 256: "128x128@2x.png"}
    for size, name in sizes.items():
        png = make_png(size, size, render(size))
        path = os.path.join(ICON_DIR, name)
        with open(path, "wb") as f:
            f.write(png)
        print(f"生成 {path} ({size}x{size})")

    # ICO
    ico_png = make_png(256, 256, render(256))
    ico = write_ico(ico_png)
    ico_path = os.path.join(ICON_DIR, "icon.ico")
    with open(ico_path, "wb") as f:
        f.write(ico)
    print(f"生成 {ico_path}")

    print("图标生成完成")


if __name__ == "__main__":
    main()
