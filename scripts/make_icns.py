#!/usr/bin/env python3
"""从 .ico（或 .png）生成 macOS .icns 图标。

用法: python3 make_icns.py <源图标> <输出.icns>

- 取源文件中最大的一帧作为母版，高质量缩放出全部标准尺寸
- ICNS 使用 PNG 编码 chunk（ic07..ic14），macOS 10.7+ 均支持
"""
import io
import struct
import sys

from PIL import Image

# OSType -> 像素尺寸（覆盖 Finder 需要的全部标准档位，含 @2x）
SIZES = {
    "ic07": 128,    # 128x128
    "ic08": 256,    # 256x256
    "ic09": 512,    # 512x512
    "ic10": 1024,   # 1024x1024 (512@2x)
    "ic11": 32,     # 16@2x
    "ic12": 64,     # 32@2x
    "ic13": 256,    # 128@2x
    "ic14": 512,    # 256@2x
}


def load_master(path: str) -> Image.Image:
    img = Image.open(path)
    frames = getattr(img, "info", {}).get("variant") and [] or None
    # ICO 可含多帧：手动挑最大的一帧
    best = None
    if hasattr(img, "ico") or path.lower().endswith((".ico", ".cur")):
        try:
            sizes = img.info.get("sizes") or []
        except Exception:
            sizes = []
        if sizes:
            for s in sizes:
                img.size = s
                f = img.copy()
                if best is None or f.width * f.height > best.width * best.height:
                    best = f
    if best is None:
        # PIL 打开 ICO 默认就是最大帧；png/jpg 直接用
        best = img
    if best.mode not in ("RGBA", "RGB"):
        best = best.convert("RGBA")
    return best


def make_icns(src: str, dst: str) -> None:
    master = load_master(src)
    # 母版不足 1024 时放大补齐（LANCZOS），保证 ic10 档位存在
    if max(master.size) < 1024:
        scale = 1024 / max(master.size)
        master = master.resize(
            (round(master.width * scale), round(master.height * scale)),
            Image.LANCZOS,
        )
    # 统一成正方形画布，短边留透明
    side = max(master.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(master, ((side - master.width) // 2, (side - master.height) // 2))

    chunks = []
    for ostype, px in SIZES.items():
        im = canvas.resize((px, px), Image.LANCZOS)
        png = io.BytesIO()
        im.save(png, "PNG", optimize=True)
        data = png.getvalue()
        chunks.append(ostype.encode("ascii") + struct.pack(">I", len(data) + 8) + data)

    body = b"".join(chunks)
    with open(dst, "wb") as f:
        f.write(b"icns" + struct.pack(">I", len(body) + 8) + body)
    print(f"OK: {dst} <- {src} (master {canvas.width}x{canvas.height}, {len(SIZES)} 尺寸档)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    make_icns(sys.argv[1], sys.argv[2])
