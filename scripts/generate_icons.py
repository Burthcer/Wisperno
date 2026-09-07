"""
Generates Wisperno's app icon: a hand-authored SVG source (assets/icon.svg)
rasterized via Qt's own QtSvg renderer (already a PySide6 dependency - no new
package needed) at each target resolution, then packed into a real
multi-resolution .ico container.

Writes to both the paths the running app/build actually wire up
(assets/icons/wisperno.ico, assets/icons/wisperno.png) and the paths named in
the mission brief (assets/icon.ico, assets/icon.png) - identical content,
just two locations.

Usage: python scripts/generate_icons.py
"""

import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

ICON_SIZES = [16, 32, 48, 64, 128, 256]

SVG_SOURCE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <defs>
    <linearGradient id="mark" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#818CF8"/>
      <stop offset="55%" stop-color="#6366F1"/>
      <stop offset="100%" stop-color="#A855F7"/>
    </linearGradient>
  </defs>
  <circle cx="128" cy="128" r="122" fill="#12121A"/>
  <circle cx="128" cy="128" r="122" fill="none" stroke="url(#mark)" stroke-width="3" opacity="0.5"/>
  <path d="M 66 74 L 96 178 L 128 104 L 160 178 L 190 74"
        fill="none" stroke="url(#mark)" stroke-width="17"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M 200 46 L 204 56 L 214 60 L 204 64 L 200 74 L 196 64 L 186 60 L 196 56 Z" fill="url(#mark)"/>
  <path d="M 54 182 L 57 189 L 64 192 L 57 195 L 54 202 L 51 195 L 44 192 L 51 189 Z" fill="url(#mark)"/>
</svg>
"""


def render_svg_to_pngs(svg_path: Path) -> dict:
    """Returns {size: png_bytes} for each of ICON_SIZES, rendered from the SVG
    via Qt's own rasterizer for pixel-accurate parity with the source file."""
    from PySide6.QtCore import QByteArray, QBuffer, QIODevice
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    renderer = QSvgRenderer(str(svg_path))

    pngs = {}
    for size in ICON_SIZES:
        img = QImage(size, size, QImage.Format.Format_ARGB32)
        img.fill(0)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        renderer.render(painter)
        painter.end()

        buf = QByteArray()
        qbuf = QBuffer(buf)
        qbuf.open(QIODevice.OpenModeFlag.WriteOnly)
        img.save(qbuf, "PNG")
        pngs[size] = bytes(buf)
    return pngs


def write_ico(pngs: dict, dest: Path) -> None:
    """Hand-assembled ICO container (PNG-compressed frames, supported since
    Vista) - no new dependency needed for a ~20-line, well-defined format."""
    sizes = sorted(pngs.keys())
    count = len(sizes)
    header = struct.pack("<HHH", 0, 1, count)  # reserved, type=icon, count

    entries = b""
    offset = 6 + 16 * count
    for size in sizes:
        data = pngs[size]
        wh = size if size < 256 else 0  # 0 means 256 per the ICO spec
        entries += struct.pack("<BBBBHHII", wh, wh, 0, 0, 1, 32, len(data), offset)
        offset += len(data)

    with open(dest, "wb") as f:
        f.write(header)
        f.write(entries)
        for size in sizes:
            f.write(pngs[size])


def main() -> None:
    svg_path = BASE_DIR / "assets" / "icon.svg"
    svg_path.write_text(SVG_SOURCE, encoding="utf-8")
    print(f"Wrote {svg_path}")

    pngs = render_svg_to_pngs(svg_path)

    for dest_dir, name in [
        (BASE_DIR / "assets" / "icons", "wisperno"),  # wired into theme.py / wisperno.spec
        (BASE_DIR / "assets", "icon"),  # mission-named path
    ]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        write_ico(pngs, dest_dir / f"{name}.ico")
        (dest_dir / f"{name}.png").write_bytes(pngs[256])
        print(f"Wrote {dest_dir / f'{name}.ico'} ({sorted(pngs.keys())}) and {dest_dir / f'{name}.png'}")


if __name__ == "__main__":
    main()
