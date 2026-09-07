"""
Small in-memory line icons for the dashboard sidebar - drawn with QPainter
(same approach as src/ui/tray.py's tray icon) rather than bundling SVG
assets or adding a QtSvg dependency for five simple glyphs.
"""

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen

_SIZE = 20


def _new_painter(color: str) -> tuple:
    pix = QPixmap(_SIZE, _SIZE)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(QPen(QColor(color), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    return pix, p


def history_icon(color: str) -> QIcon:
    pix, p = _new_painter(color)
    p.drawEllipse(QRectF(3, 3, 14, 14))
    p.drawLine(10, 7, 10, 10)
    p.drawLine(10, 10, 13, 12)
    p.end()
    return QIcon(pix)


def dictionary_icon(color: str) -> QIcon:
    pix, p = _new_painter(color)
    p.drawLine(10, 4, 10, 16)
    p.drawRoundedRect(QRectF(3, 4, 7, 12), 1.5, 1.5)
    p.drawRoundedRect(QRectF(10, 4, 7, 12), 1.5, 1.5)
    p.end()
    return QIcon(pix)


def transforms_icon(color: str) -> QIcon:
    """Sparkle/wand glyph, matching the mission's 'Sparkles / Wand' spec for Transforms."""
    pix, p = _new_painter(color)
    cx, cy = 10.0, 10.0
    for dx, dy in [(0, -6), (5, 3), (-5, 3)]:
        p.drawLine(int(cx), int(cy), int(cx + dx), int(cy + dy))
    p.setBrush(QColor(color))
    p.drawEllipse(QRectF(cx - 1.5, cy - 1.5, 3, 3))
    p.end()
    return QIcon(pix)


def scissors_icon(color: str) -> QIcon:
    pix, p = _new_painter(color)
    p.drawEllipse(QRectF(3, 4, 4, 4))
    p.drawEllipse(QRectF(3, 12, 4, 4))
    p.drawLine(7, 6, 17, 15)
    p.drawLine(7, 14, 17, 5)
    p.end()
    return QIcon(pix)


def settings_icon(color: str) -> QIcon:
    pix, p = _new_painter(color)
    p.drawEllipse(QRectF(6.5, 6.5, 7, 7))
    p.drawEllipse(QRectF(9, 9, 2, 2))
    import math
    for i in range(8):
        angle = i * math.pi / 4
        x1, y1 = 10 + 7.5 * math.cos(angle), 10 + 7.5 * math.sin(angle)
        x2, y2 = 10 + 9.5 * math.cos(angle), 10 + 9.5 * math.sin(angle)
        p.drawLine(int(x1), int(y1), int(x2), int(y2))
    p.end()
    return QIcon(pix)


def usage_icon(color: str) -> QIcon:
    """Simple bar-chart/gauge glyph for the System Usage tab."""
    pix, p = _new_painter(color)
    p.drawLine(5, 16, 5, 10)
    p.drawLine(10, 16, 10, 6)
    p.drawLine(15, 16, 15, 12)
    p.drawLine(3, 16, 17, 16)
    p.end()
    return QIcon(pix)


NAV_ICONS = {
    "History": history_icon,
    "Transforms": transforms_icon,
    "Dictionary": dictionary_icon,
    "Snippets": scissors_icon,
    "Settings": settings_icon,
    "System Usage": usage_icon,
}
