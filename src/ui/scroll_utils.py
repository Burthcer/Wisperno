"""
Shared helper for giving a dashboard tab its own isolated, internal scroll
area - own scrollbar, own scroll position, never shared with another tab's
(see main_window.py's MainWindow._on_nav_changed / content_scroll comments
for the cross-tab scroll-leakage bug this pattern fixes).
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame, QScrollArea


def make_scrollable_layout(outer: QWidget) -> QVBoxLayout:
    """Wraps `outer`'s content in its own QScrollArea and returns the inner
    QVBoxLayout to build content on - a drop-in replacement for
    QVBoxLayout(outer). Stores the QScrollArea as `outer._scroll_area` so
    callers can reset scroll position (`outer._scroll_area.verticalScrollBar().setValue(0)`)
    on tab switch, e.g. via a `reset_scroll()` method."""
    outer_layout = QVBoxLayout(outer)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    outer_layout.addWidget(scroll)
    container = QWidget()
    inner_layout = QVBoxLayout(container)
    scroll.setWidget(container)
    outer._scroll_area = scroll
    return inner_layout
