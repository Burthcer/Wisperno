"""
Writing Styles HUD (Alt+V): a frameless floating modal showing the original
selection alongside cards for each style in src/writing_styles.WRITING_STYLES,
each populated progressively as its own LLM call finishes
(src/workers.py:WritingStylesWorker). Purely a view + signals - all capture/
generation/injection/history logic lives in engine.py and the worker, which
this window's owner (main.py) wires to show_selection()/set_style_ready()/
close_hud().

Simplification, disclosed rather than hidden: "shimmer/skeleton loader" is a
plain muted "Generating..." placeholder, not an animated skeleton - a real
shimmer animation is a pure cosmetic upgrade with no functional difference,
not worth the added QPropertyAnimation machinery for this feature's first cut.
"""

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeyEvent
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from src.ui import theme
from src.writing_styles import WRITING_STYLES

GENERATING_PLACEHOLDER = "Generating..."

QSS = f"""
QWidget#stylesRoot {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1A1A28, stop:0.06 #13131D, stop:1 #111116);
    border: 1px solid {theme.BORDER_SUBTLE};
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
}}
QLabel#stylesTitle {{ color: {theme.TEXT_PRIMARY}; font-weight: 700; font-size: 12pt; }}
QLabel#stylesHint {{ color: {theme.TEXT_SECONDARY}; font-size: 9pt; }}
QFrame#originalDock {{
    background-color: {theme.INPUT_BG};
    border: 1px solid {theme.INPUT_BORDER};
    border-radius: 8px;
}}
QLabel#originalText {{ color: {theme.TEXT_SECONDARY}; font-size: 9pt; }}
QPushButton#originalToggle {{
    background: transparent; border: none; color: {theme.TEXT_SECONDARY};
    font-size: 9pt; text-align: left; padding: 2px;
}}
QFrame#styleCard {{
    background-color: #161622;
    border: 1px solid {theme.BORDER_SUBTLE};
    border-radius: 10px;
}}
QFrame#styleCard[focused="true"] {{
    border: 1px solid {theme.ACCENT_PRIMARY};
}}
QLabel#styleTitle {{ color: {theme.TEXT_PRIMARY}; font-weight: 700; font-size: 10pt; }}
QLabel#styleKeyHint {{
    color: {theme.ACCENT_PRIMARY}; background-color: rgba(168, 85, 247, 0.14);
    border-radius: 4px; padding: 1px 6px; font-family: {theme.FONT_FAMILY_MONO}; font-size: 8pt; font-weight: 700;
}}
QLabel#stylePreview {{ color: {theme.TEXT_PRIMARY}; font-size: 9pt; }}
QLabel#stylePreview[generating="true"] {{ color: {theme.TEXT_SECONDARY}; font-style: italic; }}
QPushButton#insertBtn {{
    background-color: {theme.ACCENT_PRIMARY}; border: none; border-radius: 6px;
    color: white; font-weight: 600; padding: 4px 12px; font-size: 9pt;
}}
QPushButton#insertBtn:hover {{ background-color: {theme.ACCENT_PRIMARY_HOVER}; }}
QPushButton#insertBtn:disabled {{ background-color: {theme.BG_CARD_HOVER}; color: {theme.TEXT_SECONDARY}; }}
QScrollArea {{ background: transparent; border: none; }}
"""


class _StyleCardWidget(QFrame):
    chosen = Signal(str)  # style_id

    def __init__(self, style_id: str, title: str, icon: str, key_hint: str, parent=None):
        super().__init__(parent)
        self.style_id = style_id
        self.setObjectName("styleCard")
        self.setProperty("focused", False)
        self._styled_text: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        top = QHBoxLayout()
        heading = QLabel(f"{icon}  {title}")
        heading.setObjectName("styleTitle")
        top.addWidget(heading)
        top.addStretch()
        key_badge = QLabel(f"[{key_hint}]")
        key_badge.setObjectName("styleKeyHint")
        top.addWidget(key_badge)
        layout.addLayout(top)

        self.preview_label = QLabel(GENERATING_PLACEHOLDER)
        self.preview_label.setObjectName("stylePreview")
        self.preview_label.setProperty("generating", True)
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        bottom = QHBoxLayout()
        bottom.addStretch()
        self.insert_btn = QPushButton("Insert")
        self.insert_btn.setObjectName("insertBtn")
        self.insert_btn.setEnabled(False)
        self.insert_btn.clicked.connect(lambda: self.chosen.emit(self.style_id))
        bottom.addWidget(self.insert_btn)
        layout.addLayout(bottom)

    def set_ready(self, styled_text: str) -> None:
        self._styled_text = styled_text
        self.preview_label.setText(styled_text)
        self.preview_label.setProperty("generating", False)
        self._repolish(self.preview_label)
        self.insert_btn.setEnabled(True)

    def set_generating(self) -> None:
        self._styled_text = None
        self.preview_label.setText(GENERATING_PLACEHOLDER)
        self.preview_label.setProperty("generating", True)
        self._repolish(self.preview_label)
        self.insert_btn.setEnabled(False)

    def set_focused(self, focused: bool) -> None:
        self.setProperty("focused", focused)
        self._repolish(self)

    @property
    def is_ready(self) -> bool:
        return self._styled_text is not None

    @staticmethod
    def _repolish(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class WritingStylesWindow(QWidget):
    """Purely a view + signals - engine.py owns capture/generation/injection/history."""

    style_chosen = Signal(str, str)  # (style_id, styled_text)
    dismissed = Signal()

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("stylesRoot")
        self.setStyleSheet(QSS)
        self.setFixedWidth(420)
        self._original_expanded = False
        self._focused_index = 0
        self._cards: Dict[str, _StyleCardWidget] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("✨  Writing Styles")
        title.setObjectName("stylesTitle")
        header.addWidget(title)
        header.addStretch()
        hint = QLabel("[1-5] select   [Esc] close")
        hint.setObjectName("stylesHint")
        header.addWidget(hint)
        root.addLayout(header)

        self.original_toggle = QPushButton("▸ Original text")
        self.original_toggle.setObjectName("originalToggle")
        self.original_toggle.clicked.connect(self._toggle_original)
        root.addWidget(self.original_toggle)

        self.original_dock = QFrame()
        self.original_dock.setObjectName("originalDock")
        dock_layout = QVBoxLayout(self.original_dock)
        dock_layout.setContentsMargins(10, 8, 10, 8)
        self.original_label = QLabel("")
        self.original_label.setObjectName("originalText")
        self.original_label.setWordWrap(True)
        dock_layout.addWidget(self.original_label)
        self.original_dock.setVisible(False)
        root.addWidget(self.original_dock)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(420)
        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)
        self._card_order = []
        for style in WRITING_STYLES:
            card = _StyleCardWidget(style.id, style.title, style.icon, style.key_hint)
            card.chosen.connect(self._on_card_chosen)
            cards_layout.addWidget(card)
            self._cards[style.id] = card
            self._card_order.append(style.id)
        cards_layout.addStretch()
        scroll.setWidget(cards_container)
        root.addWidget(scroll)

        self._key_to_style = {style.key_hint: style.id for style in WRITING_STYLES}
        self._update_focus_ring()

    def _toggle_original(self) -> None:
        self._original_expanded = not self._original_expanded
        self.original_dock.setVisible(self._original_expanded)
        self.original_toggle.setText(("▾" if self._original_expanded else "▸") + " Original text")

    def show_selection(self, raw_text: str) -> None:
        """Called when the worker has captured a selection - show the HUD now,
        every card starting in its 'generating' state."""
        self.original_label.setText(raw_text)
        self._original_expanded = False
        self.original_dock.setVisible(False)
        self.original_toggle.setText("▸ Original text")
        for card in self._cards.values():
            card.set_generating()
        self._focused_index = 0
        self._update_focus_ring()

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.adjustSize()
            self.move(geo.center().x() - self.width() // 2, geo.center().y() - self.height() // 2)
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def set_style_ready(self, style_id: str, styled_text: str) -> None:
        card = self._cards.get(style_id)
        if card is not None:
            card.set_ready(styled_text)

    def close_hud(self) -> None:
        self.hide()

    def _on_card_chosen(self, style_id: str) -> None:
        card = self._cards.get(style_id)
        if card is None or not card.is_ready:
            return
        self.style_chosen.emit(style_id, card._styled_text)
        self.close_hud()

    def _update_focus_ring(self) -> None:
        for i, style_id in enumerate(self._card_order):
            self._cards[style_id].set_focused(i == self._focused_index)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key_text = event.text()
        if key_text in self._key_to_style:
            style_id = self._key_to_style[key_text]
            card = self._cards.get(style_id)
            if card is not None and card.is_ready:
                self._on_card_chosen(style_id)
            return
        if event.key() == Qt.Key.Key_Escape:
            self.close_hud()
            self.dismissed.emit()
            return
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Tab):
            self._focused_index = (self._focused_index + 1) % len(self._card_order)
            self._update_focus_ring()
            return
        if event.key() == Qt.Key.Key_Up:
            self._focused_index = (self._focused_index - 1) % len(self._card_order)
            self._update_focus_ring()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_card_chosen(self._card_order[self._focused_index])
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        # "clicking outside the modal closes the HUD cleanly" - a Tool-flagged
        # frameless window loses focus the moment the user clicks anywhere
        # else, so this is the natural hook for that requirement.
        self.close_hud()
        self.dismissed.emit()
        super().focusOutEvent(event)
