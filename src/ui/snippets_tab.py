"""
Snippets & Voice Macros tab: a card grid of trigger-phrase -> expansion-text
voice macros, each with a collapsible preview and an instant active/inactive
toggle. When a dictated transcript exactly matches a trigger phrase, the
expansion is injected immediately - the LLM transform step is skipped entirely.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QFrame, QScrollArea,
)

from src.ui import theme


def _label_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


class SnippetsTab(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.addWidget(_label_title("Snippets & Voice Macros"))
        desc = QLabel(
            f'<span style="color:{theme.TEXT_SECONDARY}">When your dictation exactly matches a '
            "trigger phrase, the expansion is injected immediately - the LLM transform step is "
            "skipped entirely.</span>"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        top_bar = QHBoxLayout()
        self.search_box = QLineEdit(placeholderText="Search snippets...")
        self.search_box.textChanged.connect(lambda _t: self.refresh())
        top_bar.addWidget(self.search_box, 1)
        layout.addLayout(top_bar)

        form_card = QWidget()
        form_card.setStyleSheet(f"background-color: {theme.BG_CARD}; border: 1px solid {theme.BORDER_SUBTLE}; border-radius: 10px;")
        form = QHBoxLayout(form_card)
        form.setContentsMargins(12, 12, 12, 12)
        self.trigger_in = QLineEdit(placeholderText='Trigger (e.g. "insert email")')
        self.expansion_in = QLineEdit(placeholderText="Expansion text")
        add_btn = QPushButton("+ Create Snippet")
        add_btn.setObjectName("primary")
        add_btn.clicked.connect(self._add)
        for w in (self.trigger_in, self.expansion_in, add_btn):
            form.addWidget(w)
        layout.addWidget(form_card)

        # An isolated scroll area for the card grid - own scrollbar, own
        # position, never leaks into or borrows from another tab's scroll state.
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        grid_container = QWidget()
        self.grid = QGridLayout(grid_container)
        self.grid.setSpacing(12)
        self.scroll.setWidget(grid_container)
        layout.addWidget(self.scroll, 1)

        self.refresh()

    def reset_scroll(self) -> None:
        self.scroll.verticalScrollBar().setValue(0)

    def refresh(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        query = self.search_box.text().strip().lower()
        rows = self.engine.db.list_snippets()
        if query:
            rows = [r for r in rows if query in r["trigger_phrase"].lower() or query in r["expansion_text"].lower()]

        for i, r in enumerate(rows):
            row, col = divmod(i, 2)
            self.grid.addWidget(self._snippet_card(r), row, col)

    def _snippet_card(self, r: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"background-color: {theme.BG_CARD}; border: 1px solid {theme.BORDER_SUBTLE}; border-radius: 10px;")
        v = QVBoxLayout(card)
        v.setContentsMargins(14, 12, 14, 12)

        header = QHBoxLayout()
        badge = QLabel(f'When I say "{r["trigger_phrase"]}"')
        badge.setStyleSheet(
            f"background-color: {theme.INPUT_BG}; color: {theme.ACCENT_PRIMARY}; "
            f"border-radius: 5px; padding: 3px 9px; font-size: 8pt; font-weight: 600;"
        )
        header.addWidget(badge)
        header.addStretch()
        active_cb = QCheckBox("Active")
        active_cb.setChecked(bool(r["is_active"]))
        active_cb.stateChanged.connect(lambda state, sid=r["id"]: self.engine.db.set_snippet_active(sid, state != 0))
        header.addWidget(active_cb)
        del_btn = QPushButton("\U0001F5D1")
        del_btn.setToolTip("Delete")
        del_btn.setStyleSheet(f"background: transparent; border: none; color: {theme.TEXT_SECONDARY};")
        del_btn.clicked.connect(lambda checked=False, sid=r["id"]: self._delete(sid))
        header.addWidget(del_btn)
        v.addLayout(header)

        preview_toggle = QPushButton("▶ Show expansion")
        preview_toggle.setFlat(True)
        preview_toggle.setStyleSheet(f"text-align: left; border: none; color: {theme.TEXT_SECONDARY}; padding: 4px 0;")
        preview = QLabel(r["expansion_text"])
        preview.setWordWrap(True)
        preview.setStyleSheet(f"color: {theme.TEXT_PRIMARY};")
        preview.setVisible(False)

        def _toggle(checked=False, btn=preview_toggle, lbl=preview):
            lbl.setVisible(not lbl.isVisible())
            btn.setText(("▼ Hide expansion" if lbl.isVisible() else "▶ Show expansion"))

        preview_toggle.clicked.connect(_toggle)
        v.addWidget(preview_toggle)
        v.addWidget(preview)

        return card

    def _add(self) -> None:
        trigger, expansion = self.trigger_in.text().strip(), self.expansion_in.text().strip()
        if not trigger or not expansion:
            return
        self.engine.db.add_snippet(trigger, expansion, True)
        self.trigger_in.clear()
        self.expansion_in.clear()
        self.refresh()

    def _delete(self, snippet_id: int) -> None:
        self.engine.db.delete_snippet(snippet_id)
        self.refresh()
