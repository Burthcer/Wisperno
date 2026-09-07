"""
Custom Vocabulary & Dictionary tab: dark-glass layout matching History/
Transforms - search filter, a spoken-phrase -> replacement table with a
category badge, and icon actions to edit/delete each entry.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSizePolicy,
)

from src.ui import theme

CATEGORIES = ["Jargon", "Acronym", "Name", "Tech"]


def _label_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


def _badge(text: str) -> QLabel:
    """Bold uppercase purple TEXT tag - no background pill, matching the
    mockup's clean typography-only category tag."""
    label = QLabel(text.upper())
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet(f"color: {theme.ACCENT_SECONDARY}; font-weight: 700; font-size: 8pt; letter-spacing: 0.5px;")
    return label


def _text_link_btn(text: str, tooltip: str, color: str, hover_color: str) -> QPushButton:
    """Minimal lowercase text link ("edit"/"del") - no button chrome,
    matching the mockup's flat action links instead of icon buttons."""
    btn = QPushButton(text)
    btn.setToolTip(tooltip)
    btn.setFlat(True)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(
        f"QPushButton {{ background: transparent; border: none; color: {color}; font-size: 9pt; padding: 2px 4px; }}"
        f"QPushButton:hover {{ color: {hover_color}; text-decoration: underline; }}"
    )
    return btn


class DictionaryTab(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._editing_id = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.addWidget(_label_title("Custom Vocabulary & Dictionary"))
        layout.addWidget(QLabel(
            f'<span style="color:{theme.TEXT_SECONDARY}">Phonetic corrections and jargon Whisper '
            "should recognize and auto-replace.</span>"
        ))

        top_bar = QHBoxLayout()
        self.search_box = QLineEdit(placeholderText="Search dictionary...")
        self.search_box.textChanged.connect(lambda _t: self.refresh())
        top_bar.addWidget(self.search_box, 1)
        layout.addLayout(top_bar)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["SPOKEN", "REPLACEMENT", "CATEGORY", "ACTIONS"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # Category/Actions were left on Qt's default Interactive mode with no
        # explicit width - with an empty header label (the old "" for Actions),
        # Qt auto-sizes an Interactive column from its header text, which
        # collapsed this column to near-zero width and clipped the edit/delete
        # buttons out of view. Fixed widths make both columns always show fully.
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 110)
        self.table.setColumnWidth(3, 100)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Expanding + a layout stretch factor of 1 makes the table (not the whole
        # tab) absorb any extra vertical space, so it scrolls internally via its
        # own built-in scrollbar once the list outgrows that space - the fixed
        # dock below stays pinned on screen instead of getting pushed off the
        # bottom of a tab that grows taller than the window with every word added.
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.table, 1)

        form_card = QWidget()
        form_card.setStyleSheet(f"background-color: {theme.BG_CARD}; border: 1px solid {theme.BORDER_SUBTLE}; border-radius: 10px;")
        form = QHBoxLayout(form_card)
        form.setContentsMargins(12, 12, 12, 12)
        self.word_in = QLineEdit(placeholderText='Spoken (e.g. "pie torch")')
        self.repl_in = QLineEdit(placeholderText="Replacement (e.g. PyTorch)")
        self.case_cb = QCheckBox("Case-sensitive")
        self.category_cb = QComboBox()
        self.category_cb.wheelEvent = lambda event: event.ignore()  # scrolling the page must not silently change the selected category
        self.category_cb.addItems(CATEGORIES)
        self.add_btn = QPushButton("+ Add Word")
        self.add_btn.setObjectName("primary")
        self.add_btn.clicked.connect(self._add)
        for w in (self.word_in, self.repl_in, self.case_cb, self.category_cb, self.add_btn):
            form.addWidget(w)
        layout.addWidget(form_card, 0)  # fixed dock: never grows/shrinks, always visible

        self.refresh()

    def reset_scroll(self) -> None:
        self.table.verticalScrollBar().setValue(0)

    def refresh(self) -> None:
        query = self.search_box.text().strip().lower()
        entries = self.engine.db.list_dictionary()
        if query:
            entries = [e for e in entries if query in e["word"].lower() or query in e["replacement"].lower()]

        self.table.setRowCount(len(entries))
        mono_font = QFont(theme.FONT_FAMILY_MONO)
        for i, e in enumerate(entries):
            spoken_item = QTableWidgetItem(e["word"])
            spoken_item.setFont(mono_font)
            self.table.setItem(i, 0, spoken_item)
            repl_item = QTableWidgetItem(f'→ {e["replacement"]}')
            repl_item.setForeground(QColor(theme.TEXT_PRIMARY))
            self.table.setItem(i, 1, repl_item)
            self.table.setCellWidget(i, 2, _badge(e["category"]))

            actions = QWidget()
            actions_row = QHBoxLayout(actions)
            actions_row.setContentsMargins(0, 0, 0, 0)
            edit_btn = _text_link_btn("edit", "Edit", theme.TEXT_SECONDARY, theme.TEXT_PRIMARY)
            edit_btn.clicked.connect(lambda checked=False, entry=e: self._start_edit(entry))
            del_btn = _text_link_btn("del", "Delete", theme.TEXT_SECONDARY, theme.DANGER)
            del_btn.clicked.connect(lambda checked=False, eid=e["id"]: self._delete(eid))
            actions_row.addWidget(edit_btn)
            actions_row.addWidget(del_btn)
            self.table.setCellWidget(i, 3, actions)

    def _start_edit(self, entry: dict) -> None:
        """Loads the entry into the form for editing - saving re-adds it (word is the
        dictionary's natural key), so no separate update path is needed."""
        self.word_in.setText(entry["word"])
        self.repl_in.setText(entry["replacement"])
        self.case_cb.setChecked(bool(entry["case_sensitive"]))
        idx = self.category_cb.findText(entry["category"])
        if idx >= 0:
            self.category_cb.setCurrentIndex(idx)
        self._editing_id = entry["id"]
        self.add_btn.setText("Save Word")

    def _add(self) -> None:
        word, repl = self.word_in.text().strip(), self.repl_in.text().strip()
        if not word or not repl:
            return
        if self._editing_id is not None:
            self.engine.vocabulary.remove_entry(self._editing_id)
            self._editing_id = None
            self.add_btn.setText("+ Add Word")
        self.engine.vocabulary.add_entry(word, repl, self.case_cb.isChecked(), self.category_cb.currentText())
        self.word_in.clear()
        self.repl_in.clear()
        self.refresh()

    def _delete(self, entry_id: int) -> None:
        self.engine.vocabulary.remove_entry(entry_id)
        self.refresh()
