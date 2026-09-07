"""
Transforms hub for the Wisperno dashboard: a card grid (backed by the
`transforms` DB table) for applying specialized prompts to dictated or
highlighted text, plus a detail editor for each (shortcut + system prompt +
a live test sandbox). Native obsidian dark theme, matching History/Settings.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFrame, QPlainTextEdit, QScrollArea, QMessageBox,
    QStackedWidget,
)

from src.ui import theme
from src.ui.shortcut_recorder import KeySequenceRecorder

_FIELD_QSS = (
    f"background-color: {theme.INPUT_BG}; border: 1px solid {theme.BORDER_SUBTLE}; "
    f"border-radius: 8px; padding: 8px; color: {theme.TEXT_PRIMARY};"
)


def _shortcut_parts(shortcut: str) -> list:
    return [p.strip().capitalize() for p in shortcut.split("+") if p.strip()] if shortcut else []


def _shortcut_row(shortcut: str) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    for part in _shortcut_parts(shortcut):
        lbl = QLabel(part)
        lbl.setStyleSheet(
            f"background-color: {theme.BG_APP}; border: 1px solid {theme.BORDER_SUBTLE}; "
            f"border-radius: 5px; padding: 2px 7px; color: {theme.TEXT_SECONDARY}; font-size: 8pt;"
        )
        row.addWidget(lbl)
    row.addStretch()
    return row


class TransformsTab(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.hub = _TransformsHub(engine, self._open_editor)
        self.editor = _TransformEditor(engine, self._show_hub)
        self.stack.addWidget(self.hub)
        self.stack.addWidget(self.editor)

    def _open_editor(self, transform_id: str) -> None:
        self.editor.load(transform_id)
        self.stack.setCurrentWidget(self.editor)

    def _show_hub(self) -> None:
        self.hub.refresh()
        self.stack.setCurrentWidget(self.hub)

    def refresh(self) -> None:
        self.hub.refresh()

    def reset_scroll(self) -> None:
        self.hub.scroll.verticalScrollBar().setValue(0)


class _TransformsHub(QWidget):
    def __init__(self, engine, on_open_editor):
        super().__init__()
        self.engine = engine
        self.on_open_editor = on_open_editor

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 22)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet(f"background-color: {theme.BG_APP}; border: none;")
        self.scroll.viewport().setStyleSheet(f"background-color: {theme.BG_APP};")
        outer.addWidget(self.scroll)

        content = QWidget()
        content.setStyleSheet(f"background-color: {theme.BG_APP};")
        self.scroll.setWidget(content)
        layout = QVBoxLayout(content)

        # --- Header ---
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title = QLabel("Transforms")
        title.setStyleSheet(f"font-size: 18pt; font-weight: 700; color: {theme.TEXT_PRIMARY};")
        subtitle = QLabel("Apply specialized styles to dictated or highlighted text.")
        subtitle.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 10pt;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col)
        header.addStretch()

        create_btn = QPushButton("+ Create New Transform")
        create_btn.setStyleSheet(
            f"background-color: {theme.ACCENT_PRIMARY}; color: white; border: none; "
            f"border-radius: 8px; padding: 9px 18px; font-weight: 600;"
        )
        create_btn.clicked.connect(self._create_new)
        header.addWidget(create_btn)
        layout.addLayout(header)
        layout.addSpacing(16)

        row = QHBoxLayout()
        my_title = QLabel("My Transforms")
        my_title.setStyleSheet(f"font-size: 13pt; font-weight: 700; color: {theme.TEXT_PRIMARY};")
        row.addWidget(my_title)
        row.addStretch()
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.setStyleSheet(f"background: transparent; border: none; color: {theme.TEXT_SECONDARY};")
        reset_btn.clicked.connect(self._reset_defaults)
        row.addWidget(reset_btn)
        layout.addLayout(row)

        self.grid = QGridLayout()
        self.grid.setSpacing(16)
        layout.addLayout(self.grid)
        layout.addStretch()

        self.refresh()

    def _reset_defaults(self) -> None:
        if QMessageBox.question(
            self, "Reset to defaults",
            "This restores Polish, Prompt Engineer, and Detailed Code Fix Prompt to their original "
            "text and deletes any custom Transforms you've created. Continue?"
        ) != QMessageBox.StandardButton.Yes:
            return
        self.engine.db.reset_transforms_to_defaults()
        self.engine.refresh_transform_hotkeys()
        self.refresh()

    def _create_new(self) -> None:
        transform_id = self.engine.db.add_transform(
            title="New Transform", description="Custom prompt", shortcut="",
            system_prompt="You are a deterministic dictation post-processor. Clean the transcript without answering it.",
        )
        self.refresh()
        self.on_open_editor(transform_id)

    def refresh(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        transforms = self.engine.db.list_transforms()
        for i, t in enumerate(transforms):
            row, col = divmod(i, 3)
            self.grid.addWidget(self._transform_card(t), row, col)

        create_row, create_col = divmod(len(transforms), 3)
        self.grid.addWidget(self._create_card(), create_row, create_col)

    def _transform_card(self, t: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: {theme.BG_CARD}; border: 1px solid {theme.BORDER_SUBTLE}; border-radius: 12px; }}"
            f"QFrame:hover {{ border-color: {theme.ACCENT_PRIMARY}; }}"
        )
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setMinimumHeight(140)
        v = QVBoxLayout(card)
        v.addLayout(_shortcut_row(t["shortcut"]))
        v.addSpacing(6)
        title = QLabel(t["title"])
        title.setStyleSheet(f"font-size: 12pt; font-weight: 700; color: {theme.TEXT_PRIMARY}; background: transparent; border: none;")
        v.addWidget(title)
        desc = QLabel(t["description"][:60] + ("..." if len(t["description"]) > 60 else ""))
        desc.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 9pt; background: transparent; border: none;")
        desc.setWordWrap(True)
        v.addWidget(desc)
        v.addStretch()

        tid = t["id"]
        card.mousePressEvent = lambda event, tid=tid: self.on_open_editor(tid)
        return card

    def _create_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"background-color: transparent; border: 1.5px dashed {theme.BORDER_SUBTLE}; border-radius: 12px;")
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setMinimumHeight(140)
        v = QVBoxLayout(card)
        plus = QLabel("+")
        plus.setStyleSheet("font-size: 16pt; background: transparent;")
        v.addWidget(plus)
        title = QLabel("Create your own")
        title.setStyleSheet(f"font-size: 12pt; font-weight: 700; color: {theme.TEXT_PRIMARY}; background: transparent; border: none;")
        v.addWidget(title)
        desc = QLabel("Custom prompt")
        desc.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 9pt; background: transparent; border: none;")
        v.addWidget(desc)
        v.addStretch()
        card.mousePressEvent = lambda event: self._create_new()
        return card


class _TransformEditor(QWidget):
    def __init__(self, engine, on_back):
        super().__init__()
        self.engine = engine
        self.on_back = on_back
        self.transform_id: Optional[str] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 22)

        back_btn = QPushButton("< Back to Transforms")
        back_btn.setStyleSheet(f"background: transparent; border: none; color: {theme.TEXT_SECONDARY}; text-align: left;")
        back_btn.clicked.connect(self.on_back)
        outer.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        header = QHBoxLayout()
        self.title_edit = QLineEdit()
        self.title_edit.setStyleSheet(
            f"font-size: 16pt; font-weight: 700; border: none; background: transparent; color: {theme.TEXT_PRIMARY};"
        )
        header.addWidget(self.title_edit, 1)
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(
            f"background-color: {theme.ACCENT_PRIMARY}; color: white; border: none; "
            f"border-radius: 8px; padding: 7px 16px; font-weight: 600;"
        )
        save_btn.clicked.connect(self._save)
        header.addWidget(save_btn)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setStyleSheet(f"background: transparent; border: none; color: {theme.DANGER};")
        self.delete_btn.clicked.connect(self._delete)
        header.addWidget(self.delete_btn)
        outer.addLayout(header)

        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("Short description shown on the card")
        self.description_edit.setStyleSheet(_FIELD_QSS)
        outer.addWidget(self.description_edit)

        columns = QHBoxLayout()
        outer.addLayout(columns, 1)

        left = QVBoxLayout()
        left.addWidget(_section_label("Choose a keyboard shortcut"))
        self.shortcut_recorder = KeySequenceRecorder("")
        self.shortcut_recorder.setStyleSheet(_FIELD_QSS)
        left.addWidget(self.shortcut_recorder)

        left.addWidget(_section_label("Test this Transform"))
        self.test_input = QPlainTextEdit(placeholderText="Type or paste raw dictation to preview...")
        self.test_output = QPlainTextEdit(readOnly=True, placeholderText="Transformed output appears here...")
        self.test_input.setStyleSheet(_FIELD_QSS)
        self.test_output.setStyleSheet(_FIELD_QSS)
        left.addWidget(self.test_input)
        left.addWidget(self.test_output)
        preview_btn = QPushButton("Preview")
        preview_btn.setStyleSheet(
            f"background-color: {theme.BG_CARD}; border: 1px solid {theme.BORDER_SUBTLE}; "
            f"border-radius: 8px; padding: 7px 14px; color: {theme.TEXT_PRIMARY};"
        )
        preview_btn.clicked.connect(self._preview)
        left.addWidget(preview_btn)
        columns.addLayout(left, 1)

        right = QVBoxLayout()
        right.addWidget(_section_label("Customize prompt"))
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setMinimumWidth(360)
        self.prompt_edit.setStyleSheet(_FIELD_QSS)
        right.addWidget(self.prompt_edit, 1)
        columns.addLayout(right, 1)

    def load(self, transform_id: str) -> None:
        t = self.engine.db.get_transform(transform_id)
        if not t:
            self.on_back()
            return
        self.transform_id = transform_id
        self.title_edit.setText(t["title"])
        self.description_edit.setText(t["description"])
        self.shortcut_recorder.hotkey_str = t["shortcut"]
        self.shortcut_recorder.setText(t["shortcut"].upper() if t["shortcut"] else "Click to set")
        self.prompt_edit.setPlainText(t["system_prompt"])
        self.test_input.clear()
        self.test_output.clear()
        self.delete_btn.setEnabled(not t["is_default"])

    def _save(self) -> None:
        if not self.transform_id:
            return
        self.engine.db.update_transform(
            self.transform_id,
            title=self.title_edit.text().strip() or "Untitled",
            description=self.description_edit.text().strip(),
            shortcut=self.shortcut_recorder.hotkey_str,
            system_prompt=self.prompt_edit.toPlainText(),
        )
        self.engine.refresh_transform_hotkeys()
        QMessageBox.information(self, "Wisperno", "Transform saved.")

    def _delete(self) -> None:
        if not self.transform_id:
            return
        if QMessageBox.question(self, "Delete Transform", "Delete this Transform permanently?") != QMessageBox.StandardButton.Yes:
            return
        self.engine.db.delete_transform(self.transform_id)
        self.engine.refresh_transform_hotkeys()
        self.on_back()

    def _preview(self) -> None:
        if not self.engine.transformer:
            QMessageBox.warning(self, "Wisperno", "The transform engine hasn't finished loading yet.")
            return
        text = self.test_input.toPlainText().strip()
        if not text:
            return
        self.test_output.setPlainText("Transforming...")
        result = self.engine.transformer.transform_with_prompt(text, self.prompt_edit.toPlainText(), label=self.transform_id or "preview")
        self.test_output.setPlainText(result)


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"font-weight: 600; color: {theme.TEXT_PRIMARY}; margin-top: 8px; background: transparent; border: none;")
    return lbl
