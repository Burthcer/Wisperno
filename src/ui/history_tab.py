"""
History & Activity Log tab for the Wisperno dashboard: metrics, search, a
segmented All/Dictations/Live Transcripts/Pinned filter bar, and cards
showing the polished transcript with a collapsible raw-transcript disclosure.
"""

import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QFrame, QMessageBox, QSizePolicy, QLayout,
)

from src.ui import theme

MODE_LABELS = {
    "polish": "Polish", "prompt_engineer": "Prompt Engineer", "bullets": "Bullets",
    "code": "Code & CLI", "raw": "Raw", "text_polish": "Text Polish", "live": "Live Session",
    "code_fix": "Code Fix", "grammar_correct": "Grammar Correct",
}

# Header badge height is a hard invariant (see HEADER_HEIGHT's comment on the
# accordion-jitter fix below) - every badge in the header row uses this exact
# height, never "however tall its text/padding happens to compute to".
BADGE_HEIGHT = 22
HEADER_HEIGHT = 32


def _card(layout_cls=QVBoxLayout):
    frame = QFrame()
    frame.setObjectName("card")
    layout = layout_cls(frame)
    return frame, layout


def _label_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


def _format_duration(seconds: float) -> str:
    """45.2s for anything under a minute, 1m 32s beyond that - matches how
    people actually read a dictation length instead of a raw seconds count."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(round(seconds)), 60)
    return f"{minutes}m {rem:02d}s"


def _badge(text: str) -> QLabel:
    """Metadata value (duration/words/latency/mode) - plain monospace text,
    no pill background, matching the flat-list mockup (assets/screenshots/
    new assets/) rather than the earlier boxed-pill look. Fixed height +
    Fixed vertical size policy - see the accordion-jitter fix note on
    _history_card()'s header_widget: without this, a QLabel defaults to a
    Preferred vertical size policy, meaning Qt's layout engine treats its
    height as negotiable, not fixed. Under normal conditions that negotiation
    settles back to the same natural height and nothing visibly changes -
    but the moment a SIBLING widget lower in the card changes visibility (the
    raw-transcript disclosure), Qt runs a fresh layout pass over the whole
    card, and a badge with a negotiable height can get stretched to fill
    whatever intermediate space that pass momentarily assigns it before
    settling - the reported 500ms vertical "stretch" glitch. Locking every
    badge to a hard pixel height removes the negotiation entirely: there's
    nothing left to stretch.
    """
    label = QLabel(text)
    label.setFixedHeight(BADGE_HEIGHT)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-family: {theme.FONT_FAMILY_MONO}; font-size: 8pt;")
    return label


# Color-coded so the source of an entry (voice dictation vs. polishing an
# existing selection vs. a named Transforms-hub run vs. a Live Transcription
# session) is legible at a glance, without reading the card's body text.
SOURCE_BADGE_COLORS = {
    "dictation": theme.ACCENT_PRIMARY,     # purple
    "text_polish": theme.ACCENT_SECONDARY,  # muted purple
    "transform": "#10B981",                 # emerald
    "live_transcript": "#10B981",           # emerald - matches the mission's "distinct emerald badge" spec
    "writing_style": theme.ACCENT_PRIMARY_ACTIVE,  # brighter purple - a Writing Styles (Alt+V) result
}


def _source_badge(text: str, color: str) -> QLabel:
    """Bold uppercase TEXT badge - no background pill, matching the mockup's
    clean typography-only source tag."""
    label = QLabel(text.upper())
    label.setFixedHeight(BADGE_HEIGHT)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 8pt; letter-spacing: 0.5px;")
    return label


def _link_btn(text: str, color: str = None) -> QPushButton:
    """Minimal text-only action link (Copy/Pin/Favorite/Delete) - no button
    chrome, matching the mockup's flat action row instead of bordered buttons."""
    btn = QPushButton(text)
    btn.setFlat(True)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    color = color or theme.TEXT_SECONDARY
    btn.setStyleSheet(
        f"QPushButton {{ background: transparent; border: none; color: {color}; font-size: 9pt; padding: 2px 0; }}"
        f"QPushButton:hover {{ color: {theme.TEXT_PRIMARY if color == theme.TEXT_SECONDARY else color}; text-decoration: underline; }}"
    )
    return btn


# Filter bar segments: (key, label). "pinned" filters is_pinned regardless of
# entry_type; "dictation"/"live_transcript" filter entry_type regardless of
# pin state - each is its own top-level view, not combinable, matching the
# mission's single-row segmented-toggle spec (only one active at a time).
FILTER_SEGMENTS = [
    ("all", "All"),
    ("dictation", "Dictations"),
    ("live_transcript", "Live Transcripts"),
    ("pinned", "Pinned"),
]


class HistoryTab(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._filter_mode = "all"
        self._filter_buttons = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.addWidget(_label_title("History"))
        subtitle = QLabel("Every dictation, transcribed and polished locally.")
        subtitle.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 10pt;")
        layout.addWidget(subtitle)

        self.metrics_row = QHBoxLayout()
        self.metrics_row.setSpacing(0)
        metrics_frame = QFrame()
        metrics_frame.setObjectName("card")
        metrics_frame.setLayout(self.metrics_row)
        layout.addWidget(metrics_frame)

        search_row = QHBoxLayout()
        self.search_box = QLineEdit(placeholderText="⌕  Search transcripts...")
        self.search_box.textChanged.connect(lambda _t: self.refresh())
        search_row.addWidget(self.search_box, 1)
        search_row.addStretch()

        for key, label in FILTER_SEGMENTS:
            btn = QPushButton(label)
            btn.setObjectName("filterPill")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, k=key: self._set_filter(k))
            self._filter_buttons[key] = btn
            search_row.addWidget(btn)
        layout.addLayout(search_row)
        self._update_filter_buttons()

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        # Prevent the accordion's expand/collapse from ever flexing anything
        # OUTSIDE the one card being toggled: SetMinimumSize means this layout
        # only ever grows to fit its current content, it never lets a
        # transient intermediate size (mid layout-pass) push other cards
        # around - the card list grows/shrinks by exactly one card's worth of
        # height when a disclosure opens, nothing more.
        self.list_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.list_layout.addStretch()
        self.scroll.setWidget(self.list_container)
        layout.addWidget(self.scroll)

        self.refresh()

    def reset_scroll(self) -> None:
        self.scroll.verticalScrollBar().setValue(0)

    def _set_filter(self, mode: str) -> None:
        self._filter_mode = mode
        self._update_filter_buttons()
        self.refresh()

    def _update_filter_buttons(self) -> None:
        for key, btn in self._filter_buttons.items():
            btn.setProperty("active", key == self._filter_mode)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _metric_column(self, value: str, label: str, divider: bool) -> QWidget:
        """One column of the unified metrics strip - a divider (not its own
        bordered card) separates it from the next column, matching the
        mockup's single-container-with-vertical-rules layout rather than 4
        separate boxed cards."""
        col = QWidget()
        if divider:
            col.setStyleSheet(f"border-left: 1px solid {theme.BORDER_SUBTLE};")
        v = QVBoxLayout(col)
        v.setContentsMargins(20 if divider else 4, 16, 4, 16)
        val = QLabel(value)
        val.setStyleSheet(f"font-family: {theme.FONT_FAMILY_MONO}; font-size: 20pt; font-weight: 700; color: {theme.TEXT_PRIMARY};")
        lab = QLabel(label)
        lab.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 8pt;")
        v.addWidget(val)
        v.addWidget(lab)
        return col

    def _rebuild_metrics(self) -> None:
        while self.metrics_row.count():
            item = self.metrics_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        stats = self.engine.db.get_stats()
        columns = [
            (str(stats["total_words"]), "Total Words Dictated"),
            (f'{stats["hours_saved"]:.1f}h', "Hours Saved"),
            (f'{stats["avg_wpm"]:.0f}', "Avg WPM"),
            (str(stats["total_dictations"]), "Dictations"),
        ]
        for i, (value, label) in enumerate(columns):
            self.metrics_row.addWidget(self._metric_column(value, label, divider=i > 0), 1)

    def refresh(self) -> None:
        self._rebuild_metrics()
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        pinned_only = self._filter_mode == "pinned"
        entry_type = self._filter_mode if self._filter_mode in ("dictation", "live_transcript") else None
        rows = self.engine.db.list_history(
            search=self.search_box.text().strip() or None, pinned_only=pinned_only, entry_type=entry_type,
        )
        for row in rows:
            self.list_layout.insertWidget(self.list_layout.count() - 1, self._history_card(row))

    def _source_badge_for_row(self, row: dict) -> QLabel:
        if row.get("entry_type") == "live_transcript":
            return _source_badge("Live Session", SOURCE_BADGE_COLORS["live_transcript"])
        source = row["source"]
        if source == "text_polish":
            return _source_badge("Selection Polish", SOURCE_BADGE_COLORS["text_polish"])
        if source == "writing_style":
            style_title = row["mode_used"].removeprefix("style_").replace("_", " ").title()
            return _source_badge(f"Style: {style_title}", SOURCE_BADGE_COLORS["writing_style"])
        if source == "transform":
            transform = self.engine.db.get_transform(row["mode_used"])
            name = transform["title"] if transform else row["mode_used"]
            return _source_badge(f"Transform: {name}", SOURCE_BADGE_COLORS["transform"])
        return _source_badge("Dictation", SOURCE_BADGE_COLORS["dictation"])

    def _build_header(self, row: dict, ts: str) -> QWidget:
        """The metadata row (timestamp/badges/duration/words/latency) as its
        own fixed-height QWidget, not a bare layout added straight to the
        card - THIS is the actual accordion-jitter fix (see _badge()'s
        docstring for the full mechanism). A bare QHBoxLayout has no size of
        its own to fix; wrapping it in a QWidget with setFixedHeight() gives
        Qt's layout engine a hard constraint it cannot renegotiate away
        during the card's re-layout when the raw-transcript disclosure below
        toggles visibility."""
        header_widget = QWidget()
        header_widget.setFixedHeight(HEADER_HEIGHT)
        header_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header = QHBoxLayout(header_widget)
        header.setContentsMargins(0, 0, 0, 0)

        if row["is_pinned"]:
            pin_label = QLabel("📌")
            pin_label.setFixedHeight(BADGE_HEIGHT)
            header.addWidget(pin_label)
        ts_label = QLabel(f"<b>{ts}</b>")
        ts_label.setFixedHeight(BADGE_HEIGHT)
        header.addWidget(ts_label)
        header.addWidget(self._source_badge_for_row(row))
        if row.get("entry_type") == "live_transcript":
            # A live session's "duration" is the whole session length, not a
            # single utterance - shown alongside word count like every other
            # card, just sourced from the session total rather than one clip.
            header.addWidget(_badge(_format_duration(row["duration_seconds"])))
            header.addWidget(_badge(f'{row["word_count"]} words'))
        elif row["source"] in ("text_polish", "writing_style"):
            # Neither a "Polish Selected Text" nor a Writing Styles entry has
            # audio - duration_seconds is a meaningless 0.0 placeholder for
            # both (see workers.py's SelectionPolishWorker/WritingStylesWorker,
            # which always insert duration_seconds=0.0). Show what actually
            # happened instead of a misleading "0.0s".
            header.addWidget(_badge("Selection"))
            header.addWidget(_badge(f'{row["word_count"]} words'))
            header.addWidget(_badge(f'{row["latency_ms"]:.0f}ms'))
        else:
            header.addWidget(_badge(_format_duration(row["duration_seconds"])))
            header.addWidget(_badge(f'{row["word_count"]} words'))
            header.addWidget(_badge(f'{row["latency_ms"]:.0f}ms'))
        # A writing-style entry's source badge already reads "Style: Professional" -
        # a second badge repeating the raw mode_used ("style_professional") would
        # just be redundant noise next to it.
        if row.get("entry_type") != "live_transcript" and row["source"] != "writing_style":
            header.addWidget(_badge(MODE_LABELS.get(row["mode_used"], row["mode_used"])))
        header.addStretch()
        return header_widget

    def _history_card(self, row: dict) -> QWidget:
        # A flat list row with a bottom divider, not a separately-bordered/
        # rounded card - matches the mockup's continuous-list feel rather
        # than a card grid. Still a QWidget (not a bare layout) so its own
        # background/border are real, paintable properties.
        card = QWidget()
        card.setStyleSheet(f"background-color: #0D0D14; border-bottom: 1px solid {theme.BORDER_SUBTLE};")
        v = QVBoxLayout(card)
        v.setContentsMargins(4, 14, 4, 14)
        v.setSpacing(8)
        try:
            ts = datetime.datetime.fromisoformat(row["timestamp"]).strftime("%b %d, %I:%M %p")
        except Exception:
            ts = row["timestamp"]

        v.addWidget(self._build_header(row, ts))

        polished_label = QLabel(row["polished_transcript"])
        polished_label.setWordWrap(True)
        polished_label.setStyleSheet(f"color: #E2E8F0; font-size: 10pt; line-height: 1.6;")
        v.addWidget(polished_label)

        raw_toggle = QPushButton("▶ Show Raw Transcript")
        raw_toggle.setFlat(True)
        raw_toggle.setStyleSheet(f"text-align: left; border: none; color: {theme.TEXT_SECONDARY}; padding: 2px 0; background: transparent;")
        raw_label = QLabel(f'<span style="color:{theme.TEXT_SECONDARY}">{row["raw_transcript"]}</span>')
        raw_label.setWordWrap(True)
        raw_label.hide()

        def _toggle_raw(checked=False, btn=raw_toggle, lbl=raw_label):
            expanded = lbl.isVisible()
            lbl.setVisible(not expanded)
            btn.setText("▼ Hide Raw Transcript" if not expanded else "▶ Show Raw Transcript")

        raw_toggle.clicked.connect(_toggle_raw)
        v.addWidget(raw_toggle)
        v.addWidget(raw_label)

        actions = QHBoxLayout()
        actions.setSpacing(18)
        copy_btn = _link_btn("Copy")
        copy_btn.clicked.connect(lambda: self._copy(row["polished_transcript"], copy_btn))
        pin_btn = _link_btn("Unpin" if row["is_pinned"] else "Pin")
        pin_btn.clicked.connect(lambda: self._toggle_pin(row["id"]))
        fav_btn = _link_btn("Unfavorite" if row["is_favorite"] else "Favorite")
        fav_btn.clicked.connect(lambda: self._toggle_favorite(row["id"]))
        dict_btn = _link_btn("+ Dictionary")
        dict_btn.clicked.connect(lambda: self._add_to_dictionary(row["raw_transcript"]))
        del_btn = _link_btn("Delete", color=theme.DANGER)
        del_btn.clicked.connect(lambda: self._delete(row["id"]))
        for b in (copy_btn, pin_btn, fav_btn, dict_btn, del_btn):
            actions.addWidget(b)
        actions.addStretch()
        v.addLayout(actions)

        return card

    def _copy(self, text: str, button: QPushButton) -> None:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(text)
        original = button.text()
        button.setText("Copied!")
        QTimer.singleShot(1200, lambda: button.setText(original))

    def _toggle_favorite(self, history_id: int) -> None:
        self.engine.db.toggle_favorite(history_id)
        self.refresh()

    def _toggle_pin(self, history_id: int) -> None:
        self.engine.db.toggle_pin(history_id)
        self.refresh()

    def _delete(self, history_id: int) -> None:
        self.engine.db.delete_history(history_id)
        self.refresh()

    def _add_to_dictionary(self, raw_text: str) -> None:
        word = raw_text.strip().split(" ")[0] if raw_text.strip() else ""
        self.engine.vocabulary.add_entry(word, word, category="Name")
        QMessageBox.information(self, "Wisperno", f"Added '{word}' to the dictionary. Edit it in the Dictionary tab.")
