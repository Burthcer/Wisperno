"""
Floating Live Transcription overlay: a frameless, always-on-top window
showing a real-time autoscrolling transcript plus session controls. Distinct
from src/ui/floating_pill.py (the always-present dictation-status pill) -
this window only exists while a Live Transcription session is active. Built
from standard Qt widgets (QTextEdit, QPushButton) rather than a hand-rolled
paintEvent like the pill, since a scrolling multi-line transcript needs a
real text widget, not a shape to paint.
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget

from src.ui import theme

QSS = f"""
QWidget#liveRoot {{
    background-color: {theme.PILL_BG};
    border: 1px solid {theme.PILL_BORDER};
    border-radius: 12px;
}}
QLabel#liveTitle {{ color: {theme.PILL_TEXT_PRIMARY}; font-weight: 600; font-size: 10pt; }}
QLabel#liveTimer {{ color: {theme.TEXT_SECONDARY}; font-size: 9pt; }}
QTextEdit#liveTranscript {{
    background-color: #17171F;
    border: 1px solid {theme.PILL_BORDER};
    border-radius: 8px;
    color: {theme.TEXT_PRIMARY};
    padding: 8px;
    font-size: 10pt;
}}
QPushButton {{
    background-color: {theme.BG_CARD};
    border: 1px solid {theme.BORDER_SUBTLE};
    border-radius: 7px;
    padding: 6px 14px;
    color: {theme.TEXT_PRIMARY};
}}
QPushButton:hover {{ background-color: {theme.BG_CARD_HOVER}; }}
QPushButton#stopSave {{ background-color: {theme.ACCENT_PRIMARY}; border: none; color: white; font-weight: 600; }}
QPushButton#stopSave:hover {{ background-color: {theme.ACCENT_PRIMARY_HOVER}; }}
QPushButton#discardBtn {{ color: {theme.DANGER}; }}
"""


class LiveTranscriptionWindow(QWidget):
    """Purely a view + a set of button signals - all session logic lives in
    src/live_transcriber.py:LiveTranscriptionWorker and src/engine.py, which
    connect to these signals and call back into append_chunk()/show_cleaning_up()/
    close_session_ui()."""

    pause_clicked = Signal()
    resume_clicked = Signal()
    stop_save_clicked = Signal()
    discard_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("liveRoot")
        self.setStyleSheet(QSS)
        self.resize(420, 320)
        self._paused = False
        self._elapsed_sec = 0
        self._blink_on = True
        self._confirmed_plain = ""   # permanent, appended text - never rewritten
        self._speculative_plain = ""  # current unconfirmed tail - REPLACED each hop, never appended

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        self.dot = QLabel("●")
        self.dot.setStyleSheet(f"color: {theme.ACCENT_RECORDING}; font-size: 12pt;")
        header.addWidget(self.dot)
        self.title_label = QLabel("Live Transcribing")
        self.title_label.setObjectName("liveTitle")
        header.addWidget(self.title_label)
        header.addStretch()
        self.timer_label = QLabel("00:00")
        self.timer_label.setObjectName("liveTimer")
        header.addWidget(self.timer_label)
        root.addLayout(header)

        self.transcript_view = QTextEdit()
        self.transcript_view.setObjectName("liveTranscript")
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self.transcript_view, 1)

        actions = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        self.stop_btn = QPushButton("Stop && Save")
        self.stop_btn.setObjectName("stopSave")
        self.stop_btn.clicked.connect(self.stop_save_clicked)
        self.discard_btn = QPushButton("Discard")
        self.discard_btn.setObjectName("discardBtn")
        self.discard_btn.clicked.connect(self.discard_clicked)
        for b in (self.pause_btn, self.stop_btn, self.discard_btn):
            actions.addWidget(b)
        root.addLayout(actions)

        # A simple alpha-swap blink + a 1s clock tick - matches floating_pill.py's
        # own lightweight QTimer approach rather than pulling in QPropertyAnimation
        # for a two-state pulse.
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink)
        self._blink_timer.start(600)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)

    def _on_pause_clicked(self) -> None:
        self._paused = not self._paused
        self.pause_btn.setText("Resume" if self._paused else "Pause")
        (self.resume_clicked if self._paused else self.pause_clicked).emit()

    def _blink(self) -> None:
        if self._paused:
            self.dot.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 12pt;")
            return
        self._blink_on = not self._blink_on
        color = theme.ACCENT_RECORDING if self._blink_on else theme.ACCENT_RECORDING_DIM
        self.dot.setStyleSheet(f"color: {color}; font-size: 12pt;")

    def _tick_clock(self) -> None:
        self._elapsed_sec += 1
        m, s = divmod(self._elapsed_sec, 60)
        self.timer_label.setText(f"{m:02d}:{s:02d}")

    def start_session_ui(self) -> None:
        self.transcript_view.clear()
        self._confirmed_plain = ""
        self._speculative_plain = ""
        self._elapsed_sec = 0
        self.timer_label.setText("00:00")
        self._paused = False
        self.pause_btn.setText("Pause")
        self.title_label.setText("Live Transcribing")
        self._clock_timer.start(1000)
        screen = self.screen()
        if screen is not None:
            geo = screen.geometry()
            self.move(geo.right() - self.width() - 24, geo.bottom() - self.height() - 100)
        self.show()
        self.raise_()

    def append_chunk(self, text: str) -> None:
        """A CONFIRMED chunk (LiveTranscriptionWorker.text_chunk_received) -
        permanently appended, joined by a single space unless the running
        text already ends one (a bare `append()` call would instead start a
        new paragraph/line per chunk, reading as one line per chunk rather
        than a flowing transcript)."""
        if self._confirmed_plain and not self._confirmed_plain.endswith((" ", "\n")):
            self._confirmed_plain += " "
        self._confirmed_plain += text
        self._render()

    def set_speculative_text(self, text: str) -> None:
        """The current unconfirmed tail (LiveTranscriptionWorker.speculative_text_changed)
        - REPLACES whatever speculative text was showing, never appended to
        it (each hop re-transcribes and re-guesses the same still-unconfirmed
        audio, so the previous guess is simply wrong now, not a prior word
        that also happened)."""
        self._speculative_plain = text
        self._render()

    def _render(self) -> None:
        """Re-renders the whole transcript view from confirmed + speculative
        state on every change - simpler and safer than tracking cursor
        positions to patch just the tail, and cheap enough at a 500-800ms hop
        rate for any reasonably-sized transcript."""
        import html

        confirmed_html = html.escape(self._confirmed_plain)
        speculative_html = (
            f' <span style="color:{theme.TEXT_SECONDARY}; font-style:italic;">{html.escape(self._speculative_plain)}</span>'
            if self._speculative_plain else ""
        )
        self.transcript_view.setHtml(f'<div style="color:{theme.TEXT_PRIMARY};">{confirmed_html}{speculative_html}</div>')
        scrollbar = self.transcript_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def show_cleaning_up(self) -> None:
        self.title_label.setText("Cleaning up...")
        self._clock_timer.stop()

    def close_session_ui(self) -> None:
        self._clock_timer.stop()
        self.hide()
