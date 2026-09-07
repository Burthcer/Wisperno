"""
Qt Floating Pill Widget for Wisperno.

Uses true per-pixel alpha compositing (WA_TranslucentBackground + a custom
paintEvent drawing an anti-aliased rounded capsule) instead of a tkinter Canvas,
which can only fake rounded corners with arcs/rectangles on an opaque background
and shows visible square-corner artifacts. Frameless, always-on-top, and
excluded from the taskbar/alt-tab via Qt.WindowType.Tool.
"""

import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QRectF, QPropertyAnimation, QEasingCurve, Property, Signal
from PySide6.QtGui import QPainter, QColor, QFont, QFontMetrics, QPainterPath, QGuiApplication, QCursor, QLinearGradient
from PySide6.QtWidgets import QWidget, QMenu

from src.ui import theme

STATE_LOADING = "loading"
STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_PROCESSING = "processing"
STATE_TYPING = "typing"
STATE_SUCCESS = "success"
STATE_MODE_CHANGED = "mode_changed"
STATE_ERROR = "error"
STATE_SELECTION_PROCESSING = "selection_processing"
STATE_SELECTION_SUCCESS = "selection_success"
STATE_FALLBACK_WARNING = "fallback_warning"
STATE_CPU_MODE = "cpu_mode"

DORMANT_W = 220
RECORDING_W = 320
PROCESSING_W = 260
TYPING_W = 200
SUCCESS_W = 260
ERROR_W = 300
SELECTION_PROCESSING_W = 300
FALLBACK_WARNING_W = 320
CPU_MODE_W = 360
PILL_H = 34  # compact profile: half-height capsule radius (17px), 8px/4px h/v padding
BOTTOM_OFFSET = 110

# Uniform horizontal margin every painted state lines its content up against -
# content is centered inside the capsule by starting/ending at this same
# distance from both edges, rather than each state carrying its own
# hand-picked (and previously inconsistent - 14/16/18px) inset.
CONTENT_MARGIN = 16


class FloatingPill(QWidget):
    """Always-visible dormant "Ready" pill that expands for recording/processing/success."""

    dashboard_requested = Signal()
    cycle_mode_requested = Signal()
    hide_pill_requested = Signal()
    close_requested = Signal()  # left to the controller: hide-to-tray vs full quit, per config
    quit_requested = Signal()

    def __init__(self, initial_mode: str = "POLISH", hotkey_label: str = "Ctrl+Alt"):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        # Hard ceiling independent of any width-computation code path: the pill
        # must never be able to stretch toward screen width.
        self.setMaximumWidth(420)
        self.setFixedHeight(PILL_H)
        self.setWindowTitle("Wisperno Overlay")  # matched by src.single_instance to focus a running instance
        self.setWindowIcon(theme.get_app_icon())
        self.setMouseTracking(True)  # needed for hover-to-reveal on the close button

        self._close_btn_rect = QRectF()
        self._hovering_close = False

        self.current_mode = initial_mode.upper()
        self.hotkey_label = hotkey_label
        self.state = STATE_LOADING
        # Set by _set_state() when a "processing" state arrives carrying a
        # "processing:<Label>" suffix (see set_state's docstring) - lets a
        # Transforms-hub-driven dictation show its own name ("Prompt
        # Engineering...") instead of the generic "Polishing speech..." text.
        self._status_label = ""
        self._smoothed_level = 0.0
        self._levels = [0.0, 0.0, 0.0, 0.0]
        self._recording_start = 0.0
        self._pulse_phase = 0.0
        self._progress_phase = 0.0
        self._width_current = float(DORMANT_W)
        self._width_target = float(DORMANT_W)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)  # ~60fps
        self._anim_timer.timeout.connect(self._tick)

        self._collapse_timer = QTimer(self)
        self._collapse_timer.setSingleShot(True)
        self._collapse_timer.timeout.connect(lambda: self._set_state(STATE_IDLE))

        self.resize(DORMANT_W, PILL_H)
        self._reposition()
        self.show()
        self._anim_timer.start()

    # --- Geometry ------------------------------------------------------------

    def _reposition(self) -> None:
        screen = QGuiApplication.primaryScreen().geometry()
        x = screen.x() + (screen.width() - self.width()) // 2
        y = screen.y() + screen.height() - BOTTOM_OFFSET
        self.move(x, y)

    def _tick(self) -> None:
        # Smoothly interpolate width toward target (drives the expand/collapse feel).
        if abs(self._width_current - self._width_target) > 0.5:
            self._width_current += (self._width_target - self._width_current) * 0.35
            self.resize(int(self._width_current), PILL_H)
            self._reposition()

        if self.state in (STATE_RECORDING, STATE_TYPING):
            self._pulse_phase += 0.12
        elif self.state in (STATE_PROCESSING, STATE_SELECTION_PROCESSING):
            self._progress_phase = (self._progress_phase + 0.02) % 1.0

        self.update()

    # --- Public thread-safe-ish API (Qt slots; call via signals from workers) --

    def set_state(self, state: str) -> None:
        """`state` is normally a bare state name (STATE_PROCESSING etc.). A
        "processing:<Label>" form additionally carries a display name for a
        Transforms-hub-driven dictation (e.g. "processing:Prompt Engineering") -
        painted in place of the generic "Polishing speech..." text. A
        "recording:<Label>" form likewise overrides the mode badge (e.g.
        "recording:DIRECT" when this dictation will skip the LLM entirely) -
        see _paint_mode_badge()."""
        self._set_state(state)

    def _set_state(self, state: str) -> None:
        self._collapse_timer.stop()
        if ":" in state:
            state, self._status_label = state.split(":", 1)
        else:
            self._status_label = ""
        self.state = state

        if state == STATE_IDLE:
            self._width_target = self._idle_width()
        elif state == STATE_LOADING:
            self._width_target = DORMANT_W
        elif state == STATE_RECORDING:
            self._recording_start = time.time()
            self._levels = [0.0, 0.0, 0.0, 0.0]
            self._width_target = RECORDING_W
        elif state == STATE_PROCESSING:
            self._width_target = PROCESSING_W
        elif state == STATE_TYPING:
            self._width_target = TYPING_W
        elif state == STATE_SUCCESS:
            self._width_target = SUCCESS_W
            self._collapse_timer.start(700)
        elif state == STATE_SELECTION_PROCESSING:
            self._width_target = SELECTION_PROCESSING_W
        elif state == STATE_SELECTION_SUCCESS:
            self._width_target = SUCCESS_W
            self._collapse_timer.start(700)
        elif state == STATE_MODE_CHANGED:
            self._width_target = 240
            self._collapse_timer.start(1000)
        elif state == STATE_ERROR:
            self._width_target = ERROR_W
            # No auto-collapse: an init failure stays visible until restarted/fixed,
            # rather than quietly reverting to a "Ready" label that would be a lie.
        elif state == STATE_FALLBACK_WARNING:
            self._width_target = FALLBACK_WARNING_W
            self._collapse_timer.start(2500)
        elif state == STATE_CPU_MODE:
            self._width_target = CPU_MODE_W
            self._collapse_timer.start(4000)  # one-time startup info, longer than a routine warning

        self.update()

    def set_mode(self, mode: str) -> None:
        self.current_mode = mode.upper()
        self._set_state(STATE_MODE_CHANGED)

    def set_level(self, rms: float) -> None:
        if self.state != STATE_RECORDING:
            return
        self._smoothed_level = 0.6 * self._smoothed_level + 0.4 * min(rms * 12.0, 1.0)
        self._levels = self._levels[1:] + [self._smoothed_level]

    def set_hotkey_label(self, label: str) -> None:
        self.hotkey_label = label
        if self.state == STATE_IDLE:
            self._width_target = self._idle_width()
        self.update()

    def _idle_width(self) -> int:
        """Size the dormant pill to fit its label - long rebound shortcuts (Ctrl+Alt+Shift+V) must not clip.
        Upper-bounded as well as lower-bounded: no code path (this one included) should ever be able
        to stretch the pill toward screen width, whatever the hotkey label ends up being."""
        label = f"✦ Wisperno | Ready [{self.hotkey_label}]"
        metrics = QFontMetrics(QFont(theme.FONT_FAMILY, 9))
        text_width = metrics.horizontalAdvance(label)
        return max(240, min(text_width + 60, 420))

    # --- Painting --------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w, h = self.width(), self.height()
        radius = h / 2.0

        bg, border = self._colors_for_state()
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), radius, radius)

        painter.fillPath(path, QColor(bg))
        # Apple-style specular sheen: a subtle translucent white gradient
        # clipped to the same pill shape, brightest at the top edge and
        # fading out by ~18% of the height - independent of state color.
        sheen = QLinearGradient(0, 0, 0, h)
        sheen.setColorAt(0.0, QColor(255, 255, 255, 38))
        sheen.setColorAt(0.18, QColor(255, 255, 255, 0))
        painter.fillPath(path, sheen)
        painter.setPen(QColor(border))
        painter.drawPath(path)

        self._paint_content(painter, w, h)
        painter.end()

    def _colors_for_state(self):
        if self.state == STATE_RECORDING:
            return "#1E1216", theme.ACCENT_RECORDING
        if self.state == STATE_PROCESSING:
            return "#15121F", theme.ACCENT_PROCESSING
        if self.state == STATE_TYPING:
            return "#101828", theme.ACCENT_PROCESSING
        if self.state == STATE_SUCCESS:
            return "#101C15", theme.ACCENT_SUCCESS
        if self.state == STATE_MODE_CHANGED:
            return "#15121F", theme.ACCENT_PROCESSING
        if self.state == STATE_ERROR:
            return "#1F160C", theme.ACCENT_ERROR
        if self.state == STATE_FALLBACK_WARNING:
            return "#1F160C", theme.ACCENT_ERROR
        if self.state == STATE_CPU_MODE:
            return "#1F160C", theme.ACCENT_ERROR
        if self.state == STATE_SELECTION_PROCESSING:
            return "#15121F", theme.ACCENT_PROCESSING
        if self.state == STATE_SELECTION_SUCCESS:
            return "#101C15", theme.ACCENT_SUCCESS
        return theme.PILL_BG, theme.PILL_BORDER

    def _draw_balanced_text(self, painter: QPainter, rect: QRectF, text: str) -> None:
        """Center `text` in `rect` if it comfortably fits (a short status like
        "Polishing text..." reads as deliberate, not accidentally left-stuck),
        falling back to left-alignment once it gets close to the pill's own
        boundaries (where centering would start clipping instead of helping)."""
        text_w = painter.fontMetrics().horizontalAdvance(text)
        align = Qt.AlignmentFlag.AlignCenter if text_w <= rect.width() * 0.9 else Qt.AlignmentFlag.AlignLeft
        painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter | align, text)

    def _paint_content(self, painter: QPainter, w: int, h: int) -> None:
        cy = h / 2.0
        m = CONTENT_MARGIN
        font = QFont(theme.FONT_FAMILY, 9)
        painter.setFont(font)

        if self.state in (STATE_IDLE, STATE_LOADING):
            painter.setPen(QColor(theme.ACCENT_PROCESSING if self.state == STATE_LOADING else theme.PILL_TEXT))
            label = "✦ Wisperno | Starting..." if self.state == STATE_LOADING else f"✦ Wisperno | Ready [{self.hotkey_label}]"
            text_w = w - 2 * m - (22 if self.state == STATE_IDLE else 0)  # leave room for the close glyph
            painter.drawText(QRectF(m, 0, text_w, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
            if self.state == STATE_IDLE:
                self._paint_close_button(painter, w, h)
            else:
                self._close_btn_rect = QRectF()
            return

        if self.state == STATE_RECORDING:
            self._paint_dot(painter, m, cy, theme.ACCENT_RECORDING)
            painter.setPen(QColor(theme.PILL_TEXT_PRIMARY))
            bold = QFont(theme.FONT_FAMILY, 9, QFont.Weight.DemiBold)
            painter.setFont(bold)
            painter.drawText(QRectF(m + 16, 0, 110, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "Listening...")
            self._paint_bars(painter, m + 130, cy)
            # Mode badge is measured first (it can run long - "PROMPT_ENGINEER"
            # is not "RAW") so the elapsed timer to its left never overlaps or
            # gets squeezed into it, whatever the active mode's name length.
            badge_w = self._paint_mode_badge(painter, w, cy, m)
            painter.setFont(font)
            painter.setPen(QColor(theme.TEXT_SECONDARY))
            elapsed = int(time.time() - self._recording_start)
            timer_text = f"{elapsed // 60}:{elapsed % 60:02d}"
            timer_right = w - m - badge_w - 10
            painter.drawText(QRectF(timer_right - 40, 0, 40, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, timer_text)
            return

        if self.state == STATE_PROCESSING:
            pulse = 0.5 + 0.5 * abs(((self._pulse_phase % 2.0) - 1.0))
            color = QColor(theme.ACCENT_PROCESSING)
            color.setAlphaF(0.5 + 0.5 * pulse)
            painter.setPen(color)
            bold = QFont(theme.FONT_FAMILY, 9, QFont.Weight.DemiBold)
            painter.setFont(bold)
            # A Transforms-hub-driven run carries its own name ("Prompt
            # Engineering...", "Fixing Code...") via the "processing:<Label>"
            # state suffix (see set_state) - falls back to the generic label
            # for the plain active-mode dictation path, which has no single name.
            label = f"{self._status_label}..." if self._status_label else "⚡ Polishing speech..."
            # Full m-to-(w-m) span, matching every other single-line state
            # (ERROR/FALLBACK_WARNING/CPU_MODE etc.) - this used to reserve an
            # extra 60px on the right for no reason (leftover from a copied
            # RECORDING-state layout, which needs that space for its timer/
            # badge; PROCESSING has neither), leaving the text and the
            # progress bar both visibly squeezed left of center.
            self._draw_balanced_text(painter, QRectF(m, 0, w - 2 * m, h), label)
            self._paint_progress_sweep(painter, w, cy, m)
            return

        if self.state == STATE_TYPING:
            self._paint_dot(painter, m, cy, theme.ACCENT_PROCESSING)
            bold = QFont(theme.FONT_FAMILY, 9, QFont.Weight.DemiBold)
            painter.setFont(bold)
            pulse = 0.5 + 0.5 * abs(((self._pulse_phase % 2.0) - 1.0))
            color = QColor(theme.ACCENT_PROCESSING)
            color.setAlphaF(0.6 + 0.4 * pulse)
            painter.setPen(color)
            # Direct Dictation (the default, zero-LLM path) - "Pasting..." names
            # the actual last step (Win32 paste injection), not implementation
            # detail ("Typing...") the user has no reason to care about.
            painter.drawText(QRectF(m + 16, 0, w - 2 * m - 16, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "Pasting...")
            return

        if self.state == STATE_SUCCESS:
            painter.setPen(QColor(theme.ACCENT_SUCCESS))
            bold = QFont(theme.FONT_FAMILY, 10, QFont.Weight.Bold)
            painter.setFont(bold)
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter, "✓ Injected!")
            return

        if self.state == STATE_SELECTION_PROCESSING:
            painter.setPen(QColor(theme.ACCENT_PROCESSING))
            bold = QFont(theme.FONT_FAMILY, 9, QFont.Weight.DemiBold)
            painter.setFont(bold)
            self._draw_balanced_text(painter, QRectF(m, 0, w - 2 * m, h), "✦ Polishing Selection...")
            self._paint_progress_sweep(painter, w, cy, m)
            return

        if self.state == STATE_SELECTION_SUCCESS:
            painter.setPen(QColor(theme.ACCENT_SUCCESS))
            bold = QFont(theme.FONT_FAMILY, 10, QFont.Weight.Bold)
            painter.setFont(bold)
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter, "✓ Replaced!")
            return

        if self.state == STATE_MODE_CHANGED:
            painter.setPen(QColor(theme.PILL_TEXT_PRIMARY))
            bold = QFont(theme.FONT_FAMILY, 10, QFont.Weight.DemiBold)
            painter.setFont(bold)
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter, f"✦ Mode: {self.current_mode}")
            return

        if self.state == STATE_ERROR:
            painter.setPen(QColor(theme.ACCENT_ERROR))
            bold = QFont(theme.FONT_FAMILY, 9, QFont.Weight.DemiBold)
            painter.setFont(bold)
            self._draw_balanced_text(painter, QRectF(m, 0, w - 2 * m, h), "⚠ Wisperno | Startup is stuck - see logs/wisperno.log")
            return

        if self.state == STATE_FALLBACK_WARNING:
            painter.setPen(QColor(theme.ACCENT_ERROR))
            bold = QFont(theme.FONT_FAMILY, 9, QFont.Weight.DemiBold)
            painter.setFont(bold)
            self._draw_balanced_text(painter, QRectF(m, 0, w - 2 * m, h), "⚠ Formatting fallback applied")
            return

        if self.state == STATE_CPU_MODE:
            painter.setPen(QColor(theme.ACCENT_ERROR))
            bold = QFont(theme.FONT_FAMILY, 9, QFont.Weight.DemiBold)
            painter.setFont(bold)
            self._draw_balanced_text(painter, QRectF(m, 0, w - 2 * m, h), "⚠ Running in CPU Compatibility Mode (No dedicated GPU detected)")
            return

    def _paint_close_button(self, painter: QPainter, w: int, h: int) -> None:
        """Small '×' at the right edge, low-alpha by default, full opacity on hover."""
        size = 18.0
        rect = QRectF(w - size - CONTENT_MARGIN, (h - size) / 2.0, size, size)
        self._close_btn_rect = rect

        alpha = 0.9 if self._hovering_close else 0.4
        if self._hovering_close:
            painter.setBrush(QColor(255, 255, 255, 30))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rect)

        pen_color = QColor(255, 255, 255)
        pen_color.setAlphaF(alpha)
        painter.setPen(pen_color)
        bold = QFont(theme.FONT_FAMILY, 10, QFont.Weight.DemiBold)
        painter.setFont(bold)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "×")

    def _paint_dot(self, painter: QPainter, x: float, cy: float, base_color: str) -> None:
        pulse = 0.5 + 0.5 * abs(((self._pulse_phase % 2.0) - 1.0))
        color = QColor(base_color)
        color.setAlphaF(0.55 + 0.45 * pulse)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        r = 5.0
        painter.drawEllipse(QRectF(x, cy - r, r * 2, r * 2))

    def _paint_bars(self, painter: QPainter, x0: float, cy: float) -> None:
        # Purple bars alongside the red recording dot - deliberately NOT the
        # same red (mockup: assets/screenshots/new assets/) - the dot signals
        # "recording", the waveform is the app's own purple accent, same
        # visual grammar as the rest of the redesigned UI.
        painter.setPen(Qt.PenStyle.NoPen)
        max_h = 14.0
        for i, level in enumerate(self._levels):
            bar_h = max(3.0, max_h * level)
            color = QColor(theme.ACCENT_PROCESSING if level > 0.15 else theme.ACCENT_PROCESSING_DIM)
            painter.setBrush(color)
            bx = x0 + i * 8
            painter.drawRoundedRect(QRectF(bx, cy - bar_h / 2, 3, bar_h), 1.5, 1.5)

    def _paint_progress_sweep(self, painter: QPainter, w: int, cy: float, m: int = CONTENT_MARGIN) -> None:
        # Full m-to-(w-m) span - both callers (PROCESSING, SELECTION_PROCESSING)
        # have no right-side badge/timer to leave room for, unlike RECORDING.
        track_left, track_right = m, w - m
        track_w = max(0, track_right - track_left)
        painter.setBrush(QColor(theme.ACCENT_PROCESSING_DIM))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(track_left, cy + 9, track_w, 3), 1.5, 1.5)

        sweep_w = track_w * 0.28
        x = track_left + (track_w - sweep_w) * (0.5 + 0.5 * ((self._progress_phase * 2) % 2.0 - 1.0))
        painter.setBrush(QColor(theme.ACCENT_PROCESSING))
        painter.drawRoundedRect(QRectF(x, cy + 9, sweep_w, 3), 1.5, 1.5)

    def _paint_mode_badge(self, painter: QPainter, w: int, cy: float, m: int = CONTENT_MARGIN) -> int:
        """Right-aligned "[MODE]" tag, sized to its own text rather than a fixed
        56px box - a long mode name (e.g. "PROMPT_ENGINEER") was getting
        clipped/squished against that old fixed width. Returns the measured
        width so callers (e.g. RECORDING's elapsed timer) can lay out to its left
        without overlapping, whatever the mode name's length."""
        badge_font = QFont(theme.FONT_FAMILY, 8, QFont.Weight.DemiBold)
        # A "recording:<Label>" state (see set_state's docstring) overrides
        # the badge for this specific recording - e.g. "DIRECT" when this
        # dictation will skip the LLM entirely, regardless of which
        # active_mode preset ("polish" etc.) happens to be selected.
        mode_label = self._status_label if self._status_label else self.current_mode
        text = f"[{mode_label}]"
        text_w = QFontMetrics(badge_font).horizontalAdvance(text)
        painter.setPen(QColor(theme.TEXT_SECONDARY))
        painter.setFont(badge_font)
        painter.drawText(QRectF(w - m - text_w, 0, text_w, self.height()),
                          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, text)
        return text_w

    # --- Mouse interaction -----------------------------------------------------
    # A single left-click (mousePressEvent, below) opens the dashboard - no
    # double-click requirement, so there is no separate mouseDoubleClickEvent.

    def mouseMoveEvent(self, event) -> None:
        was_hovering = self._hovering_close
        self._hovering_close = self.state == STATE_IDLE and self._close_btn_rect.contains(event.position())
        if self._hovering_close != was_hovering:
            self.setCursor(Qt.CursorShape.PointingHandCursor if self._hovering_close else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, event) -> None:
        if self._hovering_close:
            self._hovering_close = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.state == STATE_IDLE and self._close_btn_rect.contains(event.position()):
            self.close_requested.emit()
            return
        # Single click (anywhere else on the pill) opens the dashboard - no
        # double-click requirement.
        self.dashboard_requested.emit()

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        menu.addAction("Open Dashboard", self.dashboard_requested.emit)
        menu.addAction("Cycle Mode", self.cycle_mode_requested.emit)
        menu.addSeparator()
        menu.addAction("Hide Pill", self.hide_pill_requested.emit)
        menu.addAction("Quit Wisperno Completely", self.quit_requested.emit)
        menu.exec(event.globalPos())

    # --- Visibility toggle (tray) --------------------------------------------

    def set_pill_visible(self, visible: bool) -> None:
        if visible:
            self.show()
        else:
            self.hide()
