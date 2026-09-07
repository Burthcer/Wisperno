"""
Qt System Tray Integration for Wisperno (QSystemTrayIcon).
Menu: Open Dashboard, Toggle Floating Pill, Active Mode submenu, Trigger Mechanic
submenu, Exit Wisperno.
"""

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QIcon, QAction, QActionGroup, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QSystemTrayIcon, QMenu

MODES = ["polish", "prompt_engineer", "bullets", "code", "raw"]
MODE_LABELS = {"polish": "Polish", "prompt_engineer": "Prompt Engineer", "bullets": "Bullets", "code": "Code & CLI", "raw": "Raw"}


def build_icon() -> QIcon:
    """Draw a simple mic-dot glyph in-memory so no external asset is strictly required."""
    size = 64
    pix = QPixmap(size, size)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(0)
    p.setBrush(QColor(125, 87, 193))
    p.drawEllipse(6, 6, size - 12, size - 12)
    p.setBrush(QColor(245, 245, 247))
    p.drawEllipse(size // 2 - 9, size // 2 - 13, 18, 18)
    p.drawRoundedRect(size // 2 - 2, size // 2 + 4, 4, 10, 2, 2)
    p.end()
    return QIcon(pix)


class TrayIcon(QObject):
    """System tray icon wired to a WispernoEngine-like controller."""

    open_dashboard_requested = Signal()

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller

        self.icon = QSystemTrayIcon(build_icon(), parent)
        self.icon.setToolTip("Wisperno")

        menu = QMenu()
        menu.addAction("Open Dashboard", self.open_dashboard_requested.emit)
        self._pill_action = menu.addAction("Toggle Floating Pill", self._toggle_pill)
        menu.addSeparator()

        mode_menu = menu.addMenu("Active Mode")
        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_actions = {}
        for mode in MODES:
            action = QAction(MODE_LABELS[mode], self, checkable=True)
            action.triggered.connect(lambda checked, m=mode: self.controller.set_active_mode(m))
            mode_menu.addAction(action)
            self._mode_group.addAction(action)
            self._mode_actions[mode] = action

        # Two independent on/off triggers now (not a mutually-exclusive
        # picker) - either, both, or neither can be checked at once.
        trigger_menu = menu.addMenu("Dictation Triggers")
        self._trigger_actions = {}
        for key, label, setter in [
            ("tap_toggle", "Tap-to-Toggle", "set_tap_toggle_enabled"),
            ("hold_to_talk", "Hold-to-Talk", "set_hold_to_talk_enabled"),
        ]:
            action = QAction(label, self, checkable=True)
            action.triggered.connect(lambda checked, s=setter: getattr(self.controller, s)(checked))
            trigger_menu.addAction(action)
            self._trigger_actions[key] = action

        menu.addSeparator()
        menu.addAction("Exit Wisperno", self._exit)

        self.icon.setContextMenu(menu)
        self.icon.activated.connect(self._on_activated)
        self.refresh()
        self.icon.show()

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.open_dashboard_requested.emit()

    def refresh(self) -> None:
        """Sync checkmarks with current controller state."""
        mode = self.controller.get_active_mode()
        if mode in self._mode_actions:
            self._mode_actions[mode].setChecked(True)
        cfg = getattr(self.controller, "config", None)
        if cfg is not None:
            self._trigger_actions["tap_toggle"].setChecked(cfg.tap_toggle_enabled)
            self._trigger_actions["hold_to_talk"].setChecked(cfg.hold_to_talk_enabled)

    def _toggle_pill(self) -> None:
        if hasattr(self.controller, "toggle_pill"):
            self.controller.toggle_pill()

    def _exit(self) -> None:
        if hasattr(self.controller, "shutdown_application"):
            self.controller.shutdown_application()
        else:
            from PySide6.QtWidgets import QApplication

            QApplication.quit()
