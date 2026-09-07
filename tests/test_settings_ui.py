"""
KeySequenceRecorder Verification for Wisperno.
Drives the widget with synthetic Qt key events (offscreen platform, no visible
window needed) to prove the reported "Press keys... doesn't register anything"
bug is fixed: clicking arms recording, the next chord is captured and
formatted into the canonical string src.hotkey_manager expects, and Escape
cancels back to the previous value.

Run: python tests/test_settings_ui.py
"""

import os
import sys
from pathlib import Path
from loguru import logger

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no visible window needed

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from src.ui.shortcut_recorder import KeySequenceRecorder
from src.hotkey_manager import parse_hotkey_to_vks


def _press(widget, key, modifiers=Qt.KeyboardModifier.NoModifier):
    event = QKeyEvent(QEvent.Type.KeyPress, key, modifiers)
    widget.keyPressEvent(event)


def test_capture_ctrl_b() -> None:
    logger.info("--- Testing KeySequenceRecorder: capture Ctrl+B ---")
    app = QApplication.instance() or QApplication([])
    recorder = KeySequenceRecorder("ctrl+b")
    captured = []
    recorder.sequence_captured.connect(captured.append)

    recorder._start_recording()
    assert recorder._recording, "Clicking did not arm recording."
    assert "Recording" in recorder.text()

    _press(recorder, Qt.Key.Key_Control, Qt.KeyboardModifier.ControlModifier)  # bare modifier: ignored
    assert recorder._recording, "A bare modifier press ended recording prematurely."

    _press(recorder, Qt.Key.Key_G, Qt.KeyboardModifier.ControlModifier)
    assert not recorder._recording, "Recording did not end after a real chord was pressed."
    assert recorder.hotkey_str == "ctrl+g", f"Expected 'ctrl+g', got {recorder.hotkey_str!r}"
    assert captured == ["ctrl+g"], f"sequence_captured did not fire with the right value: {captured}"

    # Must resolve to real Win32 VK codes downstream, not just look right as a string.
    vks = parse_hotkey_to_vks(recorder.hotkey_str)
    assert len(vks) == 2, f"Captured hotkey did not resolve to 2 VK codes: {vks}"

    logger.success(f"PASS: captured '{recorder.hotkey_str}' -> button text '{recorder.text()}', VKs={vks}")


def test_escape_cancels() -> None:
    logger.info("--- Testing KeySequenceRecorder: Escape cancels back to the previous value ---")
    app = QApplication.instance() or QApplication([])
    recorder = KeySequenceRecorder("ctrl+b")

    recorder._start_recording()
    _press(recorder, Qt.Key.Key_Escape)

    assert not recorder._recording
    assert recorder.hotkey_str == "ctrl+b", f"Escape should restore the previous value, got {recorder.hotkey_str!r}"
    logger.success("PASS: Escape restored the previous hotkey without committing a new one.")


def test_wheel_does_not_change_settings_widgets() -> None:
    """Regression guard: scrolling the mouse wheel over the GPU-offload slider,
    the Engine Preset dropdown, or the VAD sensitivity spinbox while scrolling
    past them must never silently change their value - only an explicit click/
    drag or the arrow keys should. Verified by actually sending a QWheelEvent,
    not just checking a wheelEvent override exists."""
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    from tests.test_ui_layout import _build_main_window

    logger.info("--- Testing Settings widgets: mouse wheel never changes slider/combo/spin values ---")
    app, win, engine = _build_main_window()
    try:
        settings_tab = next(
            win.stack.widget(i) for i in range(win.stack.count())
            if type(win.stack.widget(i)).__name__ == "SettingsTab"
        )
        adv = settings_tab.advanced_tab
        wheel_event = QWheelEvent(
            QPointF(0, 0), QPointF(0, 0), QPoint(0, 120), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        for name, widget in [
            ("GPU offload slider", adv.n_gpu_layers_slider),
            ("Engine Preset dropdown", adv.preset_cb),
            ("VAD sensitivity spinbox", adv.vad_sensitivity_spin),
        ]:
            before = widget.value() if hasattr(widget, "value") else widget.currentIndex()
            app.sendEvent(widget, wheel_event)
            after = widget.value() if hasattr(widget, "value") else widget.currentIndex()
            assert before == after, f"{name} changed from {before} to {after} on a wheel event over it."
            logger.info(f"{name}: {before} -> {after} after wheel event (unchanged, as required).")

        logger.success("PASS: mouse wheel does not change any Settings slider/combo/spin value.")
    finally:
        engine.db.close()


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO SETTINGS UI TEST SUITE")
    logger.info("=========================================================")
    test_capture_ctrl_b()
    test_escape_cancels()
    test_wheel_does_not_change_settings_widgets()
    logger.success("=========================================================")
    logger.success(" ALL SETTINGS UI TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
