"""
Robust Multi-Key Shortcut Recorder Verification for Wisperno.
Drives KeySequenceRecorder with PySide6.QtTest.QTest (real Qt event
dispatch, not hand-built QKeyEvents) to prove single keys, 2-key, and
3-key modifier chords all register correctly.

Run: python tests/test_shortcut_recorder.py
"""

import os
import sys
from pathlib import Path
from loguru import logger

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no visible window needed

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from src.ui.shortcut_recorder import KeySequenceRecorder
from src.hotkey_manager import parse_hotkey_to_vks

CASES = [
    ("Single key: F8", Qt.Key.Key_F8, Qt.KeyboardModifier.NoModifier, "f8", 1),
    ("Two-key chord: Ctrl+B", Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier, "ctrl+b", 2),
    (
        "Three-key chord: Ctrl+Alt+V",
        Qt.Key.Key_V,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier,
        "ctrl+alt+v",
        3,
    ),
    (
        "Three-key chord: Ctrl+Shift+Space",
        Qt.Key.Key_Space,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        "ctrl+shift+space",
        3,
    ),
]


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO SHORTCUT RECORDER TEST SUITE")
    logger.info("=========================================================")

    app = QApplication.instance() or QApplication([])

    for label, key, modifiers, expected, expected_vk_count in CASES:
        logger.info(f"--- Testing: {label} ---")
        recorder = KeySequenceRecorder("ctrl+b")
        captured = []
        recorder.sequence_captured.connect(captured.append)

        recorder._start_recording()
        assert recorder._recording, f"[{label}] Clicking did not arm recording."

        QTest.keyClick(recorder, key, modifiers)

        assert not recorder._recording, f"[{label}] Recording did not end after the chord was pressed."
        assert recorder.hotkey_str == expected, f"[{label}] Expected {expected!r}, got {recorder.hotkey_str!r}"
        assert captured == [expected], f"[{label}] sequence_captured mismatch: {captured}"

        vks = parse_hotkey_to_vks(recorder.hotkey_str)
        assert len(vks) == expected_vk_count, (
            f"[{label}] Expected {expected_vk_count} real VK codes, got {len(vks)}: {vks}"
        )

        logger.success(f"PASS: {label} -> '{recorder.hotkey_str}' ({expected_vk_count} VK codes: {vks})")

    logger.success("=========================================================")
    logger.success(f" ALL {len(CASES)} SHORTCUT RECORDER TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
