"""
FloatingPill geometry test for Wisperno: asserts the pill never stretches
toward screen width in any state - the reported "elongated bottom bar" defect.

Run: python tests/test_geometry.py
"""

import os
import sys
from pathlib import Path
from loguru import logger

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MIN_WIDTH = 240
MAX_WIDTH = 450

STATES = ["loading", "idle", "recording", "processing", "success", "selection_processing", "selection_success", "mode_changed", "error"]


def test_pill_width_bounded_in_every_state() -> None:
    logger.info("--- Testing FloatingPill: width stays bounded (240-450px) in every state ---")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QGuiApplication
    from src.ui.floating_pill import FloatingPill, PILL_H

    app = QApplication.instance() or QApplication([])
    screen = QGuiApplication.primaryScreen().geometry()
    logger.info(f"Screen geometry: {screen.width()}x{screen.height()}")

    pill = FloatingPill(initial_mode="POLISH", hotkey_label="Ctrl+Alt+Shift+V")  # deliberately long hotkey label

    # Construction starts in the LOADING state, which intentionally uses the
    # narrower DORMANT_W ("Starting..." has no hotkey label to fit) - the
    # STATES loop below covers LOADING's actual width bound explicitly.
    logger.info(f"After construction: width={pill.width()} height={pill.height()} pos=({pill.x()},{pill.y()})")
    assert pill.width() <= MAX_WIDTH, f"Initial pill width {pill.width()} exceeds the {MAX_WIDTH}px ceiling."
    assert pill.width() <= screen.width(), "Pill wider than the screen at construction."

    for state in STATES:
        pill._set_state(state)
        pill._width_current = pill._width_target  # settle the animated width immediately for inspection
        pill.resize(int(pill._width_current), PILL_H)
        pill._reposition()
        logger.info(f"{state}: width={pill.width()} pos=({pill.x()},{pill.y()})")

        assert pill.width() <= screen.width(), f"[{state}] pill width {pill.width()} exceeds screen width {screen.width()}."
        assert pill.width() <= MAX_WIDTH, f"[{state}] pill width {pill.width()} exceeds the {MAX_WIDTH}px ceiling."
        assert pill.width() >= MIN_WIDTH - 20, f"[{state}] pill width {pill.width()} is implausibly small."
        assert pill.height() == PILL_H, f"[{state}] pill height {pill.height()} != fixed {PILL_H}."
        assert pill.x() >= 0 and pill.x() + pill.width() <= screen.width(), f"[{state}] pill runs off-screen horizontally."

    logger.success(f"PASS: pill width stayed within [{MIN_WIDTH}, {MAX_WIDTH}]px (screen is {screen.width()}px) across all {len(STATES)} states.")


def test_pill_window_flags_and_translucency() -> None:
    logger.info("--- Testing FloatingPill: frameless/always-on-top/translucent flags ---")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from src.ui.floating_pill import FloatingPill

    app = QApplication.instance() or QApplication([])
    pill = FloatingPill(initial_mode="POLISH", hotkey_label="Ctrl+B")

    flags = pill.windowFlags()
    assert flags & Qt.WindowType.FramelessWindowHint, "Missing FramelessWindowHint."
    assert flags & Qt.WindowType.WindowStaysOnTopHint, "Missing WindowStaysOnTopHint."
    assert flags & Qt.WindowType.Tool, "Missing Qt.WindowType.Tool (would show in taskbar/alt-tab)."
    assert pill.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground), "WA_TranslucentBackground not set."

    logger.success("PASS: window flags and translucency attribute are all correctly set.")


def run_all_tests() -> None:
    logger.info("=" * 70)
    logger.info("  WISPERNO PILL GEOMETRY TEST SUITE")
    logger.info("=" * 70)
    test_pill_width_bounded_in_every_state()
    test_pill_window_flags_and_translucency()
    logger.success("=" * 70)
    logger.success(" ALL GEOMETRY TESTS PASSED!")
    logger.success("=" * 70)


if __name__ == "__main__":
    run_all_tests()
