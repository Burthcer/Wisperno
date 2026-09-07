"""
MainWindow layout test for Wisperno: asserts the window is genuinely resizable
(not locked by a layout-computed minimum size blowing out to thousands of
pixels from an unwrapped label somewhere) and the sidebar can never collapse
or get squished by the content area.

Run: python tests/test_ui_layout.py
"""

import os
import sys
from pathlib import Path
from loguru import logger

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _build_main_window():
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication
    from src.config import load_config, get_db_path
    from src.database import WispernoDB
    from src.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])

    class FakeController:
        def set_pill_visible(self, v):
            pass

        def shutdown_application(self):
            pass

        def toggle_live_transcription(self):
            pass

    class FakeEngine(QObject):
        history_added = Signal(dict)

        def __init__(self):
            super().__init__()
            self.config = load_config()
            self.db = WispernoDB(get_db_path())
            self.transformer = None
            self.transcriber = None
            self.audio_worker = None

        def set_tap_toggle_hotkey(self, h): pass
        def set_tap_toggle_enabled(self, e): pass
        def set_hold_to_talk_hotkey(self, h): pass
        def set_hold_to_talk_enabled(self, e): pass
        def set_audio_device(self, i): pass
        def set_profanity_filter(self, m): pass
        def set_auto_llm_polish(self, e): pass
        def set_minimize_to_tray(self, e): pass
        def save_config(self): pass
        def restart_engines(self): pass
        def refresh_transform_hotkeys(self): pass

    engine = FakeEngine()
    win = MainWindow(engine, FakeController())
    win.show()
    # SystemUsageTab runs a 2s QTimer that outlives this single test function
    # (this same QApplication instance is reused test-to-test) - left running,
    # it eventually fires against a LATER test's already-closed db. No test
    # here exercises live telemetry refresh, so stop it immediately.
    win.system_usage_tab._timer.stop()
    app.processEvents()
    return app, win, engine


def test_window_resizing_unlocked() -> None:
    logger.info("--- Testing MainWindow: resizing is unlocked (no runaway minimum size) ---")
    app, win, engine = _build_main_window()
    try:
        min_size = win.minimumSize()
        max_size = win.maximumSize()
        logger.info(f"minimumSize={min_size.toTuple()} maximumSize={max_size.toTuple()}")

        assert min_size.width() >= 800, f"minimumSize width {min_size.width()} is suspiciously small."
        assert min_size.width() < 1000, (
            f"minimumSize width {min_size.width()} is far larger than expected (~880) - a layout-"
            f"computed minimum (an unwrapped long QLabel somewhere) is likely leaking through despite "
            f"the explicit setMinimumSize() call."
        )
        assert max_size.width() > 5000, f"maximumSize width {max_size.width()} is unexpectedly capped."

        # Actually exercise resizing, not just read the properties.
        win.resize(1600, 1000)
        app.processEvents()
        assert win.size().toTuple() == (1600, 1000), f"Window did not grow to 1600x1000: {win.size().toTuple()}"

        win.resize(880, 560)
        app.processEvents()
        assert win.size().toTuple() == (880, 560), f"Window did not shrink to its stated minimum: {win.size().toTuple()}"

        logger.success("PASS: window resizes freely and respects its stated 880x560 minimum (not a much larger computed one).")
    finally:
        engine.db.close()


def test_sidebar_never_collapses() -> None:
    logger.info("--- Testing MainWindow: sidebar stays exactly 220px across every resize ---")
    app, win, engine = _build_main_window()
    try:
        sidebar = win.centralWidget().layout().itemAt(0).widget()
        assert sidebar.width() == 220, f"Sidebar width at default size is {sidebar.width()}, not 220."

        for w, h in [(1600, 1000), (880, 560), (1020, 680)]:
            win.resize(w, h)
            app.processEvents()
            assert sidebar.width() == 220, f"Sidebar collapsed to {sidebar.width()}px at window size {w}x{h}."

        logger.success("PASS: sidebar stayed exactly 220px across every tested window size.")
    finally:
        engine.db.close()


def test_every_tab_has_bounded_minimum_width() -> None:
    """Regression guard: catches the actual root cause this round found - an
    unwrapped QLabel in any one tab silently inflating the whole window's
    minimum width by thousands of pixels via QStackedWidget's minimumSizeHint."""
    logger.info("--- Testing MainWindow: no single tab pushes minimum width past a sane ceiling ---")
    from PySide6.QtWidgets import QApplication

    app, win, engine = _build_main_window()
    try:
        CEILING = 1400
        for idx in range(win.stack.count()):
            page = win.stack.widget(idx)
            hint = page.minimumSizeHint()
            logger.info(f"Tab {idx} ('{win.nav.item(idx).text()}'): minimumSizeHint={hint.toTuple()}")
            assert hint.width() < CEILING, (
                f"Tab {idx} ('{win.nav.item(idx).text()}') has minimumSizeHint width {hint.width()}px - "
                f"over the {CEILING}px sanity ceiling. Look for an unwrapped QLabel with a long dynamic "
                f"string (setWordWrap(True) is the fix, as it was for Settings/Snippets/System Usage this round)."
            )
        logger.success(f"PASS: every tab's minimumSizeHint width stayed under {CEILING}px.")
    finally:
        engine.db.close()


def test_no_double_scrollbar_on_history() -> None:
    """Regression guard: the outer content QScrollArea (added to decouple window
    resizability from tab content width) must never show its own scrollbar - it
    would nest a second, redundant one alongside History's own internal one."""
    logger.info("--- Testing MainWindow: no double vertical scrollbar on History ---")
    from PySide6.QtCore import Qt

    app, win, engine = _build_main_window()
    try:
        content_scroll = win.centralWidget().layout().itemAt(1).widget()
        assert content_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff, (
            "Outer content scroll area's vertical scrollbar is not disabled - it will show "
            "a redundant second scrollbar alongside any tab with its own internal QScrollArea "
            "(History's card list)."
        )
        assert win.history_tab.scroll.verticalScrollBarPolicy() != Qt.ScrollBarPolicy.ScrollBarAlwaysOff, (
            "History's own internal scroll area is disabled - the history list would not scroll at all."
        )
        logger.success("PASS: exactly one scrollbar (History's own) can ever render, not two.")
    finally:
        engine.db.close()


def test_dictionary_actions_column_visible() -> None:
    """Regression guard: the Category/Actions columns had no explicit resize
    mode, so Qt auto-sized them from their header text - an empty '""' header
    on Actions collapsed it to near-zero width, clipping the edit/delete
    buttons out of view. Fixed widths must keep both columns usable."""
    logger.info("--- Testing DictionaryTab: Category/Actions columns stay wide enough to use ---")
    from PySide6.QtWidgets import QPushButton

    app, win, engine = _build_main_window()
    try:
        tab = win.dictionary_tab
        win.stack.setCurrentWidget(tab)  # a non-active QStackedWidget page reports its children as not-visible
        engine.db.add_dictionary_entry("pie torch", "PyTorch", False, "Tech")
        tab.refresh()
        app.processEvents()

        table = tab.table
        assert table.columnWidth(2) >= 80, f"Category column too narrow: {table.columnWidth(2)}px"
        assert table.columnWidth(3) >= 80, f"Actions column too narrow: {table.columnWidth(3)}px"
        assert table.rowCount() > 0, "No dictionary rows rendered to check."

        actions_widget = table.cellWidget(0, 3)
        assert actions_widget is not None, "Actions column has no cell widget."
        buttons = actions_widget.findChildren(QPushButton)
        assert len(buttons) == 2, f"Expected 2 action buttons (edit/delete), found {len(buttons)}."
        assert all(b.isVisible() for b in buttons), "An action button is not visible."

        logger.success(f"PASS: Category={table.columnWidth(2)}px, Actions={table.columnWidth(3)}px, "
                        f"{len(buttons)} action buttons visible.")
    finally:
        engine.db.close()


def test_dictionary_add_word_dock_stays_pinned() -> None:
    """Regression guard: the Add Word dock must stay a fixed-height dock, not
    grow the whole tab (and get pushed off past a long scroll) as entries pile
    up - the table (with stretch=1 + Expanding policy) absorbs the growth via
    its own native scrolling instead."""
    logger.info("--- Testing DictionaryTab: Add Word dock stays pinned regardless of entry count ---")
    app, win, engine = _build_main_window()
    try:
        tab = win.dictionary_tab
        win.stack.setCurrentWidget(tab)
        for i in range(50):
            engine.db.add_dictionary_entry(f"word{i}", f"replacement{i}")
        tab.refresh()
        app.processEvents()

        assert tab.table.rowCount() >= 50, f"Expected at least 50 rows, got {tab.table.rowCount()}."
        form_height_50 = tab.add_btn.parent().sizeHint().height()

        # The dock's own natural height must not depend on row count - it's a
        # fixed-height row of inputs, not something that grows with the list.
        assert form_height_50 < 150, f"Add Word dock height ({form_height_50}px) grew with row count - not pinned."
        assert tab.add_btn.isVisible(), "Add Word button is not visible with 50 entries loaded."
        logger.success(f"PASS: Add Word dock stayed {form_height_50}px tall with 50 dictionary entries loaded.")
    finally:
        engine.db.close()


def test_scroll_does_not_leak_across_tabs() -> None:
    """Regression guard for the reported cross-tab scroll bug: scrolling one
    tab's content must never move the shared outer QStackedWidget wrapper
    (content_scroll), and switching tabs must reset the incoming tab's own
    scroll position to the top."""
    logger.info("--- Testing MainWindow: scrolling one tab does not leak into another ---")
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    app, win, engine = _build_main_window()
    try:
        # The outer wrapper must never itself scroll - find it via the stack's parent chain.
        content_scroll = win.stack.parent().parent()
        assert content_scroll.verticalScrollBar().value() == 0
        wheel_event = QWheelEvent(
            QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(0, -480),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        content_scroll.wheelEvent(wheel_event)
        app.processEvents()
        assert content_scroll.verticalScrollBar().value() == 0, (
            "The outer content_scroll wrapper moved on a wheel event - it must be fully inert."
        )

        # Scroll History's own inner scroll area down, then switch to Transforms
        # and back - the incoming tab must always land at the top (0), not
        # wherever a PREVIOUS visit happened to leave it.
        win.stack.setCurrentWidget(win.history_tab)  # a non-active QStackedWidget page never lays out its children
        bar = win.history_tab.scroll.verticalScrollBar()
        # Force a scrollable range directly rather than relying on the offscreen
        # test platform's layout engine to compute one organically from N cards -
        # what's under test is reset_scroll() actually being invoked on tab
        # switch, not Qt's own geometry computation.
        bar.setRange(0, 1000)
        bar.setValue(500)
        assert bar.value() > 0, "Setup failed: History didn't actually scroll."

        win._on_nav_changed(1)  # Transforms
        app.processEvents()
        win._on_nav_changed(0)  # back to History
        app.processEvents()
        assert win.history_tab.scroll.verticalScrollBar().value() == 0, (
            "History's scroll position was not reset to the top on re-entry."
        )
        logger.success("PASS: outer wrapper never scrolls, and re-entering a tab always resets its own scroll to the top.")
    finally:
        engine.db.close()


def test_settings_reopens_on_general_tab() -> None:
    """Returning to Settings must always land on General, even if Advanced was
    left selected on a previous visit."""
    logger.info("--- Testing SettingsTab: always resets to General on re-open ---")
    app, win, engine = _build_main_window()
    try:
        settings = win.settings_tab
        settings.sub_tabs.setCurrentIndex(1)  # Advanced
        assert settings.sub_tabs.currentIndex() == 1, "Setup failed: could not select Advanced."

        settings.hide()
        app.processEvents()
        settings.show()
        app.processEvents()

        assert settings.sub_tabs.currentIndex() == 0, "Settings did not reset to General (index 0) on re-open."
        logger.success("PASS: Settings always reopens on the General sub-tab.")
    finally:
        engine.db.close()


def run_all_tests() -> None:
    logger.info("=" * 70)
    logger.info("  WISPERNO UI LAYOUT TEST SUITE")
    logger.info("=" * 70)
    test_window_resizing_unlocked()
    test_sidebar_never_collapses()
    test_every_tab_has_bounded_minimum_width()
    test_no_double_scrollbar_on_history()
    test_dictionary_actions_column_visible()
    test_dictionary_add_word_dock_stays_pinned()
    test_scroll_does_not_leak_across_tabs()
    test_settings_reopens_on_general_tab()
    logger.success("=" * 70)
    logger.success(" ALL UI LAYOUT TESTS PASSED!")
    logger.success("=" * 70)


if __name__ == "__main__":
    run_all_tests()
