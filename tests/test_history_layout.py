"""
History tab layout/filtering verification for Wisperno.

1. Accordion-jitter regression test: toggling "Show Raw Transcript" must never
   change the header row's or any header badge's height - see
   src/ui/history_tab.py's _badge()/_build_header() docstrings for the root
   cause this fixes (a badge/header row with no fixed height gets
   renegotiated, and can transiently stretch, during the card's re-layout
   when a sibling widget's visibility changes).
2. The new segmented filter bar: [Dictations] / [Live Transcripts] / [All]
   return the correct subsets by entry_type.

Uses a real (temp-file) WispernoDB, not the shared %APPDATA% one other UI
tests in this repo use - filter-count assertions here need to know EXACTLY
what rows exist, which a shared real-user database can't guarantee.

Run: python tests/test_history_layout.py
"""

import os
import sys
import tempfile
from pathlib import Path

from loguru import logger

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _make_history_tab(db):
    from PySide6.QtWidgets import QApplication
    from src.ui.history_tab import HistoryTab

    app = QApplication.instance() or QApplication([])

    class _FakeVocabulary:
        def add_entry(self, *args, **kwargs):
            pass

    class _FakeEngine:
        def __init__(self, db):
            self.db = db
            self.vocabulary = _FakeVocabulary()

    tab = HistoryTab(_FakeEngine(db))
    tab.show()
    app.processEvents()
    return app, tab


def _temp_db():
    from src.database import WispernoDB

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return WispernoDB(path), path


def test_header_badge_height_stable_across_accordion_toggle() -> None:
    from PySide6.QtWidgets import QLabel
    from src.ui.history_tab import BADGE_HEIGHT, HEADER_HEIGHT

    logger.info("--- Testing HistoryTab: badge/header height stays fixed through the raw-transcript toggle ---")
    db, db_path = _temp_db()
    try:
        db.add_history("2026-01-01T00:00:00", 12.3, "um so this is a test", "So this is a test.", "polish", 250.0)
        app, tab = _make_history_tab(db)

        card = tab.list_layout.itemAt(0).widget()
        assert card is not None, "Expected one history card"
        header_widget = card.layout().itemAt(0).widget()
        assert header_widget.height() == HEADER_HEIGHT, f"Header height was {header_widget.height()}px before toggle, expected {HEADER_HEIGHT}px"
        badges_before = {id(lbl): lbl.height() for lbl in header_widget.findChildren(QLabel)}
        assert badges_before, "Expected at least one badge/label in the header"
        assert all(h == BADGE_HEIGHT for h in badges_before.values()), f"Badge heights before toggle: {badges_before}"

        # Find and click "Show Raw Transcript" - the exact interaction the
        # mission reports triggers the jitter.
        from PySide6.QtWidgets import QPushButton
        raw_toggle = next(btn for btn in card.findChildren(QPushButton) if "Raw Transcript" in btn.text())
        raw_toggle.click()
        app.processEvents()

        assert header_widget.height() == HEADER_HEIGHT, f"Header height was {header_widget.height()}px AFTER toggle, expected {HEADER_HEIGHT}px (accordion jitter regressed)"
        badges_after = {id(lbl): lbl.height() for lbl in header_widget.findChildren(QLabel)}
        assert all(h == BADGE_HEIGHT for h in badges_after.values()), f"Badge heights after toggle: {badges_after} (accordion jitter regressed)"

        # Toggle back (collapse) - must also stay stable, not just on the way open.
        raw_toggle.click()
        app.processEvents()
        assert header_widget.height() == HEADER_HEIGHT
        assert all(lbl.height() == BADGE_HEIGHT for lbl in header_widget.findChildren(QLabel))

        db.close()
        logger.success(f"PASS: header ({HEADER_HEIGHT}px) and every badge ({BADGE_HEIGHT}px) held constant through expand + collapse.")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_filter_segments_return_correct_subsets() -> None:
    logger.info("--- Testing HistoryTab: [Dictations] / [Live Transcripts] / [All] filter segments ---")
    db, db_path = _temp_db()
    try:
        db.add_history("2026-01-01T00:00:00", 5.0, "raw dictation", "Raw dictation.", "polish", 200.0)
        db.add_live_transcript("2026-01-01T00:05:00", 180.0, "raw live text", "Cleaned live text.")
        app, tab = _make_history_tab(db)

        def card_count() -> int:
            return tab.list_layout.count() - 1  # last item is the trailing stretch

        tab._set_filter("all")
        assert card_count() == 2, f"'All' should show both rows, got {card_count()}"

        tab._set_filter("dictation")
        assert card_count() == 1, f"'Dictations' should show exactly the dictation row, got {card_count()}"

        tab._set_filter("live_transcript")
        assert card_count() == 1, f"'Live Transcripts' should show exactly the live-session row, got {card_count()}"

        tab._set_filter("pinned")
        assert card_count() == 0, "Neither row is pinned yet"

        db.close()
        logger.success("PASS: All/Dictations/Live Transcripts/Pinned each return the correct row subset.")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO HISTORY TAB LAYOUT/FILTER TEST SUITE")
    logger.info("=========================================================")
    test_header_badge_height_stable_across_accordion_toggle()
    test_filter_segments_return_correct_subsets()
    logger.success("=========================================================")
    logger.success(" ALL HISTORY LAYOUT TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
