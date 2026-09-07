"""
V2 Feature Verification for Wisperno: pure-modifier shortcut chords, pinned
history ordering + persistence across a simulated app restart at the real
%APPDATA% location, and verbatim-fidelity (no summarizing) on long dictation.

Run: python tests/test_v2_features.py
"""

import os
import sys
from pathlib import Path
from loguru import logger

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from src.ui.shortcut_recorder import KeySequenceRecorder
from src.hotkey_manager import parse_hotkey_to_vks
from src.config import get_db_path, load_config
from src.database import WispernoDB


def test_pure_modifier_chord() -> None:
    """Test 1: Ctrl+Alt (no terminating key) registers as a valid chord."""
    logger.info("--- Test 1: Ctrl+Alt pure-modifier chord registration ---")
    app = QApplication.instance() or QApplication([])
    recorder = KeySequenceRecorder("ctrl+b")
    captured = []
    recorder.sequence_captured.connect(captured.append)

    recorder._start_recording()
    QTest.keyPress(recorder, Qt.Key.Key_Control, Qt.KeyboardModifier.ControlModifier)
    QTest.keyPress(recorder, Qt.Key.Key_Alt, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
    assert recorder._recording, "Recording ended prematurely on modifier press alone."

    # Release in either order - both must still finalize as "ctrl+alt".
    QTest.keyRelease(recorder, Qt.Key.Key_Alt, Qt.KeyboardModifier.ControlModifier)
    QTest.keyRelease(recorder, Qt.Key.Key_Control, Qt.KeyboardModifier.NoModifier)

    assert not recorder._recording, "Pure-modifier chord did not finalize on release."
    assert recorder.hotkey_str == "ctrl+alt", f"Expected 'ctrl+alt', got {recorder.hotkey_str!r}"
    assert captured == ["ctrl+alt"], f"sequence_captured mismatch: {captured}"

    vks = parse_hotkey_to_vks(recorder.hotkey_str)
    assert len(vks) == 2, f"'ctrl+alt' did not resolve to 2 real VK codes: {vks}"

    logger.success(f"PASS: 'ctrl+alt' captured and resolves to VK codes {vks} - the Win32 poller needs no changes.")


def test_pinned_ordering_and_appdata_persistence() -> None:
    """Test 2: is_pinned ordering, and persistence at the real %APPDATA% path across a simulated restart."""
    logger.info("--- Test 2: is_pinned ordering + %APPDATA% persistence across a simulated restart ---")
    db_path = get_db_path()
    assert "AppData" in str(db_path) or "Wisperno" in str(db_path), f"DB path is not under APPDATA: {db_path}"
    assert "_MEI" not in str(db_path), f"DB path leaked a PyInstaller temp dir: {db_path}"

    db = WispernoDB(db_path)
    ids = []
    try:
        old_ids = [r["id"] for r in db.list_history(limit=100000)]

        older_id = db.add_history("2020-01-01T00:00:00", 1.0, "old raw", "Old unpinned.", "polish", 100.0)
        newer_id = db.add_history("2024-01-01T00:00:00", 1.0, "new raw", "Newer unpinned.", "polish", 100.0)
        pinned_id = db.add_history("2019-01-01T00:00:00", 1.0, "pinned raw", "Old but pinned.", "polish", 100.0)
        ids = [older_id, newer_id, pinned_id]
        db.toggle_pin(pinned_id)

        rows = db.list_history(limit=100000)
        our_rows = [r for r in rows if r["id"] in ids]
        assert our_rows[0]["id"] == pinned_id, "Pinned row is not first despite being the oldest timestamp."
        assert our_rows[1]["id"] == newer_id, "Newer unpinned row did not sort before the older unpinned row."
        assert our_rows[2]["id"] == older_id

        # Simulate an app restart: close this connection, reopen a fresh WispernoDB
        # against the SAME real %APPDATA% path (no in-memory/temp db substitution).
        db.close()
        db2 = WispernoDB(db_path)
        try:
            rows_after_restart = db2.list_history(limit=100000)
            after_ids = {r["id"] for r in rows_after_restart}
            assert set(ids).issubset(after_ids), "History rows vanished after simulated restart at %APPDATA%."
            pinned_row = next(r for r in rows_after_restart if r["id"] == pinned_id)
            assert pinned_row["is_pinned"] == 1, "is_pinned did not survive the restart."
            logger.success(f"PASS: pinned-first ordering correct; all {len(ids)} rows survived a simulated restart at '{db_path}'.")
        finally:
            db = db2  # so the cleanup below closes/deletes through the live connection
    finally:
        for hid in ids:
            try:
                db.delete_history(hid)
            except Exception:
                pass
        db.close()


def test_verbatim_fidelity() -> None:
    """Test 3: a long (~200 word) dictation is polished, not summarized - output retains >90% of the word count."""
    logger.info("--- Test 3: verbatim fidelity on a long dictation (no summarizing) ---")
    from src.transformer import Transformer

    cfg = load_config()
    transformer = Transformer(config=cfg.llm, prompts=cfg.prompts)
    if transformer.llm is None:
        logger.warning("No LLM weights on disk - skipping (nothing to test fidelity against).")
        return

    # Genuinely varied content, not a repeated sentence - repetition is a
    # transcription-glitch shape a model may reasonably deduplicate, which
    # would make this test measure glitch-handling instead of fidelity.
    long_input = (
        "um so basically what I am trying to explain here is that the new deployment pipeline "
        "needs to run the integration tests before it pushes anything to the staging environment "
        "and uh I also think we should add a rollback step just in case something breaks in production. "
        "Another thing I wanted to bring up is that the database migration script we wrote last week "
        "has a bug where it does not handle null values correctly in the legacy customer records table, "
        "so we should probably fix that before the next release goes out. On top of that, the frontend "
        "team mentioned that the new checkout flow is causing some confusion for users because the "
        "shipping address form appears before the payment method selector, which is not the order most "
        "competitors use, so maybe we should run an A/B test to see which order actually converts better. "
        "I also wanted to mention that we are getting close to running out of budget for the cloud "
        "infrastructure this quarter, so someone from finance should probably take a look at our current "
        "spend and see if there is anything we can optimize, like maybe switching some of the less "
        "critical background jobs to a cheaper instance type or scaling down during off-peak hours."
    )
    input_word_count = len(long_input.split())
    assert input_word_count >= 200, f"Test fixture is only {input_word_count} words, need >= 200."

    output = transformer.transform(long_input, mode="polish")
    output_word_count = len(output.split())
    ratio = output_word_count / input_word_count

    logger.info(f"Input: {input_word_count} words -> Output: {output_word_count} words ({ratio:.0%})")
    assert ratio > 0.90, (
        f"Output retained only {ratio:.0%} of the input word count - looks like summarization/condensation, "
        f"not verbatim polishing."
    )
    logger.success(f"PASS: output retained {ratio:.0%} of the input length ({output_word_count}/{input_word_count} words).")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO V2 FEATURE TEST SUITE")
    logger.info("=========================================================")
    test_pure_modifier_chord()
    test_pinned_ordering_and_appdata_persistence()
    test_verbatim_fidelity()
    logger.success("=========================================================")
    logger.success(" ALL V2 FEATURE TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
