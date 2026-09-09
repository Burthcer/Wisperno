"""
Dictionary sanitization verification for Wisperno.

Directly checks the REAL production database
(%APPDATA%\\Wisperno\\wisperno.db) for synthetic test junk ("word0" ->
"replacement0" style rows) - the exact pattern tests/test_ui_layout.py's
test_dictionary_add_word_dock_stays_pinned used to seed, before it was fixed
to use an isolated temp DB. This test exists specifically so that kind of
test-isolation regression gets caught immediately, not months later when a
user notices their real dictionary is full of junk.

Run: python tests/test_dictionary_sanitization.py
"""

import re
import sys
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

_JUNK_RE = re.compile(r"^(word|replacement|test|dummy)\d*$", re.IGNORECASE)


def test_production_database_has_no_synthetic_junk() -> None:
    from src.config import get_db_path
    from src.database import WispernoDB

    logger.info("--- Testing the REAL production dictionary for synthetic test junk ---")
    db_path = get_db_path()
    if not db_path.exists():
        logger.warning(f"No production database yet at {db_path} - nothing to check.")
        return

    db = WispernoDB(db_path)
    entries = db.list_dictionary()
    junk = [r for r in entries if _JUNK_RE.match(r["word"]) or _JUNK_RE.match(r["replacement"])]
    db.close()

    assert not junk, (
        f"Found {len(junk)} synthetic test entries in the REAL production dictionary: "
        f"{[(r['word'], r['replacement']) for r in junk[:5]]}{'...' if len(junk) > 5 else ''} - "
        f"a test harness wrote to the real database instead of an isolated temp one."
    )
    logger.success(f"PASS: production dictionary ({len(entries)} entries) has no synthetic test junk.")


def test_ui_layout_helper_uses_an_isolated_database() -> None:
    """Regression guard for the actual root cause: test_ui_layout.py's shared
    FakeEngine must never point at get_db_path() (the real production file)."""
    logger.info("--- Testing test_ui_layout.py: FakeEngine must use an isolated temp DB ---")
    source = (BASE_DIR / "tests" / "test_ui_layout.py").read_text(encoding="utf-8")
    assert "WispernoDB(get_db_path())" not in source, (
        "test_ui_layout.py's FakeEngine points at the REAL production database again - "
        "this is exactly what seeded 50 'word0'->'replacement0' junk rows into a real user's dictionary."
    )
    logger.success("PASS: test_ui_layout.py no longer touches the real production database.")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO DICTIONARY SANITIZATION TEST SUITE")
    logger.info("=========================================================")
    test_ui_layout_helper_uses_an_isolated_database()
    test_production_database_has_no_synthetic_junk()
    logger.success("=========================================================")
    logger.success(" ALL DICTIONARY SANITIZATION TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
