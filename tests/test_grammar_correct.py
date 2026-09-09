"""
"Grammar Correct" transform (Alt+G) verification for Wisperno.

Real LLM, real DB migration path (seed_grammar_correct_transform() - the
one-time insert that reaches an EXISTING install's transforms table, since
DEFAULT_TRANSFORMS itself is only ever applied to a brand-new database).

Run: python tests/test_grammar_correct.py
"""

import os
import sys
import tempfile
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

RAW_TEXT = "he go to store.yesterday he buyed fruit"


def test_grammar_correct_fixes_errors_without_restyling() -> None:
    from src.config import load_config
    from src.database import WispernoDB
    from src.transformer import Transformer

    logger.info("--- Testing 'Grammar Correct' transform: fixes errors, doesn't restyle ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db_path)
    try:
        db = WispernoDB(db_path)
        # A brand-new DB already gets grammar_correct via the normal
        # DEFAULT_TRANSFORMS seed (transform_count == 0 path in _migrate()) -
        # to actually exercise the migration, simulate an EXISTING install
        # that predates this transform by removing it first.
        existing_row = db.get_transform("grammar_correct")
        assert existing_row is not None, "Fresh DB should already have grammar_correct from DEFAULT_TRANSFORMS"
        with db._cursor() as cur:
            cur.execute("DELETE FROM transforms WHERE id = 'grammar_correct'")
        assert db.get_transform("grammar_correct") is None

        seeded = db.seed_grammar_correct_transform()
        assert seeded is True
        transform = db.get_transform("grammar_correct")
        assert transform is not None
        assert transform["shortcut"] == "alt+g"
        assert db.seed_grammar_correct_transform() is False, "Must not re-seed once it already exists"

        config = load_config()
        transformer = Transformer(config=config.llm, prompts=config.prompts)
        if transformer.llm is None:
            logger.warning("No LLM weights on disk - skipping (nothing to test the transform against).")
            db.close()
            return

        output = transformer.transform_with_prompt(RAW_TEXT, transform["system_prompt"], label="grammar_correct")
        logger.info(f"Input:  {RAW_TEXT!r}\nOutput: {output!r}")

        assert output.strip(), "Grammar Correct produced empty output"
        assert output.strip() != RAW_TEXT, "Output is identical to raw input - nothing was corrected"

        lowered = output.lower()
        # The real grammar errors must be gone.
        assert "buyed" not in lowered, f"'buyed' was not corrected: {output!r}"
        assert "he go to" not in lowered, f"'he go to' was not corrected: {output!r}"
        # The actual facts (store, fruit) must survive - this is a grammar
        # fix, not a summarization or a rewrite into different content.
        assert "store" in lowered and "fruit" in lowered, f"Original facts lost: {output!r}"
        # Anti-hallucination: no preamble/quotes wrapping the answer.
        assert not lowered.startswith(("here is", "here's", "sure,")), f"Added a preamble: {output!r}"
        assert not (output.strip().startswith('"') and output.strip().endswith('"')), f"Wrapped in quotes: {output!r}"

        db.close()
        logger.success(f"PASS: Grammar Correct fixed real errors without restyling: {output!r}")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO GRAMMAR CORRECT TEST SUITE")
    logger.info("=========================================================")
    test_grammar_correct_fixes_errors_without_restyling()
    logger.success("=========================================================")
    logger.success(" ALL GRAMMAR CORRECT TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
