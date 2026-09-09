"""
Typography spacing normalization verification for Wisperno.

Covers src/direct_formatter.py's normalize_punctuation_spacing() (shared by
the rule formatter and transformer.py's basic_capitalize()/clean_output())
plus the real bug this round's own testing found while verifying it: two
ordinal words ("first"/"second") used in normal prose - not as a spoken list
- were misdetected as a numbered list once space-normalization put them
right after sentence-ending punctuation, producing garbage output ("1. .").
Fixed in _split_list() by gating on real word content between cues, not just
cue count - covered directly here, not just incidentally.

Run: python tests/test_typography_spacing.py
"""

import sys
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_missing_space_after_punctuation_is_inserted() -> None:
    from src.direct_formatter import format_direct

    logger.info("--- Testing format_direct(): missing space after punctuation is inserted ---")
    assert format_direct("first.second,third") == "First. Second, third.", format_direct("first.second,third")
    assert format_direct("Hello.How are you?") == "Hello. How are you?", format_direct("Hello.How are you?")
    assert format_direct("end of sentence.next sentence") == "End of sentence. Next sentence.", (
        format_direct("end of sentence.next sentence")
    )
    logger.success("PASS: missing space after punctuation inserted and following letter capitalized.")


def test_numeric_punctuation_is_never_touched() -> None:
    from src.direct_formatter import format_direct

    logger.info("--- Testing format_direct(): decimals and thousands-separators stay untouched ---")
    result = format_direct("the value is 3.14 and there were 1,234 attendees")
    assert "3.14" in result and "1,234" in result, result
    logger.success(f"PASS: numeric punctuation preserved: {result!r}")


def test_two_ordinal_words_in_prose_do_not_become_a_fake_list() -> None:
    """The real bug this round's testing caught: normalize_punctuation_spacing()
    turning 'first.second,third' into 'first. second, third' put 'second'
    right after sentence-ending punctuation, which used to satisfy the
    numbered-list detector's cue-count check even though there's no actual
    list content between the cues - producing '1. .' instead of a normal
    sentence."""
    from src.direct_formatter import format_direct

    logger.info("--- Testing format_direct(): ordinal words in plain prose don't misfire as a list ---")
    result = format_direct("first.second,third")
    assert "\n" not in result, f"Misfired as a multi-line list: {result!r}"
    assert result == "First. Second, third.", result
    logger.success(f"PASS: 'first.second,third' formatted as prose, not a fake list: {result!r}")


def test_real_spoken_list_still_detected() -> None:
    """Regression guard alongside the fix above: a GENUINE spoken list (real
    words between cues, no punctuation forcing a false sentence boundary)
    must still be detected."""
    from src.direct_formatter import format_direct

    logger.info("--- Testing format_direct(): a real spoken numbered list still works ---")
    result = format_direct("we need three things first buy milk second get eggs third go home")
    assert result.split("\n") == ["1. Buy milk.", "2. Get eggs.", "3. Go home."], result
    logger.success("PASS: a genuine spoken list is still detected correctly.")


def test_llm_path_also_normalizes_spacing() -> None:
    """transformer.py's basic_capitalize() (the guardrail-fallback path) must
    also apply the same fix, not just the zero-LLM formatter."""
    from src.transformer import basic_capitalize

    logger.info("--- Testing transformer.basic_capitalize(): also normalizes punctuation spacing ---")
    result = basic_capitalize("end of sentence.next sentence")
    # basic_capitalize()'s own sentence-split regex requires the space to
    # already be there (see its docstring) - normalize_punctuation_spacing()
    # runs first specifically so this split can find the boundary at all.
    assert result == "End of sentence. Next sentence.", result
    logger.success(f"PASS: transformer.basic_capitalize() also fixes fused punctuation: {result!r}")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO TYPOGRAPHY SPACING TEST SUITE")
    logger.info("=========================================================")
    test_missing_space_after_punctuation_is_inserted()
    test_numeric_punctuation_is_never_touched()
    test_two_ordinal_words_in_prose_do_not_become_a_fake_list()
    test_real_spoken_list_still_detected()
    test_llm_path_also_normalizes_spacing()
    logger.success("=========================================================")
    logger.success(" ALL TYPOGRAPHY SPACING TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
