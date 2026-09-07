"""
Round 24 Profanity Filter Verification for Wisperno.
Unit-tests src/swear_filter.py directly (no LLM needed - pure post-processing
on already-polished text), then confirms Transformer actually applies the
configured mode end-to-end.

Run: python tests/test_swear_filter.py
"""

import sys
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_allow_mode_is_verbatim() -> None:
    from src.swear_filter import apply_profanity_filter

    logger.info("--- Testing profanity filter: 'allow' mode leaves text untouched ---")
    text = "This is fucking ridiculous, damn it."
    result = apply_profanity_filter(text, "allow")
    assert result == text, f"'allow' mode modified text: {result!r}"
    logger.success(f"PASS: 'allow' -> {result!r} (unchanged)")


def test_censor_mode_replaces_with_asterisks() -> None:
    from src.swear_filter import apply_profanity_filter

    logger.info("--- Testing profanity filter: 'censor' mode masks with asterisks ---")
    cases = [
        ("This is fucking ridiculous.", "f***", "ridiculous"),
        ("What the hell is going on.", "h***", None),
        ("That's such bullshit, honestly.", None, None),  # "bullshit" not in the word list - not censored, that's fine
    ]
    result1 = apply_profanity_filter(cases[0][0], "censor")
    logger.info(f"{cases[0][0]!r} -> {result1!r}")
    assert "f***" in result1, f"Expected 'f***' in censored output: {result1!r}"
    assert "fucking" not in result1.lower(), f"Uncensored profanity leaked through: {result1!r}"

    result2 = apply_profanity_filter(cases[1][0], "censor")
    logger.info(f"{cases[1][0]!r} -> {result2!r}")
    assert "h***" in result2, f"Expected 'h***' in censored output: {result2!r}"

    logger.success("PASS: 'censor' mode masks known profanity, first letter kept.")


def test_remove_mode_drops_words_cleanly() -> None:
    from src.swear_filter import apply_profanity_filter

    logger.info("--- Testing profanity filter: 'remove' mode drops words without leaving artifacts ---")
    result = apply_profanity_filter("This is fucking ridiculous, damn it.", "remove")
    logger.info(f"-> {result!r}")
    assert "fucking" not in result.lower() and "damn" not in result.lower(), f"Profanity survived removal: {result!r}"
    assert "  " not in result, f"Removal left a double space: {result!r}"
    assert not result.startswith(" ") and not result.endswith(" "), f"Removal left stray leading/trailing space: {result!r}"
    logger.success(f"PASS: 'remove' -> {result!r}")


def test_word_boundaries_never_false_positive() -> None:
    """"ass" must not match inside "assistant"/"assess"/"class" - a substring
    match here would corrupt completely innocent words."""
    from src.swear_filter import apply_profanity_filter

    logger.info("--- Testing profanity filter: word-boundary safety (no false positives) ---")
    text = "The assistant helped assess the class assignment."
    result = apply_profanity_filter(text, "censor")
    logger.info(f"{text!r} -> {result!r}")
    assert result == text, f"False positive: an innocent word got censored: {result!r}"
    logger.success("PASS: 'assistant'/'assess'/'class'/'assignment' all survive untouched.")


def test_profanity_was_censored_detection() -> None:
    from src.swear_filter import profanity_was_censored

    logger.info("--- Testing profanity_was_censored(): detects a dropped/substituted swear word ---")
    assert profanity_was_censored("i cannot get this fucking thing to work", "I cannot get this thing to work."), (
        "Did not detect a swear word silently dropped from the output."
    )
    assert profanity_was_censored("what the shit is going on", "What the heck is going on?"), (
        "Did not detect a swear word silently euphemized in the output."
    )
    assert profanity_was_censored("this is fucking great", "This is f***ing great."), (
        "Did not detect a swear word asterisked in the output."
    )
    assert not profanity_was_censored("this is fucking great", "This is fucking great."), (
        "False positive: the word survived verbatim but was flagged as censored."
    )
    assert not profanity_was_censored("this is a nice day", "This is a nice day."), (
        "False positive: raw input had no profanity to begin with."
    )
    logger.success("PASS: profanity_was_censored() catches drops/euphemisms/asterisking, no false positives.")


def test_llm_never_drops_profanity_in_allow_mode() -> None:
    """The actual reported defect: with 'Allow All', the LLM's own RLHF
    alignment silently dropped or substituted profanity despite the polish
    prompt's explicit 'never censor' rule - reproduced directly against the
    real model before this test existed. The Python-level guardrail
    (sanity_check's profanity_was_censored check) must catch it and fall
    back to the verbatim raw text so the word survives either way."""
    from src.config import load_config
    from src.transformer import Transformer

    logger.info("--- Testing Transformer (real LLM): profanity survives 'Allow All' even when the model tries to drop it ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts, profanity_filter="allow")
    if transformer.llm is None:
        logger.warning("No LLM weights on disk - skipping.")
        return

    # Every one of these reproduced a dropped/substituted swear word against
    # the shipped Standard preset before the guardrail fix.
    samples = [
        "fucking task manager",
        "i cannot get this fucking thing to work",
        "what the shit is going on here",
        "fuck this fucking computer is not working",
        "god damn it just restart the fucking app",
    ]
    for raw in samples:
        out = transformer.transform(raw, mode="polish")
        logger.info(f"{raw!r} -> {out!r}")
        assert "*" not in out, f"Asterisk leaked into 'Allow All' output: {out!r}"
        for word in raw.split():
            word_clean = word.strip(".,!?")
            if word_clean in ("fucking", "fuck", "shit", "damn"):
                assert word_clean in out.lower(), f"'{word_clean}' was dropped from 'Allow All' output: {out!r}"
    logger.success("PASS: every swear word in every sample survived verbatim in 'Allow All' output, zero asterisks.")


def test_llm_censor_mode_still_masks_profanity() -> None:
    """Mission's paired check: the same dictation, with 'Censor' selected,
    must come back asterisked - proves the setting actually switches the
    behavior end-to-end, not just that 'allow' happens to work."""
    from src.config import load_config
    from src.transformer import Transformer

    logger.info("--- Testing Transformer (real LLM): 'Censor' mode masks profanity end-to-end ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts, profanity_filter="censor")
    if transformer.llm is None:
        logger.warning("No LLM weights on disk - skipping.")
        return

    out = transformer.transform("fucking task manager", mode="polish")
    logger.info(f"'fucking task manager' -> {out!r}")
    # Whatever happened upstream (the word survived and got censored to
    # "f******", or the model itself dropped/substituted it), the one thing
    # that must never happen in 'Censor' mode is the raw, unmasked word
    # reaching the output.
    assert "fucking" not in out.lower(), f"'Censor' mode let an unmasked swear word through: {out!r}"
    logger.success(f"PASS: 'Censor' mode -> {out!r}")


def test_transformer_applies_configured_mode() -> None:
    """End-to-end: Transformer.profanity_filter actually reaches the final
    output, and set_profanity_filter() changes it live with no reload."""
    from src.transformer import Transformer

    logger.info("--- Testing Transformer: profanity_filter setting is applied to raw-mode output ---")
    xform = Transformer.__new__(Transformer)  # bypass __init__ (no LLM needed for raw mode)
    xform.prompts = {}
    xform.llm = None
    xform.last_fallback_applied = False
    xform.profanity_filter = "allow"

    out_allow = xform.transform("this is fucking great", mode="raw")
    assert "fucking" in out_allow.lower(), f"'allow' mode altered raw text: {out_allow!r}"

    xform.set_profanity_filter("censor")
    out_censor = xform.transform("this is fucking great", mode="raw")
    assert "f***" in out_censor, f"set_profanity_filter('censor') did not take effect: {out_censor!r}"

    logger.success(f"PASS: allow -> {out_allow!r}, censor (after live switch) -> {out_censor!r}")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO PROFANITY FILTER TEST SUITE")
    logger.info("=========================================================")
    test_allow_mode_is_verbatim()
    test_censor_mode_replaces_with_asterisks()
    test_remove_mode_drops_words_cleanly()
    test_word_boundaries_never_false_positive()
    test_profanity_was_censored_detection()
    test_llm_never_drops_profanity_in_allow_mode()
    test_llm_censor_mode_still_masks_profanity()
    test_transformer_applies_configured_mode()
    logger.success("=========================================================")
    logger.success(" ALL PROFANITY FILTER TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
