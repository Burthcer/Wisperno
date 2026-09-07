"""
Round 21 Conversational Guardrail Verification for Wisperno.
Two layers under test, per the mission's own framing:
1. The prompt itself (config.yaml's polish prompt) - does the live LLM stay
   headless and preserve exact vocabulary on realistic adversarial input.
2. src/transformer.py's programmatic sanity_check()/basic_capitalize() - the
   Python-level safety net that fires independent of whether the model
   obeyed the prompt, unit-tested directly against synthetic bad output so
   the guardrail logic itself is verified even if the live LLM never drifts.

Run: python tests/test_conversational_guard.py
"""

import sys
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_llm_does_not_answer_or_solicit() -> None:
    """Mission's Test Input 1: a direct question must come back as a cleaned
    question, never answered, and never met with the model asking its own
    question back (the reported 'What is your goal?' leak)."""
    from src.config import load_config
    from src.transformer import Transformer

    logger.info("--- Testing Transformer: dictated question is neither answered nor countered ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)
    if transformer.llm is None:
        logger.warning("No LLM weights on disk - skipping (nothing to test the guardrail against).")
        return

    raw = "where did i leave my keys"
    output = transformer.transform(raw, mode="polish")
    logger.info(f"Input:  {raw!r}\nOutput: {output!r}")

    lowered = output.lower()
    assert output.strip().endswith("?"), f"Dictated question was not preserved as a question: {output!r}"
    banned = ("what is your goal", "how can i help", "could you clarify", "i think",
              "you should", "try checking", "have you looked")
    for phrase in banned:
        assert phrase not in lowered, f"Model answered/solicited instead of cleaning: {phrase!r} found in {output!r}"

    logger.success(f"PASS: {raw!r} -> {output!r} (question preserved, no answer or counter-question).")


def test_llm_preserves_on_not_or() -> None:
    """Mission's Test Input 2: the exact word 'on' must survive verbatim, not
    mutate into 'or' (the reported hallucinated substitution)."""
    from src.config import load_config
    from src.transformer import Transformer

    logger.info("--- Testing Transformer: 'on' is never silently swapped for 'or' ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)
    if transformer.llm is None:
        logger.warning("No LLM weights on disk - skipping (nothing to test the guardrail against).")
        return

    raw = "he sat on the chair"
    output = transformer.transform(raw, mode="polish")
    logger.info(f"Input:  {raw!r}\nOutput: {output!r}")

    words = output.lower().replace(".", "").replace(",", "").split()
    assert "on" in words, f"'on' was dropped or mutated: output={output!r}"
    assert "or" not in words, f"'on' was hallucinated into 'or': output={output!r}"

    logger.success(f"PASS: {raw!r} -> {output!r} ('on' preserved exactly).")


def test_sanity_check_catches_conversational_leak() -> None:
    """Unit-test the Python-level guardrail directly, independent of the LLM -
    it must fire even if a future model/prompt regresses and starts leaking."""
    from src.transformer import sanity_check

    logger.info("--- Testing sanity_check(): conversational-leak phrases are caught ---")
    cases = [
        ("build me a login page", "What is your goal for this project?"),
        ("fix the bug in my code", "Sure, here is how I can help you with that."),
    ]
    for raw, bad_output in cases:
        reason = sanity_check(raw, bad_output)
        logger.info(f"raw={raw!r} bad_output={bad_output!r} -> reason={reason!r}")
        assert reason is not None, f"sanity_check failed to flag a conversational leak: {bad_output!r}"

    logger.success("PASS: sanity_check() flags conversational leaks the prompt alone might miss.")


def test_sanity_check_does_not_flag_legitimate_paraphrase() -> None:
    """Round 24 regression guard: an earlier, stricter version of sanity_check()
    flagged word-count deviation and dropped "guarded words" (on/or/and/...) -
    both were REMOVED after real production evidence they fired on completely
    correct output. Pin that removal down so it can't silently regress:
    "going to work" -> "will work" tripped the old guarded-word check (dropped
    "to"); "so wait what was I saying oh right um the deadline moved to friday"
    -> "The deadline moved to Friday." (a clean, correct filler removal)
    tripped the old length-ratio check. Neither should flag anything now."""
    from src.transformer import sanity_check

    logger.info("--- Testing sanity_check(): legitimate paraphrase/condensing is never flagged ---")
    cases = [
        ("honestly i am not sure this design is going to work for mobile users",
         "Honestly, I am not sure this design will work for mobile users."),
        ("so wait what was i saying oh right um the deadline moved to friday",
         "The deadline moved to Friday."),
        ("he sat on the chair", "He sat on the chair."),  # sanity: not flagged for an unrelated reason either
    ]
    for raw, out in cases:
        reason = sanity_check(raw, out)
        logger.info(f"{raw!r} -> {out!r} -> reason={reason!r}")
        assert reason is None, f"False positive on legitimate output: {reason!r} for {out!r}"

    logger.success("PASS: legitimate paraphrase and aggressive-but-correct filler removal are never flagged.")


def test_sanity_check_catches_essay_length_overgeneration() -> None:
    """The guarded-word/length-ratio removal traded away one real catch: a
    smaller model directly answering a short question with a full essay
    (found testing Turbo: "can you explain python pointers to me" -> a
    230-word technical explanation). ESSAY_CEILING_RATIO replaces the old,
    over-eager length check with a much looser one that only fires on
    genuine over-generation, not ordinary rephrasing."""
    from src.transformer import sanity_check

    logger.info("--- Testing sanity_check(): catastrophic essay-length over-generation is still caught ---")
    raw = "can you explain python pointers to me"
    essay = (
        "In Python, a pointer is not a direct reference to a memory location. "
        "Instead, variables hold references to objects, and assignment copies the reference, not the value. "
        "When you assign one variable to another, both now point to the same underlying object in memory. "
        "This becomes especially important with mutable objects like lists and dictionaries. "
        "If you modify a mutable object through one reference, the change is visible through every other reference to it. "
        "Immutable objects like integers and strings behave differently, since any 'modification' actually creates a new object. "
        "You can inspect an object's memory address using the built-in id() function for debugging purposes. "
        "Understanding this reference model is essential for avoiding subtle bugs involving shared mutable state. "
        "Many beginners are surprised the first time they encounter this behavior with lists passed into functions. "
        "In summary, Python's model trades explicit pointer arithmetic for automatic reference management."
    )
    reason = sanity_check(raw, essay)
    logger.info(f"reason={reason!r}")
    assert reason is not None, "sanity_check failed to flag essay-length over-generation."
    assert "essay" in reason, f"Unexpected reason (expected essay-ceiling, not another check): {reason!r}"

    logger.success(f"PASS: {len(essay.split())}-word answer to a {len(raw.split())}-word question was caught.")


def test_sanity_check_catches_repetition_loop() -> None:
    """New Round 24 check, replacing the removed length/vocabulary checks for
    one specific pathological case: the model looping on the same sentence
    instead of stopping."""
    from src.transformer import sanity_check

    logger.info("--- Testing sanity_check(): repetition loops are caught ---")
    looped = "This is not working. " * 8
    reason = sanity_check("this is not working", looped)
    logger.info(f"reason={reason!r}")
    assert reason is not None and "repetition" in reason, f"Repetition loop not caught: {reason!r}"

    logger.success(f"PASS: repetition loop caught -> {reason!r}")


def test_sanity_check_catches_answered_question() -> None:
    """Real defect found live-testing Turbo Flagship for this round's own
    test suite: "what is your goal for this quarter in terms of revenue
    targets" -> "Your goal for this quarter in terms of revenue targets is
    to achieve [specific target]." - a short (1.3x), non-essay answer with a
    hallucinated fill-in-the-blank placeholder, invisible to both the essay
    ceiling and every other check. Directly enforces the polish prompt's own
    rule 2 (a dictated question must survive as a cleaned question) in Python."""
    from src.transformer import sanity_check

    logger.info("--- Testing sanity_check(): a question answered instead of cleaned is caught ---")
    raw = "what is your goal for this quarter in terms of revenue targets"
    answered = "Your goal for this quarter in terms of revenue targets is to achieve [specific target]."
    reason = sanity_check(raw, answered)
    logger.info(f"reason={reason!r}")
    assert reason is not None and "answered" in reason, f"Answered question not caught: {reason!r}"

    # Must not false-positive on a correctly-cleaned multi-clause output where
    # the question mark isn't the very last character.
    ok = sanity_check("what is the capital of japan i forgot", "What is the capital of Japan? I forgot.")
    assert ok is None, f"False positive on a correct multi-clause question output: {ok!r}"

    # Must not false-positive on a legitimate self-correction that changes the sentence's shape.
    ok2 = sanity_check(
        "is it returning the dictionary wait no the list is it fine now actually never mind",
        "Never mind.",
    )
    assert ok2 is None, f"False positive on a legitimate self-correction: {ok2!r}"

    logger.success(f"PASS: answered question caught ({reason!r}); multi-clause and self-correction cases unaffected.")


def test_sanity_check_exempts_restructuring_modes() -> None:
    """Regression guard: bullets/prompt_engineer intentionally condense prose
    into a list or expand a short instruction into a structured spec - the
    essay-ceiling check must not fire there (mode-gated to polish/code only),
    only the universal conversational-leak check should."""
    from src.transformer import sanity_check

    logger.info("--- Testing sanity_check(): bullets/prompt_engineer are exempt from the essay ceiling ---")
    raw = "fix the login bug and update the docs"
    long_expansion = (
        "Task: Fix the login bug and update the documentation. "
        "Context: Users are reporting an intermittent failure during authentication, and the "
        "related documentation has fallen out of date with the current login flow. "
        "Instructions: Reproduce the login bug, identify the root cause, apply a fix, and verify "
        "it resolves the reported failures. Once fixed, update the documentation to accurately "
        "describe the corrected authentication flow. "
        "Constraints: Do not change the public API surface of the authentication module."
    )

    reason_polish = sanity_check(raw, long_expansion, mode="polish")
    reason_prompt_engineer = sanity_check(raw, long_expansion, mode="prompt_engineer")
    logger.info(f"mode=polish -> {reason_polish!r}; mode=prompt_engineer -> {reason_prompt_engineer!r}")
    assert reason_polish is not None, "Sanity check should still flag essay-length over-generation under strict polish mode."
    assert reason_prompt_engineer is None, "Sanity check false-positived on legitimate prompt_engineer-mode expansion."

    # Leak detection must still apply regardless of mode.
    leak_reason = sanity_check(raw, "Sure, here is your restructured prompt:", mode="prompt_engineer")
    assert leak_reason is not None, "Conversational leak check should apply to every mode, not just polish."

    logger.success("PASS: the essay ceiling is polish/code-only; leak detection stays universal.")


def test_basic_capitalize_fallback() -> None:
    """Unit-test the raw-transcript fallback formatter used when sanity_check() fires."""
    from src.transformer import basic_capitalize

    logger.info("--- Testing basic_capitalize(): fallback formatting ---")
    result = basic_capitalize("he sat on the chair. then he left")
    logger.info(f"result={result!r}")
    assert result[0].isupper(), f"First letter not capitalized: {result!r}"
    assert "He sat" in result and "Then he left" in result, f"Sentence starts not capitalized: {result!r}"

    logger.success(f"PASS: basic_capitalize() -> {result!r}")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO CONVERSATIONAL GUARDRAIL TEST SUITE")
    logger.info("=========================================================")
    test_sanity_check_catches_conversational_leak()
    test_sanity_check_does_not_flag_legitimate_paraphrase()
    test_sanity_check_catches_essay_length_overgeneration()
    test_sanity_check_catches_repetition_loop()
    test_sanity_check_catches_answered_question()
    test_sanity_check_exempts_restructuring_modes()
    test_basic_capitalize_fallback()
    test_llm_does_not_answer_or_solicit()
    test_llm_preserves_on_not_or()
    logger.success("=========================================================")
    logger.success(" ALL CONVERSATIONAL GUARDRAIL TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
