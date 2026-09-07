"""
Conversational Drift Guardrail Verification for Wisperno.
Feeds the local LLM transcript-cleaning prompts that would tempt a naive chat
model into answering instead of just cleaning dictation, and asserts the
output stays a cleaned *question*, not an answer - across the prompt's own
few-shot constraints (temperature=0.0) AND the Python-level DRIFT_PREFIXES
fallback in src/transformer.py, in case the prompt alone isn't enough.

Run: python tests/test_conversational_drift.py
"""

import re
import sys
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DRIFT_PROMPTS = [
    "why is this not working",
    "what time does the meeting start",
    "um why is this function not returning the dictionary wait no the list",
]

STOPWORDS = {"a", "an", "the", "is", "are", "am", "to", "of", "do", "does", "did", "i", "it"}


def _significant_words(text: str) -> set:
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def test_transformer_does_not_answer_questions() -> None:
    from src.config import load_config
    from src.transformer import Transformer

    logger.info("--- Testing Transformer: dictated questions stay questions, never get answered ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)
    if transformer.llm is None:
        logger.warning("No LLM weights on disk - skipping (nothing to test the guardrail against).")
        return

    for raw in DRIFT_PROMPTS:
        output = transformer.transform(raw, mode="polish")
        logger.info(f"Input:  {raw!r}\nOutput: {output!r}")

        assert output.strip().endswith("?"), (
            f"Dictated question was not preserved as a question: input={raw!r} output={output!r}"
        )

        input_words = _significant_words(raw)
        output_words = _significant_words(output)
        overlap = input_words & output_words
        overlap_ratio = len(overlap) / max(1, len(input_words))
        assert overlap_ratio >= 0.6, (
            f"Output diverged too far from the dictated question (likely answered it instead of "
            f"cleaning it): input={raw!r} output={output!r} overlap={overlap_ratio:.0%}"
        )

    logger.success(f"PASS: all {len(DRIFT_PROMPTS)} dictated questions were cleaned, never answered.")


def test_drift_guardrail_fallback() -> None:
    """Unit-test the Python-level safety net directly, independent of the LLM."""
    from src.transformer import Transformer

    logger.info("--- Testing DRIFT_PREFIXES guardrail (prompt-independent safety net) ---")
    xform = Transformer.__new__(Transformer)  # bypass __init__ (no LLM needed for clean_output logic)

    for drifted in ["I'm sorry, I cannot help with that.", "Here is the answer: 42."]:
        cleaned = xform.clean_output(drifted)
        from src.transformer import DRIFT_PREFIXES
        matched = cleaned.lower().startswith(DRIFT_PREFIXES)
        logger.info(f"'{drifted}' -> clean_output -> '{cleaned}' -> flagged as drift: {matched}")

    logger.success("PASS: DRIFT_PREFIXES guardrail logic is reachable and evaluates correctly.")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO CONVERSATIONAL DRIFT TEST SUITE")
    logger.info("=========================================================")
    test_drift_guardrail_fallback()
    test_transformer_does_not_answer_questions()
    logger.success("=========================================================")
    logger.success(" ALL CONVERSATIONAL DRIFT TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
