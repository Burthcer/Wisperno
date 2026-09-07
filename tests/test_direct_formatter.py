"""
Instant Direct Dictation formatter verification for Wisperno.
Unit-tests src/direct_formatter.py directly - pure regex/string logic, no
Whisper or LLM involved, so this runs instantly with no models on disk.

Run: python tests/test_direct_formatter.py
"""

import sys
import time
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_filler_and_stutter_stripping() -> None:
    from src.direct_formatter import format_direct

    logger.info("--- Testing filler-word stripping ---")
    result = format_direct("um I think we should uh go with option A")
    logger.info(f"-> {result!r}")
    assert result == "I think we should go with option A.", result
    logger.success("PASS: fillers stripped, capitalization/terminal punctuation applied.")


def test_numbered_list_detection() -> None:
    from src.direct_formatter import format_direct

    logger.info("--- Testing spoken numbered-list detection ---")
    result = format_direct("we need three things first buy milk second get eggs third go home")
    logger.info(f"-> {result!r}")
    lines = result.split("\n")
    assert lines == ["1. Buy milk.", "2. Get eggs.", "3. Go home."], result
    logger.success(f"PASS: 3-line numbered list -> {lines}")


def test_bullet_list_detection() -> None:
    from src.direct_formatter import format_direct

    logger.info("--- Testing spoken bullet-list detection ---")
    result = format_direct("bullet point buy milk bullet point get eggs")
    logger.info(f"-> {result!r}")
    assert result.split("\n") == ["* Buy milk.", "* Get eggs."], result
    logger.success("PASS: bullet cues formatted as a clean bullet list.")


def test_single_ordinal_word_is_not_a_false_positive() -> None:
    """A single incidental 'first'/'dash' in normal speech must not be
    mistaken for a spoken enumeration - only 2+ cues count as a real list."""
    from src.direct_formatter import format_direct

    logger.info("--- Testing single-cue false-positive guard ---")
    result = format_direct("I first went to the store")
    logger.info(f"-> {result!r}")
    assert "\n" not in result and result == "I first went to the store.", result
    logger.success("PASS: a single ordinal word stays a normal sentence, not a 1-item list.")


def test_spoken_slash_commands() -> None:
    from src.direct_formatter import format_direct

    logger.info("--- Testing spoken slash-command shorthand ---")
    assert format_direct("slash compact") == "/compact"
    assert format_direct("forward slash help") == "/help"
    result = format_direct("run slash help to see options")
    logger.info(f"-> {result!r}")
    assert result == "Run /help to see options.", result
    logger.success("PASS: 'slash <word>' / 'forward slash <word>' -> '/<word>'.")


def test_spoken_punctuation_shorthand() -> None:
    from src.direct_formatter import format_direct

    logger.info("--- Testing dash/underscore/backtick shorthand ---")
    result = format_direct("my project dash file underscore name")
    logger.info(f"-> {result!r}")
    assert result == "My project-file_name.", result
    logger.success("PASS: 'dash'/'hyphen' -> '-', 'underscore' -> '_'.")


def test_repeated_word_stutter_collapse() -> None:
    from src.direct_formatter import format_direct

    logger.info("--- Testing duplicated-word stutter collapse ---")
    assert format_direct("the the client wants changes") == "The client wants changes."
    result = format_direct("we're all done here etc. etc.")
    logger.info(f"-> {result!r}")
    assert "etc. etc." not in result, result
    # Comma-separated repetition is intentional emphasis, not a stutter - must survive.
    assert format_direct("great, great job team") == "Great, great job team."
    logger.success("PASS: accidental word/phrase stutters collapsed; comma-separated emphasis preserved.")


def test_performance_under_10ms() -> None:
    from src.direct_formatter import format_direct

    logger.info("--- Testing direct-formatting latency ---")
    sample = "um so uh I was thinking we could uh go with the the second option"
    iterations = 200
    t0 = time.perf_counter()
    for _ in range(iterations):
        format_direct(sample)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / iterations
    logger.info(f"Average: {elapsed_ms:.4f}ms/call over {iterations} calls.")
    assert elapsed_ms < 10.0, f"format_direct averaged {elapsed_ms:.4f}ms - expected <10ms"
    logger.success(f"PASS: {elapsed_ms:.4f}ms/call, well under the 10ms ceiling.")


def test_lowercase_i_and_contractions_always_capitalized() -> None:
    """Grammar invariant: standalone 'i' and its contractions ('i'm', 'i've',
    'i'll', 'i'd') must always come out capitalized, anywhere in the sentence -
    not just at the start, which basic sentence-capitalization already covers."""
    from src.direct_formatter import format_direct

    logger.info("--- Testing 'I' / contraction capitalization invariant ---")
    cases = [
        ("i think i'm going to head out", "I think I'm going to head out."),
        ("she said i've already told her and i'll call back", "She said I've already told her and I'll call back."),
        ("i'd rather wait if i'm being honest", "I'd rather wait if I'm being honest."),
    ]
    for raw, expected in cases:
        result = format_direct(raw)
        logger.info(f"{raw!r} -> {result!r}")
        assert result == expected, f"Expected {expected!r}, got {result!r}"
    logger.success("PASS: 'I' and its contractions are capitalized everywhere in the sentence.")


def test_inference_worker_skips_llm_when_direct_formatter_enabled() -> None:
    """
    Integration check for the actual wiring in src/workers.py: with
    use_direct_formatter=True, InferenceWorker.run() must inject
    format_direct()'s output and must NEVER call transformer.transform() -
    the whole point of Mode 1 is that llama.cpp is never invoked.
    """
    from unittest.mock import MagicMock
    from src.workers import InferenceWorker
    from src.direct_formatter import format_direct

    logger.info("--- Testing InferenceWorker: use_direct_formatter bypasses the LLM entirely ---")
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "um I think we should uh go with option A"
    transformer = MagicMock()
    transformer.profanity_filter = "allow"
    injector = MagicMock()
    injector.inject_text.return_value = True
    db = MagicMock()

    worker = InferenceWorker(
        audio_array=object(), transcriber=transcriber, transformer=transformer, injector=injector,
        db=db, snippets={}, mode="polish", record_duration_ms=1200.0, use_direct_formatter=True,
    )
    worker.run()

    transformer.transform.assert_not_called()
    transformer.transform_with_prompt.assert_not_called()
    injector.inject_text.assert_called_once_with(format_direct("um I think we should uh go with option A"))
    logger.success(f"PASS: injected {injector.inject_text.call_args[0][0]!r} with zero LLM calls.")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO DIRECT FORMATTER (INSTANT DICTATION) TEST SUITE")
    logger.info("=========================================================")
    test_filler_and_stutter_stripping()
    test_numbered_list_detection()
    test_bullet_list_detection()
    test_single_ordinal_word_is_not_a_false_positive()
    test_lowercase_i_and_contractions_always_capitalized()
    test_spoken_slash_commands()
    test_spoken_punctuation_shorthand()
    test_repeated_word_stutter_collapse()
    test_performance_under_10ms()
    test_inference_worker_skips_llm_when_direct_formatter_enabled()
    logger.success("=========================================================")
    logger.success(" ALL DIRECT FORMATTER TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
