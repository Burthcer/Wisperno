"""
Wispr Flow-style paragraph splitting verification for Wisperno.

Real SAPI-synthesized speech (same technique as tests/test_audio_chunking.py,
whose synth helpers are reused directly) fed through the REAL Transcriber, to
prove src/transcriber.py's pause-based "\n\n" join actually fires on a real
mid-dictation pause and does NOT fire on a normal short breath/comma pause -
plus a direct check that src/direct_formatter.py capitalizes each resulting
paragraph independently instead of collapsing the break.

Run: python tests/test_paragraph_pauses.py
"""

import sys
from pathlib import Path

import numpy as np
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tests.test_audio_chunking import _synthesize, _load_wav_as_float32, _resample_to_16k  # noqa: E402

FIRST_THOUGHT = "This is the first thought in the dictation."
SECOND_THOUGHT = "This is a completely separate second thought after a real pause."


def _build_two_thought_recording(gap_sec: float) -> np.ndarray:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        pieces = []
        for i, text in enumerate((FIRST_THOUGHT, SECOND_THOUGHT)):
            wav_path = Path(tmp) / f"thought_{i}.wav"
            _synthesize(text, wav_path)
            sr, audio = _load_wav_as_float32(wav_path)
            pieces.append(_resample_to_16k(audio, sr))
            if i == 0:
                pieces.append(np.zeros(int(gap_sec * 16000), dtype=np.float32))
        return np.concatenate(pieces)


def test_long_pause_inserts_paragraph_break() -> None:
    from src.config import load_config
    from src.transcriber import Transcriber
    from src.vocabulary import Vocabulary

    logger.info("--- Testing Transcriber: a real ~1.8s pause starts a new paragraph ---")
    config = load_config()
    transcriber = Transcriber(config=config.whisper, vocabulary=Vocabulary(db=None))

    audio = _build_two_thought_recording(gap_sec=1.8)
    text = transcriber.transcribe(audio)
    logger.info(f"Transcript: {text!r}")

    assert "\n\n" in text, f"No paragraph break inserted across a 1.8s pause: {text!r}"
    first, _, second = text.partition("\n\n")
    assert "first thought" in first.lower(), f"First paragraph missing expected content: {first!r}"
    assert "second thought" in second.lower(), f"Second paragraph missing expected content: {second!r}"
    logger.success("PASS: a real 1.8s mid-dictation pause produced a '\\n\\n' paragraph break.")


def test_short_pause_stays_one_paragraph() -> None:
    """Regression guard: an ordinary short breath/comma-length pause between
    sentences must NOT fragment into spurious paragraphs - only pauses at or
    above PARAGRAPH_PAUSE_SEC should."""
    from src.config import load_config
    from src.transcriber import Transcriber
    from src.vocabulary import Vocabulary

    logger.info("--- Testing Transcriber: a short ~0.4s pause does NOT start a new paragraph ---")
    config = load_config()
    transcriber = Transcriber(config=config.whisper, vocabulary=Vocabulary(db=None))

    audio = _build_two_thought_recording(gap_sec=0.4)
    text = transcriber.transcribe(audio)
    logger.info(f"Transcript: {text!r}")

    assert "\n\n" not in text, f"A short 0.4s pause incorrectly produced a paragraph break: {text!r}"
    logger.success("PASS: a short 0.4s pause stayed within one paragraph.")


def test_direct_formatter_capitalizes_each_paragraph() -> None:
    from src.direct_formatter import format_direct

    logger.info("--- Testing format_direct(): each paragraph capitalized/punctuated independently ---")
    raw = "this is the opening thought\n\nthis is a completely separate thought"
    result = format_direct(raw)
    assert result == "This is the opening thought.\n\nThis is a completely separate thought.", result
    logger.success(f"PASS: paragraphs formatted independently: {result!r}")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO PARAGRAPH-PAUSE SPLITTING TEST SUITE")
    logger.info("=========================================================")
    test_direct_formatter_capitalizes_each_paragraph()
    test_long_pause_inserts_paragraph_break()
    test_short_pause_stays_one_paragraph()
    logger.success("=========================================================")
    logger.success(" ALL PARAGRAPH-PAUSE TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
