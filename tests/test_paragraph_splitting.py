"""
Fast, mocked companion to tests/test_paragraph_pauses.py (which proves the
same behavior end-to-end with real audio through the real Whisper model).
This one stubs Whisper's output directly so it runs in milliseconds, no GPU
model load required - useful as a quick regression check.

Note: the mock below provides WORD-level timestamps, not segment-level ones.
Whisper's SEGMENT timestamps are contiguous (one segment's start always
equals the previous segment's end) even across a real multi-second silence -
verified empirically before this file was written - so a mock built only
from segment start/end would not exercise the real code path at all. Only
word-level timestamps actually carry pause information, which is exactly
what src/transcriber.py's _run_transcription() keys off.

Run: python -m pytest tests/test_paragraph_splitting.py
     python tests/test_paragraph_splitting.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


class _StubModel:
    """Stands in for faster-whisper's WhisperModel - returns pre-built
    (segments, info) instead of running real inference."""

    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, *args, **kwargs):
        return self._segments, SimpleNamespace()


def _word(text, start, end):
    return SimpleNamespace(word=text, start=start, end=end)


def _make_transcriber(segments) -> "Transcriber":
    """Builds a real Transcriber instance without loading the real GPU model -
    __new__ bypasses __init__ (which always calls _load_model()), then wires
    just the two attributes _run_transcription() actually reads."""
    from src.config import WhisperConfig
    from src.transcriber import Transcriber

    t = Transcriber.__new__(Transcriber)
    t.config = WhisperConfig()
    t.model = _StubModel(segments)
    return t


def test_mocked_1_7s_gap_produces_paragraph_break() -> None:
    logger.info("--- Testing _run_transcription(): mocked 1.7s word-level gap -> '\\n\\n' ---")
    from src.transcriber import Transcriber

    # Segment 1: "I was talking about this project" (0.0 - 2.5)
    # Segment 2: "then I remembered something else" (4.2 - 6.0) -> gap = 1.7s
    seg1_words = [
        _word(" I", 0.0, 0.3), _word(" was", 0.3, 0.6), _word(" talking", 0.6, 1.0),
        _word(" about", 1.0, 1.4), _word(" this", 1.4, 1.8), _word(" project", 1.8, 2.5),
    ]
    seg2_words = [
        _word(" then", 4.2, 4.5), _word(" I", 4.5, 4.7), _word(" remembered", 4.7, 5.2),
        _word(" something", 5.2, 5.7), _word(" else", 5.7, 6.0),
    ]
    segments = [
        SimpleNamespace(words=seg1_words, text="".join(w.word for w in seg1_words)),
        SimpleNamespace(words=seg2_words, text="".join(w.word for w in seg2_words)),
    ]

    t = _make_transcriber(segments)
    text, _info = Transcriber._run_transcription(t, None, "", vad_filter=False, vad_params=None)

    assert "\n\n" in text, f"No paragraph break for a 1.7s word-level gap: {text!r}"
    first, _, second = text.partition("\n\n")
    assert first.strip() == "I was talking about this project", first
    assert second.strip() == "then I remembered something else", second
    logger.success(f"PASS: mocked 1.7s gap -> paragraph break: {text!r}")

    from src.direct_formatter import format_direct
    formatted = format_direct(text)
    assert formatted == "I was talking about this project.\n\nThen I remembered something else.", formatted
    logger.success(f"PASS: format_direct() capitalizes the paragraph after the break: {formatted!r}")


def test_mocked_short_gap_does_not_split() -> None:
    logger.info("--- Testing _run_transcription(): a 0.5s word-level gap stays one paragraph ---")
    from src.transcriber import Transcriber

    words = [
        _word(" Hello", 0.0, 0.4), _word(" there", 0.4, 0.7),
        _word(" friend", 1.2, 1.6),  # gap from 0.7 -> 1.2 is 0.5s, well under the 1.3s threshold
    ]
    segments = [SimpleNamespace(words=words, text="".join(w.word for w in words))]

    t = _make_transcriber(segments)
    text, _info = Transcriber._run_transcription(t, None, "", vad_filter=False, vad_params=None)

    assert "\n\n" not in text, f"A 0.5s gap incorrectly produced a paragraph break: {text!r}"
    assert text.strip() == "Hello there friend", text
    logger.success(f"PASS: a 0.5s gap stayed within one paragraph: {text!r}")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO PARAGRAPH-SPLITTING TEST SUITE (mocked)")
    logger.info("=========================================================")
    test_mocked_1_7s_gap_produces_paragraph_break()
    test_mocked_short_gap_does_not_split()
    logger.success("=========================================================")
    logger.success(" ALL MOCKED PARAGRAPH-SPLITTING TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
