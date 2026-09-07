"""
Selection Polish / Transform pipeline verification for Wisperno.

Drives the REAL SelectionPolishWorker.run() (src/workers.py) with the REAL
Transformer (actual GGUF model loaded into VRAM) and a REAL (temp-file)
WispernoDB - only src/injector.py's OS-level clipboard/keyboard calls are
stubbed, since "a selection exists in some foreground window" and "a Ctrl+V
lands in that window" aren't things a headless test can create. This matches
the stubbing convention already used for AudioRecorder in
tests/test_live_transcription.py: stub the OS boundary, run the real pipeline
code on both sides of it.

Run: python tests/test_selection_polish.py
"""

import sys
import time
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

BAD_GRAMMAR_TEXT = "this are bad grammer and speling"


class _StubInjector:
    """Stands in for TextInjector: copy_selection() returns canned 'selected'
    text instead of touching the real clipboard/keyboard, inject_text()
    records what it was asked to paste instead of actually pasting it."""

    def __init__(self, selection):
        self._selection = selection
        self.injected = None

    def copy_selection(self):
        return self._selection

    def inject_text(self, text: str) -> bool:
        self.injected = text
        return True


def _run_worker(injector, transformer, db, mode="polish", system_prompt=None):
    from src.workers import SelectionPolishWorker

    states = []
    results = []
    errors = []
    worker = SelectionPolishWorker(injector, transformer, db, mode=mode, system_prompt=system_prompt)
    worker.state_changed.connect(states.append)
    worker.result_ready.connect(results.append)
    worker.error.connect(errors.append)
    worker.run()  # synchronous - same technique test_live_transcription.py uses for QThread.run()
    return states, results, errors


def test_selection_polish_end_to_end() -> None:
    from src.config import load_config
    from src.database import WispernoDB
    from src.transformer import Transformer

    logger.info("--- Testing SelectionPolishWorker: real transform + inject + history insert ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)
    if transformer.llm is None:
        logger.warning("No LLM weights on disk - skipping (nothing to test the pipeline against).")
        return

    import os
    import tempfile
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db_path)
    try:
        db = WispernoDB(db_path)
        injector = _StubInjector(BAD_GRAMMAR_TEXT)
        # DEFAULT_TRANSFORMS's real "Polish" system_prompt (src/database.py), the
        # same one a real alt+c tap-with-selection would use via
        # engine.py::_on_transform_release -> SelectionPolishWorker.
        polish_transform = db.get_transform("polish")
        assert polish_transform, "DEFAULT_TRANSFORMS did not seed a 'polish' transform"

        states, results, errors = _run_worker(
            injector, transformer, db, mode="polish", system_prompt=polish_transform["system_prompt"],
        )

        assert not errors, f"SelectionPolishWorker raised: {errors}"
        assert states[0] == "selection_processing", f"Did not enter processing state first: {states}"
        assert states[-1] == "selection_success", f"Did not reach success state: {states}"
        assert injector.injected, "Nothing was injected back - the paste step produced empty output"
        assert injector.injected.strip() != BAD_GRAMMAR_TEXT, "Output is identical to raw input - LLM did not run"
        logger.info(f"Input:  {BAD_GRAMMAR_TEXT!r}\nOutput: {injector.injected!r}")

        assert len(results) == 1
        assert results[0]["raw_transcript"] == BAD_GRAMMAR_TEXT
        assert results[0]["polished_transcript"] == injector.injected
        assert results[0]["mode_used"] == "text_polish"

        rows = db.list_history()
        assert len(rows) == 1 and rows[0]["source"] == "text_polish", "History row was not written for the polish"

        db.close()
        logger.success(f"PASS: selection polish transformed and injected real cleaned text: {injector.injected!r}")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_no_selection_exits_cleanly_without_saving() -> None:
    """Directive's 'No text selected' case: copy_selection() returning nothing
    (nothing highlighted, or the 300ms clipboard poll in injector.py timed
    out) must not throw, must not paste, and must not write a history row."""
    from src.config import load_config
    from src.transformer import Transformer

    logger.info("--- Testing SelectionPolishWorker: no selection -> clean no-op ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)

    injector = _StubInjector(None)
    states, results, errors = _run_worker(injector, transformer, None, mode="polish", system_prompt="irrelevant")

    assert not errors, f"SelectionPolishWorker raised on empty selection: {errors}"
    assert states == ["selection_processing", "idle"], f"Unexpected state sequence: {states}"
    assert not results, "result_ready fired despite nothing being selected"
    assert injector.injected is None, "inject_text() was called despite nothing being selected"
    logger.success("PASS: no selection exits cleanly with no paste and no saved result.")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO SELECTION POLISH TEST SUITE")
    logger.info("=========================================================")
    test_selection_polish_end_to_end()
    test_no_selection_exits_cleanly_without_saving()
    logger.success("=========================================================")
    logger.success(" ALL SELECTION POLISH TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
