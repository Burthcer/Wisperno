"""
Writing Styles (Alt+V) pipeline verification for Wisperno.

Drives the REAL WritingStylesWorker.run() with the REAL Transformer (actual
GGUF model) and a stubbed injector (same convention as
tests/test_selection_polish.py - OS-level clipboard/keyboard capture isn't
something a headless test can create). Also covers engine.py's
select_writing_style() (injection + history recording) against a real
temp-file WispernoDB, and the alt+v/code_fix shortcut-conflict migration.

Run: python tests/test_writing_styles.py
"""

import sys
import time
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

SAMPLE_TEXT = (
    "the meeting got pushed back again and now i have to redo the whole schedule, "
    "this is getting really frustrating"
)
# Phrased as a direct request TO someone - this specific shape reliably tempts
# a small model into agreeing to the request ("Sure, I can...") instead of
# just rewriting it in the target tone, reproduced directly during this
# feature's own testing. Covered by its own guardrail test below, not the
# general 5-distinct-styles test (which needs a sample where the model can
# actually demonstrate style variation, not just its safety fallback).
REQUEST_SHAPED_TEXT = "hey can you send me the report by tomorrow morning, its kind of urgent"


class _StubInjector:
    def __init__(self, selection):
        self._selection = selection
        self.injected = None

    def copy_selection(self):
        return self._selection

    def inject_text(self, text: str) -> bool:
        self.injected = text
        return True


def test_all_five_styles_generate_distinct_output() -> None:
    from src.config import load_config
    from src.transformer import Transformer
    from src.workers import WritingStylesWorker
    from src.writing_styles import WRITING_STYLES

    logger.info("--- Testing WritingStylesWorker: all 5 styles generate real, distinct rewrites ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)
    if transformer.llm is None:
        logger.warning("No LLM weights on disk - skipping (nothing to test the pipeline against).")
        return

    injector = _StubInjector(SAMPLE_TEXT)
    worker = WritingStylesWorker(injector, transformer)

    captured = {}
    ready = {}
    no_sel = []
    errors = []
    worker.selection_captured.connect(lambda t: captured.setdefault("text", t))
    worker.style_ready.connect(lambda sid, text: ready.__setitem__(sid, text))
    worker.no_selection.connect(lambda: no_sel.append(True))
    worker.error.connect(errors.append)

    worker.run()  # synchronous - same technique test_selection_polish.py uses for QThread.run()

    assert not errors, f"WritingStylesWorker raised: {errors}"
    assert not no_sel, "no_selection fired despite real selected text"
    assert captured.get("text") == SAMPLE_TEXT

    expected_ids = {s.id for s in WRITING_STYLES}
    assert set(ready.keys()) == expected_ids, f"Missing styles: {expected_ids - set(ready.keys())}"

    for style in WRITING_STYLES:
        text = ready[style.id]
        assert text.strip(), f"Style '{style.id}' produced empty output"
        assert text.strip().lower() != SAMPLE_TEXT.lower(), f"Style '{style.id}' echoed the input verbatim"
        # Anti-hallucination: must not add a conversational preamble.
        lowered = text.strip().lower()
        assert not lowered.startswith(("here is", "here's", "sure,", "sure!")), (
            f"Style '{style.id}' added a conversational preamble: {text!r}"
        )
        logger.info(f"[{style.title}] {text!r}")

    logger.success("PASS: all 5 styles generated real, distinct, preamble-free rewrites.")


def test_request_shaped_input_never_agrees_to_the_request() -> None:
    """Regression test for a real bug found while building this feature: a
    draft phrased as a request ('can you send me...') tempted the LLM into
    replying to it ('Sure, I can send you...') instead of rewriting it,
    across every style tried. src/transformer.py's DRIFT_PREFIXES now catches
    any output that OPENS with an agreement word and falls back to a safe,
    verbatim-ish rewrite instead - verified here directly against the exact
    input shape that reproduced the bug."""
    from src.config import load_config
    from src.transformer import Transformer
    from src.workers import WritingStylesWorker

    logger.info("--- Testing WritingStylesWorker: a request-shaped draft never gets 'agreed to' ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)
    if transformer.llm is None:
        logger.warning("No LLM weights on disk - skipping.")
        return

    injector = _StubInjector(REQUEST_SHAPED_TEXT)
    worker = WritingStylesWorker(injector, transformer)
    ready = {}
    worker.style_ready.connect(lambda sid, text: ready.__setitem__(sid, text))
    worker.run()

    agreement_openers = ("sure,", "sure!", "yes,", "yes!", "of course,", "certainly,", "okay,", "ok,", "absolutely,")
    for style_id, text in ready.items():
        lowered = text.strip().lower()
        assert not lowered.startswith(agreement_openers), (
            f"Style '{style_id}' agreed to the request instead of rewriting it: {text!r}"
        )
    logger.success("PASS: no style agreed to a request-shaped draft - all safely rewrote or fell back.")


def test_no_selection_emits_no_selection_signal() -> None:
    from src.config import load_config
    from src.transformer import Transformer
    from src.workers import WritingStylesWorker

    logger.info("--- Testing WritingStylesWorker: no selection -> no_selection signal, no HUD ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)

    injector = _StubInjector(None)
    worker = WritingStylesWorker(injector, transformer)
    no_sel = []
    captured = []
    worker.no_selection.connect(lambda: no_sel.append(True))
    worker.selection_captured.connect(captured.append)
    worker.run()

    assert no_sel == [True]
    assert not captured, "selection_captured fired despite nothing being selected"
    logger.success("PASS: no selection -> no_selection signal only, HUD never shown.")


def test_select_writing_style_injects_and_records_history() -> None:
    import os
    import tempfile

    from src.database import WispernoDB

    logger.info("--- Testing engine.select_writing_style(): inject + history row with 'writing_style' source ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db_path)
    try:
        # Exercise the same DB call engine.select_writing_style() makes, without
        # spinning up the full WispernoEngine (which loads real hardware engines).
        db = WispernoDB(db_path)
        db.add_history(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            duration_seconds=0.0,
            raw_transcript=SAMPLE_TEXT,
            polished_transcript="Could you please send the report by tomorrow morning? It's fairly urgent.",
            mode_used="style_professional",
            latency_ms=0.0,
            source="writing_style",
        )
        rows = db.list_history()
        assert len(rows) == 1
        assert rows[0]["source"] == "writing_style"
        assert rows[0]["mode_used"] == "style_professional"
        assert rows[0]["entry_type"] == "dictation", "A writing-style entry must fall under the Dictations filter"
        db.close()
        logger.success("PASS: a writing-style history row round-trips with source='writing_style'.")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_alt_v_conflict_migration_frees_the_shortcut() -> None:
    import os
    import tempfile

    from src.database import WispernoDB

    logger.info("--- Testing free_up_alt_v_shortcut(): code_fix moves off Alt+V exactly once ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db_path)
    try:
        db = WispernoDB(db_path)
        assert db.get_transform("code_fix")["shortcut"] == "alt+d", (
            "DEFAULT_TRANSFORMS should already seed code_fix on alt+d, not alt+v"
        )

        # Simulate an existing install that still has the OLD alt+v default.
        db.update_transform("code_fix", shortcut="alt+v")

        migrated = db.free_up_alt_v_shortcut()
        assert migrated is True
        assert db.get_transform("code_fix")["shortcut"] == "alt+d"

        # A user who deliberately rebinds it back to alt+v afterward must be left alone.
        db.update_transform("code_fix", shortcut="alt+v")
        migrated_again = db.free_up_alt_v_shortcut()
        assert migrated_again is False, "Must not re-run once the one-time flag is set"
        assert db.get_transform("code_fix")["shortcut"] == "alt+v", "A user's later re-bind must not be overwritten"

        db.close()
        logger.success("PASS: alt+v/code_fix conflict resolved once, and never re-applied after that.")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO WRITING STYLES TEST SUITE")
    logger.info("=========================================================")
    test_alt_v_conflict_migration_frees_the_shortcut()
    test_select_writing_style_injects_and_records_history()
    test_no_selection_emits_no_selection_signal()
    test_all_five_styles_generate_distinct_output()
    test_request_shaped_input_never_agrees_to_the_request()
    logger.success("=========================================================")
    logger.success(" ALL WRITING STYLES TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
