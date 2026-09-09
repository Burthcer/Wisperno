"""
Verification for the new General/Audio settings added this round: Always-Keep-
Pill-on-Top, Audio Cues, and Auto-Copy-to-Clipboard.

Run: python tests/test_new_settings.py
"""

import sys
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_new_config_fields_round_trip() -> None:
    import tempfile

    from src.config import load_config, save_config

    logger.info("--- Testing new config fields survive a save/load round-trip ---")
    config = load_config()
    config.pill_always_on_top = False
    config.audio_cues_enabled = True
    config.injector.auto_copy_to_clipboard = True

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        save_config(config, str(path))
        reloaded = load_config(str(path))

    assert reloaded.pill_always_on_top is False
    assert reloaded.audio_cues_enabled is True
    assert reloaded.injector.auto_copy_to_clipboard is True
    logger.success("PASS: pill_always_on_top / audio_cues_enabled / auto_copy_to_clipboard all round-trip.")


def test_auto_copy_to_clipboard_skips_restoring_old_clipboard() -> None:
    import pyperclip

    from src.config import InjectorConfig
    from src.injector import TextInjector

    logger.info("--- Testing TextInjector: auto_copy_to_clipboard leaves the injected text on the clipboard ---")
    old_clip = None
    try:
        old_clip = pyperclip.paste()
    except Exception:
        pass
    try:
        pyperclip.copy("sentinel-old-clipboard-value")

        cfg = InjectorConfig(restore_clipboard=True, auto_copy_to_clipboard=True)
        injector = TextInjector(config=cfg)
        injector.inject_text("the injected text")
        # Regardless of whether a foreground window actually received the
        # paste keystroke in this sandboxed environment, the clipboard itself
        # must now hold the injected text, not the restored sentinel.
        assert pyperclip.paste() == "the injected text", (
            f"auto_copy_to_clipboard did not leave the injected text on the clipboard: {pyperclip.paste()!r}"
        )
        logger.success("PASS: with auto_copy_to_clipboard on, the clipboard keeps the injected text.")

        pyperclip.copy("sentinel-old-clipboard-value-2")
        cfg2 = InjectorConfig(restore_clipboard=True, auto_copy_to_clipboard=False)
        injector2 = TextInjector(config=cfg2)
        injector2.inject_text("other injected text")
        assert pyperclip.paste() == "sentinel-old-clipboard-value-2", (
            "restore_clipboard (auto_copy off) should have restored the prior clipboard value"
        )
        logger.success("PASS: with auto_copy_to_clipboard off, restore_clipboard's own behavior is unchanged.")
    finally:
        if old_clip is not None:
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass


def test_pill_always_on_top_toggle_does_not_crash() -> None:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from src.ui.floating_pill import FloatingPill

    logger.info("--- Testing FloatingPill.set_always_on_top(): toggles the window flag cleanly ---")
    app = QApplication.instance() or QApplication([])
    pill = FloatingPill(always_on_top=True)
    pill.set_always_on_top(False)
    pill.set_always_on_top(True)
    logger.success("PASS: set_always_on_top() toggled both ways without raising.")


def test_audio_cues_do_not_raise() -> None:
    from src import audio_cues

    logger.info("--- Testing audio_cues: start/stop cues fire without raising ---")
    audio_cues.play_start_cue()
    audio_cues.play_stop_cue()
    import time
    time.sleep(0.3)  # let the fire-and-forget beep threads actually run before the test process exits
    logger.success("PASS: play_start_cue()/play_stop_cue() did not raise.")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO NEW SETTINGS TEST SUITE")
    logger.info("=========================================================")
    test_new_config_fields_round_trip()
    test_auto_copy_to_clipboard_skips_restoring_old_clipboard()
    test_pill_always_on_top_toggle_does_not_crash()
    test_audio_cues_do_not_raise()
    logger.success("=========================================================")
    logger.success(" ALL NEW SETTINGS TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
