"""
Startup Time Verification for Wisperno.
Confirms the Whisper model load no longer pays the ~173s Hugging Face Hub
network-probe penalty (root-caused and fixed via HF_HUB_OFFLINE=1 +
local_files_only=True). Also confirms the LLM loads promptly from its local
GGUF file (no Hub involvement there, but included so this is the single
"cold start" gate for the whole engine stack).

Run: python tests/test_startup_time.py
"""

import sys
import time
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

WHISPER_LOAD_CEILING_SEC = 10.0  # guards against the ~173s HF-Hub-network-stall regression, not sub-second variance
LLM_LOAD_CEILING_SEC = 30.0  # generous: multi-GB GGUF -> VRAM upload, unrelated to the HF bug


def test_whisper_loads_fast() -> None:
    from src.config import load_config
    from src.transcriber import Transcriber
    from src.vocabulary import Vocabulary

    logger.info("--- Testing Transcriber cold-start time (no HF network probe) ---")
    config = load_config()
    vocabulary = Vocabulary(db=None)

    t0 = time.perf_counter()
    Transcriber(config=config.whisper, vocabulary=vocabulary)
    elapsed = time.perf_counter() - t0

    logger.info(f"Whisper model loaded in {elapsed:.2f}s.")
    assert elapsed < WHISPER_LOAD_CEILING_SEC, (
        f"Whisper load took {elapsed:.2f}s, expected under {WHISPER_LOAD_CEILING_SEC:.0f}s - "
        "the HF Hub offline-mode fix may not be applied."
    )
    logger.success(f"PASS: Whisper loaded in {elapsed:.2f}s (< {WHISPER_LOAD_CEILING_SEC:.0f}s ceiling).")


def test_llm_loads_reasonably() -> None:
    from src.config import load_config
    from src.transformer import Transformer

    logger.info("--- Testing Transformer (local LLM) cold-start time ---")
    config = load_config()

    t0 = time.perf_counter()
    transformer = Transformer(config=config.llm, prompts=config.prompts)
    elapsed = time.perf_counter() - t0

    logger.info(f"LLM loaded in {elapsed:.2f}s.")
    if transformer.llm is None:
        logger.warning("No LLM weights found on disk - skipping timing assertion.")
        return
    assert elapsed < LLM_LOAD_CEILING_SEC, f"LLM load took {elapsed:.2f}s, expected under {LLM_LOAD_CEILING_SEC:.0f}s."
    logger.success(f"PASS: LLM loaded in {elapsed:.2f}s (< {LLM_LOAD_CEILING_SEC:.0f}s ceiling).")


def test_corrupt_config_falls_back_to_defaults() -> None:
    """A config.yaml that fails Pydantic validation must not crash boot -
    load_config() should log the error and return safe defaults instead."""
    import tempfile
    from src.config import AppConfig, load_config

    logger.info("--- Testing load_config(): corrupt/invalid config.yaml falls back to defaults ---")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write("hotkey: ctrl+z\nwhisper:\n  beam_size: not_a_number\n")  # wrong type -> Pydantic ValidationError
        bad_path = f.name

    try:
        cfg = load_config(bad_path)
        assert isinstance(cfg, AppConfig), "load_config did not return an AppConfig on validation failure."
        assert cfg.whisper.beam_size == 1, f"Expected safe default beam_size, got {cfg.whisper.beam_size!r}."
        # The whole file failed validation (one bad field poisons the document under
        # Pydantic), so the file's own "ctrl+z" must NOT leak through - confirms the
        # fallback is a real AppConfig() default set, not a partial/best-effort merge.
        assert cfg.tap_toggle_hotkey == "ctrl+alt", f"Expected the safe-default hotkey, got {cfg.tap_toggle_hotkey!r}."
        logger.success("PASS: invalid config.yaml fell back to safe defaults instead of raising.")
    finally:
        import os
        os.unlink(bad_path)


def test_default_hotkey_is_ctrl_alt() -> None:
    """Mission requirement: the default trigger shortcut across clean installs
    and config fallbacks must resolve to Ctrl+Alt - checked at every level that
    could otherwise disagree (the Pydantic field default, a missing config.yaml,
    and the Win32 poller's own constructor default)."""
    from src.config import AppConfig, load_config
    from src.hotkey_manager import HotkeyManager
    import inspect

    logger.info("--- Testing default hotkey resolves to Ctrl+Alt everywhere ---")
    assert AppConfig().tap_toggle_hotkey == "ctrl+alt", f"AppConfig() default hotkey is {AppConfig().tap_toggle_hotkey!r}, not 'ctrl+alt'."
    assert AppConfig().tap_toggle_enabled is True, "Tap-to-Toggle (the ctrl+alt default trigger) is not enabled by default."
    assert load_config("Z:\\definitely\\does\\not\\exist.yaml").tap_toggle_hotkey == "ctrl+alt", (
        "A missing config.yaml did not fall back to the ctrl+alt default."
    )
    tap_default = inspect.signature(HotkeyManager.__init__).parameters["tap_toggle_hotkey"].default
    assert tap_default == "ctrl+alt", f"HotkeyManager's own constructor default is {tap_default!r}, not 'ctrl+alt'."
    logger.success("PASS: AppConfig(), a missing config.yaml, and HotkeyManager's own default all agree on ctrl+alt.")


def test_boot_recovery_from_stuck_launch() -> None:
    """Simulates a previous launch that never finished (crash/hang/force-quit
    mid-load, evidenced by a leftover boot-guard file) - the next launch must
    force the known-good Standard preset instead of retrying whatever was
    configured, and must clear the guard file once it completes cleanly."""
    import shutil
    import tempfile
    from src.config import get_app_data_dir, get_config_dir
    from src.engine import WispernoEngine

    logger.info("--- Testing WispernoEngine: cross-launch recovery from a stuck previous boot ---")
    guard_path = get_app_data_dir() / "boot_in_progress.flag"
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    guard_path.touch()  # simulate a previous launch that never reached its success point

    with tempfile.TemporaryDirectory() as tmp:
        tmp_config = Path(tmp) / "config.yaml"
        shutil.copy(get_config_dir() / "config.yaml", tmp_config)

        engine = WispernoEngine(config_path=str(tmp_config))
        engine.config.model_preset = "flagship"  # simulate a bad/slow preset having been selected
        engine.config.llm.model_path = "models/Qwen2.5-3B-Instruct-Q8_0.gguf"
        try:
            engine.start()  # recovery runs synchronously before the load threads spawn
            assert engine.config.model_preset == "standard", (
                f"Expected recovery to force the Standard preset, got {engine.config.model_preset!r}"
            )
            assert engine._init_done_event.wait(timeout=30.0), "Engine did not finish loading after recovery."
            assert not guard_path.exists(), "Boot guard file was not cleared after a successful load."
            logger.success("PASS: stuck-boot recovery forced the Standard preset and cleared the guard file.")
        finally:
            engine.stop()


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO STARTUP TIME TEST SUITE")
    logger.info("=========================================================")
    t0 = time.perf_counter()
    test_whisper_loads_fast()
    test_llm_loads_reasonably()
    test_corrupt_config_falls_back_to_defaults()
    test_default_hotkey_is_ctrl_alt()
    test_boot_recovery_from_stuck_launch()
    total = time.perf_counter() - t0
    logger.success("=========================================================")
    logger.success(f" ALL STARTUP TIME TESTS PASSED in {total:.2f}s!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
