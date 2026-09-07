"""
Clean Teardown Verification for Wisperno.
Directly exercises the two concrete fixes for the reported "Suspended in
Task Manager, needs multiple End Task clicks" defect:
  1. WispernoEngine.stop() must close an audio stream left open mid-recording
     (a native PortAudio callback thread outside Python's daemon-thread
     control, previously never told to stop on shutdown).
  2. main.py's main() must terminate the process with os._exit(), not
     sys.exit() - sys.exit() waits for every OS thread (including native ones
     Python has no control over) before the process actually ends; os._exit()
     doesn't wait for anything.

Run: python tests/test_shutdown.py
"""

import ast
import inspect
import sys
import time
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_engine_stop_closes_open_audio_stream() -> None:
    """The concrete bug: a recording left open when shutdown is triggered
    (app closed before the hotkey was released) must not leave the native
    PortAudio stream running."""
    from src.config import load_config, get_db_path
    from src.database import WispernoDB
    from src.engine import WispernoEngine
    from src.workers import AudioWorker

    logger.info("--- Testing WispernoEngine.stop(): closes an audio stream left open mid-recording ---")
    engine = WispernoEngine.__new__(WispernoEngine)  # bypass __init__ (no heavy engines needed)
    engine.config = load_config()
    engine.db = WispernoDB(get_db_path())
    engine.hotkey_worker = None
    engine._live_worker = None
    engine.audio_worker = AudioWorker(engine.config.audio)

    try:
        started = engine.audio_worker.start_recording()
        if not started:
            logger.warning("Could not open a real audio stream on this machine - skipping.")
            return
        assert engine.audio_worker.recorder.is_recording, "Setup failed: recording did not actually start."

        engine.stop()

        assert not engine.audio_worker.recorder.is_recording, (
            "engine.stop() left the audio recorder in a 'recording' state - the native stream was never closed."
        )
        assert engine.audio_worker.recorder._stream is None, (
            "engine.stop() left the native PortAudio stream object open - this is the actual zombie-process cause."
        )
        logger.success("PASS: engine.stop() closed the in-flight audio stream cleanly.")
    finally:
        if engine.audio_worker.recorder.is_recording:
            engine.audio_worker.stop_recording()
        engine.db.close()


def test_main_uses_hard_exit_not_sys_exit() -> None:
    """sys.exit() alone reproduced the reported hang (waits for every OS
    thread, including native ones Python can't see); main() must call
    os._exit() as its final action instead."""
    logger.info("--- Testing main.py: main() terminates via os._exit(), not sys.exit() ---")
    main_source = (BASE_DIR / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(main_source, filename="main.py")

    main_func = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    assert main_func is not None, "Could not find a top-level main() function in main.py."

    def _dotted_name(call: ast.Call) -> str:
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            return f"{f.value.id}.{f.attr}"
        if isinstance(f, ast.Name):
            return f.id
        return ""

    call_names = [_dotted_name(c) for c in ast.walk(main_func) if isinstance(c, ast.Call)]
    assert "sys.exit" not in call_names, "main() still calls sys.exit() - it must call os._exit() instead."
    assert "os._exit" in call_names, "main() does not terminate via os._exit()."
    logger.success("PASS: main() terminates the process via os._exit(), never sys.exit().")


def test_hotkey_threads_are_all_daemons() -> None:
    """Every background thread the hotkey system spawns must be a daemon
    thread - a non-daemon thread left running blocks normal Python
    interpreter shutdown even before os._exit() is reached."""
    from src.hotkey_manager import HotkeyManager

    logger.info("--- Testing HotkeyManager: its own worker thread is a daemon ---")
    mgr = HotkeyManager()
    mgr.start()
    time.sleep(0.05)
    try:
        assert mgr._worker_thread is not None and mgr._worker_thread.is_alive()
        assert mgr._worker_thread.daemon, "HotkeyManager's polling thread is not a daemon thread."
        logger.success("PASS: HotkeyManager's polling thread is a daemon thread.")
    finally:
        mgr.stop()


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO CLEAN SHUTDOWN TEST SUITE")
    logger.info("=========================================================")
    test_engine_stop_closes_open_audio_stream()
    test_main_uses_hard_exit_not_sys_exit()
    test_hotkey_threads_are_all_daemons()
    logger.success("=========================================================")
    logger.success(" ALL SHUTDOWN TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
