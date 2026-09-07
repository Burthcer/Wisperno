"""
Audio duration accuracy test for Wisperno: simulates a multi-frame, multi-second
recording and verifies the duration Wisperno actually stores in history reflects
the full captured buffer, not a stray final chunk or a wall-clock timing quirk.

Run: python tests/test_audio_duration.py
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock
from loguru import logger

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

TARGET_DURATION_SEC = 10.0
TOLERANCE_SEC = 0.2


def test_audio_recorder_duration_from_full_buffer() -> None:
    """AudioRecorder.stop_recording() must compute duration from the whole
    concatenated buffer (many small callback chunks), not a single/last chunk."""
    logger.info("--- Testing AudioRecorder: duration reflects the full multi-chunk buffer ---")
    from src.config import AudioConfig
    from src.audio_recorder import AudioRecorder

    recorder = AudioRecorder(config=AudioConfig(min_energy_threshold=0.0, min_duration_sec=0.1))
    recorder.native_sample_rate = recorder.target_sample_rate  # skip resampling for an exact sample count
    recorder.native_channels = 1

    # Simulate ~100 small audio callbacks (like a real 1024-frame-blocksize
    # stream would deliver many times over a multi-second recording), not one
    # single chunk covering the whole duration - this is what would silently
    # mask a "only the last chunk counted" bug if one existed.
    chunk_frames = 1600  # 100ms per chunk @ 16kHz
    n_chunks = int(TARGET_DURATION_SEC * recorder.target_sample_rate / chunk_frames)
    recorder._is_recording = True
    for _ in range(n_chunks):
        chunk = (np.random.uniform(-0.3, 0.3, size=(chunk_frames, 1))).astype(np.float32)
        recorder._audio_queue.put(chunk)
    recorder._start_time = time.perf_counter() - TARGET_DURATION_SEC

    audio = recorder.stop_recording()
    assert audio is not None, "Recording was discarded (duration/energy filter) - fixture audio should pass both."

    buffer_duration = len(audio) / recorder.target_sample_rate
    logger.info(f"Fed {n_chunks} chunks ({TARGET_DURATION_SEC:.1f}s target) -> buffer duration {buffer_duration:.2f}s")
    assert abs(buffer_duration - TARGET_DURATION_SEC) <= TOLERANCE_SEC, (
        f"Buffer duration {buffer_duration:.2f}s is not within {TOLERANCE_SEC}s of the {TARGET_DURATION_SEC}s target - "
        f"looks like only part of the recording was counted."
    )
    assert abs(recorder.last_duration_seconds - TARGET_DURATION_SEC) <= TOLERANCE_SEC, (
        f"last_duration_seconds ({recorder.last_duration_seconds:.2f}s) doesn't match the buffer duration."
    )
    logger.success(f"PASS: {n_chunks}-chunk recording reports {buffer_duration:.2f}s (target {TARGET_DURATION_SEC:.1f}s +/- {TOLERANCE_SEC}s).")


def test_history_stores_buffer_duration_not_wall_clock() -> None:
    """
    engine._on_ptt_release() must pass the recorder's own buffer-derived
    duration to the history row, not a press-to-release wall-clock delta
    (which runs long from stream start/stop overhead and, if a hotkey
    press/release timestamp were ever mishandled, could be drastically wrong).
    """
    logger.info("--- Testing WispernoEngine: history duration comes from the audio buffer ---")
    from PySide6.QtCore import QObject
    from src.engine import WispernoEngine

    engine = WispernoEngine.__new__(WispernoEngine)  # bypass __init__ (no heavy engines needed)
    QObject.__init__(engine)  # still need the Qt metaobject machinery for Signals to bind
    engine._engines_ready = True
    engine._processing = False
    engine._ptt_press_time = time.perf_counter() - 999.0  # deliberately wrong/stale wall-clock delta
    engine.audio_worker = MagicMock()
    engine.audio_worker.recorder.last_duration_seconds = TARGET_DURATION_SEC
    engine.audio_worker.stop_recording.return_value = np.zeros(int(TARGET_DURATION_SEC * 16000), dtype=np.float32)
    engine.db = MagicMock()
    engine.db.list_snippets.return_value = []
    engine.active_mode = "polish"
    engine.state_changed = MagicMock()
    engine.transcriber = MagicMock()
    engine.transformer = MagicMock()
    engine.injector = MagicMock()
    engine.config = MagicMock()
    engine.config.auto_llm_polish = False

    captured = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            # InferenceWorker(audio_array, transcriber, transformer, injector, db, snippets, mode, record_duration_ms)
            captured["record_duration_ms"] = args[7]
        def start(self):
            pass
        state_changed = MagicMock()
        result_ready = MagicMock()
        finished = MagicMock()

    import src.engine as engine_mod
    original_worker_cls = engine_mod.InferenceWorker
    engine_mod.InferenceWorker = FakeWorker
    try:
        engine._on_ptt_release()
    finally:
        engine_mod.InferenceWorker = original_worker_cls

    stored_seconds = captured["record_duration_ms"] / 1000.0
    logger.info(f"Wall-clock press-to-release delta was ~999s (deliberately stale); stored duration: {stored_seconds:.2f}s")
    assert abs(stored_seconds - TARGET_DURATION_SEC) <= TOLERANCE_SEC, (
        f"History duration {stored_seconds:.2f}s doesn't match the audio buffer's {TARGET_DURATION_SEC}s - "
        f"engine is using wall-clock timing (or something else) instead of the recorder's own buffer duration."
    )
    logger.success(f"PASS: history duration ({stored_seconds:.2f}s) comes from the audio buffer, not stale wall-clock timing.")


def run_all_tests() -> None:
    logger.info("=" * 70)
    logger.info("  WISPERNO AUDIO DURATION TEST SUITE")
    logger.info("=" * 70)
    test_audio_recorder_duration_from_full_buffer()
    test_history_stores_buffer_duration_not_wall_clock()
    logger.success("=" * 70)
    logger.success(" ALL AUDIO DURATION TESTS PASSED!")
    logger.success("=" * 70)


if __name__ == "__main__":
    run_all_tests()
