"""
Live Transcription pipeline verification for Wisperno.

Drives LiveTranscriptionWorker.run() directly (the real production loop, not
a re-implementation of its hop/confirm logic) with a stubbed AudioRecorder
and a stubbed word-level Transcriber, so this exercises the actual
confirm/trim/emit/join/cleanup code with deterministic, fast inputs instead
of a real microphone + a real multi-second Whisper load. The
anti-hallucination energy/VAD gate is deliberately forced open here (see
tests/test_live_transcriber_anti_loop.py for that gate's own dedicated,
real-audio/real-model coverage), and _trailing_pause_cut is forced to always
report "the whole current window is one confirmable pause" - this file's
concern is the confirm/trim/join/cleanup/discard/save mechanics with a fully
deterministic one-hop-per-sentence schedule, not VAD pause-detection
accuracy (which the real-audio suite above already covers). The
database-insert side is verified separately against a real WispernoDB (same
technique src/database.py's own _demo() uses).

Run: python tests/test_live_transcription.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


class _StubRecorder:
    """Feeds exactly `total_hops` one-second non-silent (but content-free -
    the stub transcriber below ignores audio content entirely) audio blocks,
    one per drain_available_audio() call, then stops the worker's loop -
    standing in for AudioRecorder without touching real hardware."""

    def __init__(self, worker, total_hops: int, chunk_seconds: float = 1.0, sample_rate: int = 16000):
        self._worker = worker
        self._chunk = np.full(int(chunk_seconds * sample_rate), 0.01, dtype=np.float32)
        self._total = total_hops
        self._n = 0

    def start_recording(self) -> bool:
        return True

    def stop_recording(self):
        return None

    def drain_available_audio(self):
        self._n += 1
        if self._n > self._total:
            return None
        if self._n == self._total:
            self._worker._running = False  # this iteration still finishes its own _hop() before exiting
        return self._chunk.copy()


class _StubTranscriber:
    """Deterministic stand-in for Transcriber.transcribe_live_window() - one
    canned sentence per hop. Combined with _force_gates_open() below (which
    makes every hop's ENTIRE window a confirmable "pause"), this confirms
    exactly one sentence per hop - a fully deterministic, hand-verifiable
    schedule: 5 hops -> 5 sentences confirmed in order."""

    SENTENCES = [
        "This is the first segment of the meeting.",
        "Now we are discussing the second topic.",
        "Here comes the third important point.",
        "The fourth segment covers action items.",
        "Finally the fifth segment wraps up the call.",
    ]

    def __init__(self):
        self.calls = 0

    def transcribe_live_window(self, audio_array, context=None):
        if self.calls >= len(self.SENTENCES):
            self.calls += 1
            return []
        words = self.SENTENCES[self.calls].split()
        self.calls += 1
        return [{"word": w, "start": i * 0.3, "end": (i + 1) * 0.3} for i, w in enumerate(words)]


def _force_gates_open(worker) -> None:
    """This test file's concern is the confirm/trim/join/cleanup mechanics,
    not VAD pause-detection accuracy (tests/test_live_transcriber_anti_loop.py
    and tests/test_live_streaming.py cover that directly, with real audio and
    the real gate). _trailing_pause_cut is forced to report the END of
    whatever window it's given as an immediate confirm boundary - every hop's
    full hypothesis gets confirmed right away, which is what makes the
    one-hop-per-sentence schedule above deterministic."""
    worker._has_energy = lambda audio: True
    worker._vad_timestamps = lambda audio: [{"start": 0, "end": len(audio)}]
    worker._speech_ratio = lambda timestamps, window_len: 1.0
    # A huge (not window_len-derived) sample count: this test's stub returns
    # one FULL canned sentence per hop regardless of the stub-fed buffer's
    # actual (arbitrary, content-free) size, so the confirm cutoff must not
    # be tied to real buffer sample counts - it must simply cover every word
    # the stub could ever return in one hop.
    worker._trailing_pause_cut = lambda timestamps, window_len: 10_000_000


def _run_session(worker) -> None:
    """Runs LiveTranscriptionWorker.run() synchronously in THIS thread
    (bypassing QThread.start()/a real background thread) - run() is a plain
    method, and Qt signal emission with the default (same-thread) direct
    connection works without a QApplication event loop, so this exercises
    the real loop deterministically and fast, with no thread-timing flakiness."""
    worker._running = True
    worker._paused = False
    worker._discarded = False
    worker._buffer = np.zeros(0, dtype=np.float32)
    worker._committed_chunks = []
    worker._start_time = time.perf_counter()
    worker.run()


def test_streaming_chunks_committed_and_joined() -> None:
    from src.config import AudioConfig
    from src.live_transcriber import LiveTranscriptionWorker, clean_live_transcript

    logger.info("--- Testing LiveTranscriptionWorker: hop-based streaming -> one confirm per hop ---")

    stub_transcriber = _StubTranscriber()
    worker = LiveTranscriptionWorker(stub_transcriber, AudioConfig())
    worker.recorder = _StubRecorder(worker, total_hops=5)  # 1 hop per sentence
    _force_gates_open(worker)

    received = []
    worker.text_chunk_received.connect(received.append)
    speculative_seen = []
    worker.speculative_text_changed.connect(speculative_seen.append)
    result_holder = {}
    worker.session_stopped.connect(lambda r: result_holder.update(r))

    _run_session(worker)

    assert received == _StubTranscriber.SENTENCES, f"Chunks were not confirmed in the correct order: {received}"

    assert result_holder, "session_stopped did not fire with a result"
    joined_raw = " ".join(_StubTranscriber.SENTENCES)
    assert result_holder["raw"] == joined_raw, f"Raw transcript was not joined correctly: {result_holder['raw']!r}"
    assert result_holder["cleaned"] == clean_live_transcript(joined_raw)
    assert result_holder["cleaned"], "Cleanup pass produced empty output for non-empty input"
    assert result_holder["word_count"] == len(result_holder["cleaned"].split())
    assert result_holder["duration_seconds"] > 0

    logger.success(
        f"PASS: 5 sentences confirmed (one per hop) and joined correctly "
        f"({len(joined_raw.split())} words raw -> {result_holder['word_count']} words cleaned)."
    )


def test_discard_produces_no_saved_result() -> None:
    from src.config import AudioConfig
    from src.live_transcriber import LiveTranscriptionWorker

    logger.info("--- Testing LiveTranscriptionWorker: Discard skips the save/cleanup path ---")
    stub_transcriber = _StubTranscriber()
    worker = LiveTranscriptionWorker(stub_transcriber, AudioConfig())
    worker.recorder = _StubRecorder(worker, total_hops=5)
    _force_gates_open(worker)

    result_holder = {}
    worker.session_stopped.connect(lambda r: result_holder.update(r) if r else result_holder.setdefault("_empty", True))

    worker._running = True
    worker._paused = False
    worker._discarded = True  # simulates the user clicking Discard before the loop even starts draining
    worker._buffer = np.zeros(0, dtype=np.float32)
    worker._committed_chunks = []
    worker._start_time = time.perf_counter()
    # Discard flips _running False too (see LiveTranscriptionWorker.discard()) - mirror that here directly
    # since this test drives run() without going through the public discard() setter.
    worker._running = False
    worker.run()

    assert result_holder.get("_empty") is True, "A discarded session must emit an empty result, not a saved one."
    logger.success("PASS: discarding a session emits an empty session_stopped result - nothing gets saved.")


def test_session_stop_writes_history_row() -> None:
    """Verifies the OTHER half of 'stopping a session triggers cleanup and
    inserts records with entry_type=live_transcript': the database side,
    against a real (temp-file) WispernoDB, mirroring exactly what
    WispernoEngine._on_live_session_stopped() does with a session_stopped
    result dict."""
    import os
    import tempfile

    from src.database import WispernoDB

    logger.info("--- Testing WispernoDB.add_live_transcript(): entry_type='live_transcript' row ---")
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    try:
        db = WispernoDB(path)
        result = {
            "raw": "so um the first topic is the the budget review",
            "cleaned": "So the first topic is the budget review.",
            "duration_seconds": 245.0,
            "word_count": 8,
        }
        history_id = db.add_live_transcript(
            timestamp="2026-01-01T00:00:00",
            duration_seconds=result["duration_seconds"],
            raw_transcript=result["raw"],
            polished_transcript=result["cleaned"],
        )
        assert history_id > 0

        live_rows = db.list_history(entry_type="live_transcript")
        assert len(live_rows) == 1 and live_rows[0]["id"] == history_id
        assert live_rows[0]["raw_transcript"] == result["raw"]
        assert live_rows[0]["polished_transcript"] == result["cleaned"]
        assert live_rows[0]["entry_type"] == "live_transcript"
        assert live_rows[0]["source"] == "dictation", "A live session came from voice, not a transform/selection-polish"
        assert live_rows[0]["mode_used"] == "live"

        assert db.list_history(entry_type="dictation") == [], "A live_transcript row must not appear under the Dictations filter"
        assert len(db.list_history()) == 1, "The 'All' filter (entry_type=None) must still return the live_transcript row"

        db.close()
        logger.success("PASS: a finished Live Transcription session inserts one entry_type='live_transcript' history row.")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO LIVE TRANSCRIPTION TEST SUITE")
    logger.info("=========================================================")
    test_streaming_chunks_committed_and_joined()
    test_discard_produces_no_saved_result()
    test_session_stop_writes_history_row()
    logger.success("=========================================================")
    logger.success(" ALL LIVE TRANSCRIPTION TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
