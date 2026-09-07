"""
Long-Form Dictation Verification for Wisperno.
Confirms the pipeline handles multi-second continuous audio without truncation:
unbounded in-memory accumulation in AudioRecorder, full-segment (not segments[0])
Whisper transcription across the whole buffer, and a persisted SQLite history row.

Run: python tests/test_long_dictation.py
"""

import sys
import time
from pathlib import Path
import numpy as np
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import load_config, get_db_path
from src.audio_recorder import AudioRecorder
from src.transcriber import Transcriber
from src.vocabulary import Vocabulary
from src.database import WispernoDB

TEST_DURATION_SEC = 25.0


def generate_long_speech_like_audio(duration_sec: float, sample_rate: int, channels: int) -> np.ndarray:
    """
    Multi-tone waveform in the 4-channel Realtek-native shape, with periodic
    silence gaps (syllable-like bursts) rather than one continuous tone - closer
    to real speech than a flat sine, and it exercises VAD's silence handling
    across a long buffer instead of one uninterrupted block.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    wave = (
        0.35 * np.sin(2 * np.pi * 180 * t)
        + 0.25 * np.sin(2 * np.pi * 400 * t)
        + 0.15 * np.sin(2 * np.pi * 900 * t)
    ).astype(np.float32)
    envelope = (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 0.5 * t))).astype(np.float32)  # ~1s on/off bursts
    wave *= envelope
    fade = int(sample_rate * 0.05)
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    return np.tile(wave.reshape(-1, 1), (1, channels)).astype(np.float32)


def test_audio_recorder_no_truncation() -> np.ndarray:
    """Feed TEST_DURATION_SEC of audio through the real callback path; assert nothing is dropped."""
    logger.info(f"--- Testing AudioRecorder: {TEST_DURATION_SEC:.0f}s continuous capture (no truncation) ---")
    config = load_config()
    recorder = AudioRecorder(config=config.audio)

    multi = generate_long_speech_like_audio(TEST_DURATION_SEC, recorder.native_sample_rate, recorder.native_channels)

    # Drive the exact same callback sounddevice.InputStream would call, in 100ms blocks,
    # so this exercises the real production code path rather than a parallel test-only one.
    recorder._is_recording = True
    recorder._start_time = time.perf_counter() - TEST_DURATION_SEC
    block = max(1, int(recorder.native_sample_rate * 0.1))
    for i in range(0, len(multi), block):
        chunk = multi[i : i + block]
        recorder._audio_callback(chunk, len(chunk), None, None)

    result = recorder.stop_recording()
    assert result is not None, f"{TEST_DURATION_SEC:.0f}s of audio was entirely discarded (truncation/threshold bug)."
    captured_sec = len(result) / 16000.0
    logger.info(f"Captured duration: {captured_sec:.2f}s (fed in: {TEST_DURATION_SEC:.1f}s)")
    assert captured_sec > TEST_DURATION_SEC - 1.0, (
        f"Audio was truncated: only {captured_sec:.2f}s captured out of {TEST_DURATION_SEC:.1f}s fed in."
    )
    logger.success(f"PASS: {captured_sec:.2f}s of continuous audio captured with zero truncation.")
    return result


def test_transcriber_processes_full_buffer(audio_array: np.ndarray) -> str:
    """Confirm Whisper receives and processes the WHOLE buffer, not a truncated prefix."""
    logger.info("--- Testing Transcriber: full-buffer processing on long audio ---")
    config = load_config()
    vocabulary = Vocabulary(db=None)
    transcriber = Transcriber(config=config.whisper, vocabulary=vocabulary)

    # Bypass transcribe()'s text joining to also inspect faster-whisper's own
    # reported `info.duration` - independent proof the full buffer reached Whisper,
    # regardless of whether synthetic tones produce any recognizable words.
    segments, info = transcriber.model.transcribe(
        audio_array, beam_size=1, vad_filter=True, language="en", condition_on_previous_text=False,
    )
    segments = list(segments)  # consume the generator fully, matching transcriber.transcribe()'s own loop
    reported_duration = info.duration
    logger.info(f"faster-whisper reported audio duration: {reported_duration:.2f}s, {len(segments)} segment(s).")
    assert reported_duration > TEST_DURATION_SEC - 1.0, (
        f"faster-whisper only saw {reported_duration:.2f}s of the {TEST_DURATION_SEC:.1f}s buffer - truncated input."
    )

    full_text = transcriber.transcribe(audio_array)  # exercises the real production path (all segments joined)
    logger.info(f"Transcript (synthetic tones, no real words expected): {full_text!r}")
    logger.success("PASS: Whisper processed the entire long-form buffer (verified via info.duration).")
    return full_text


def test_history_persists_long_dictation(duration_sec: float, raw_text: str) -> None:
    """Confirm a long-duration dictation round-trips through the SQLite history table."""
    logger.info("--- Testing SQLite history: long-duration record persists correctly ---")
    db_path = get_db_path()
    db = WispernoDB(db_path)

    polished_text = raw_text or "(synthetic tone - no speech content)"
    history_id = db.add_history(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        duration_seconds=duration_sec,
        raw_transcript=raw_text,
        polished_transcript=polished_text,
        mode_used="polish",
        latency_ms=1234.5,
    )
    assert history_id > 0

    rows = db.list_history(limit=5)
    match = next((r for r in rows if r["id"] == history_id), None)
    assert match is not None, "Newly-inserted long-dictation history row was not found on read-back."
    assert abs(match["duration_seconds"] - duration_sec) < 0.01
    assert match["polished_transcript"] == polished_text

    db.delete_history(history_id)  # this test's row shouldn't pollute the user's real history
    db.close()
    logger.success(f"PASS: {duration_sec:.1f}s dictation persisted to and read back from wisperno.db correctly.")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO LONG-FORM DICTATION TEST SUITE")
    logger.info("=========================================================")

    t0 = time.perf_counter()
    audio = test_audio_recorder_no_truncation()
    transcript = test_transcriber_processes_full_buffer(audio)
    test_history_persists_long_dictation(len(audio) / 16000.0, transcript)
    total_ms = (time.perf_counter() - t0) * 1000

    logger.success("=========================================================")
    logger.success(f" ALL LONG-FORM DICTATION TESTS PASSED in {total_ms:.0f}ms!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
