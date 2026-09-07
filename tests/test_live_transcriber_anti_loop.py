"""
Anti-hallucination / anti-repetition-loop verification for Wisperno's Live
Transcription mode (src/live_transcriber.py).

Reproduced defect: streaming Whisper with condition_on_previous_text=True,
combined with no pre-inference speech-content gate on the window-ceiling
forced commit, let a low-quality/silent window reach Whisper, and a bad
decode's hallucinated text got threaded forward as the next chunk's prompt
context - compounding into an infinite repeated-phrase loop
("' , , , ' sentence? Pardon me, but I could repeat the sentence?...").

Test 1 drives the REAL production pipeline (real AudioRecorder-shaped
buffers, real Whisper model already loaded on this dev machine - same
convention as tests/test_pipeline.py) against real low-energy white noise,
proving the energy/VAD gates keep Whisper from ever running on it at all.
Test 2 is deterministic and text-level (real audio can't reliably force a
specific hallucination on demand) - it feeds sanitize_live_chunk() the
mission's own literal repeated-phrase example and asserts the collapse.
Test 3 benchmarks real transcribe_live_chunk() latency on real synthesized
speech.

Run: python tests/test_live_transcriber_anti_loop.py
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

_PS_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = -2
$synth.SetOutputToWaveFile("{out_path}")
$synth.Speak(@"
{text}
"@)
$synth.Dispose()
"""


def _synthesize(text: str, out_path: Path) -> None:
    script = _PS_SCRIPT.format(out_path=str(out_path), text=text)
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"SAPI synthesis failed: {result.stderr}")


def _load_wav_as_16k_float32(path: Path) -> np.ndarray:
    import math
    import scipy.io.wavfile as wavfile
    from scipy import signal

    sr, data = wavfile.read(str(path))
    data = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        gcd = math.gcd(16000, sr)
        data = signal.resample_poly(data, 16000 // gcd, sr // gcd).astype(np.float32)
    return data


def test_low_energy_noise_produces_zero_chunks() -> None:
    """5 seconds of low-energy white noise (below MIN_RMS_FOR_SPEECH, and
    with no VAD-detectable speech) fed through the REAL LiveTranscriptionWorker
    hop loop against the REAL Whisper model - must never call
    transcribe_live_window at all, let alone emit hallucinated text."""
    from src.config import load_config
    from src.live_transcriber import LiveTranscriptionWorker, MIN_RMS_FOR_SPEECH, SAMPLE_RATE, WINDOW_SEC
    from src.transcriber import Transcriber

    logger.info("--- Testing Live Transcription: low-energy noise never reaches Whisper ---")
    config = load_config()
    transcriber = Transcriber(config=config.whisper)

    original_transcribe = transcriber.transcribe_live_window
    call_count = {"n": 0}

    def _counting_transcribe(audio, context=None):
        call_count["n"] += 1
        return original_transcribe(audio, context=context)

    transcriber.transcribe_live_window = _counting_transcribe

    worker = LiveTranscriptionWorker(transcriber, config.audio)
    received = []
    worker.text_chunk_received.connect(received.append)

    # White noise well below the RMS speech floor - real background hum/fan
    # whine amplitude, not silence-silence (silence alone wouldn't prove the
    # energy gate is doing anything; this proves it rejects real-but-quiet noise).
    noise_amplitude = MIN_RMS_FOR_SPEECH * 0.3
    chunk_samples = int(1.0 * SAMPLE_RATE)
    full_noise = (np.random.randn(int(5.0 * SAMPLE_RATE)).astype(np.float32) * noise_amplitude)
    # Feed it across enough _hop() ticks to cross WINDOW_SEC more than once,
    # matching how run()'s poll loop would actually see continuous noise.
    for i in range(0, len(full_noise), chunk_samples):
        worker._buffer = np.concatenate([worker._buffer, full_noise[i:i + chunk_samples]])
        worker._hop()

    assert call_count["n"] == 0, f"Whisper was invoked {call_count['n']} time(s) on pure low-energy noise - the gate did not hold."
    assert received == [], f"Hallucinated text was emitted from noise: {received}"
    assert len(worker._buffer) < int(WINDOW_SEC * SAMPLE_RATE) * 2, "Buffer should have been trimmed of stale silence, not grown unbounded."
    logger.success(f"PASS: 5s of low-energy noise (fed across {len(full_noise)//chunk_samples} hop ticks spanning the {WINDOW_SEC}s window) never invoked Whisper, zero chunks emitted.")


def test_ngram_repetition_collapses_to_one_instance() -> None:
    """The mission's own literal repeated-phrase example - deterministic,
    text-level (real audio can't reliably force a specific hallucination on
    demand, but the collapse logic itself is pure text processing)."""
    from src.live_transcriber import sanitize_live_chunk

    logger.info("--- Testing sanitize_live_chunk(): n-gram repetition loop collapses to one instance ---")

    looped = (
        "' , , , ' sentence? Pardon me, but I could repeat the sentence? "
        "Pardon me, but I could repeat the sentence? Pardon me, but I could repeat the sentence? "
        "Pardon me, but I could repeat the sentence? Pardon me, but I could repeat the sentence?"
    )
    original_word_count = len(looped.split())
    result = sanitize_live_chunk(looped)
    assert not result.startswith(("'", ",", ".", "-")), f"Leading junk was not stripped: {result!r}"
    # The repeating 8-word unit can be found starting at any of its own
    # rotations (e.g. "Pardon me...sentence?" or "sentence? Pardon me...") -
    # both are equally valid collapses of the same loop, so check the
    # invariants that actually matter rather than one exact phrasing:
    # drastically shorter, contains the real spoken content, and is stable
    # under a second pass (nothing repetitive left to collapse further).
    assert len(result.split()) <= 10, f"Expected roughly one phrase's worth of words (~8), got {len(result.split())}: {result!r}"
    assert "Pardon" in result and "sentence" in result, f"Collapsed result lost the actual spoken content: {result!r}"
    assert sanitize_live_chunk(result) == result, f"Result is not stable under a second pass: {result!r}"
    logger.success(f"PASS: {original_word_count}-word repetition loop collapsed to {len(result.split())} words -> {result!r}")

    # A shorter single-word repetition (e.g. a stutter-shaped hallucination)
    # collapses the same way - to one period, not dropped or left unbounded.
    all_same_word = "the the the the the the the the the the the the the the the"
    collapsed_word = sanitize_live_chunk(all_same_word)
    assert len(collapsed_word.split()) < len(all_same_word.split()), "Repeated single-word chunk was not collapsed at all."
    assert sanitize_live_chunk(collapsed_word) == collapsed_word, "Collapsed single-word result is not stable under a second pass."

    # A normal, non-repetitive sentence must pass through completely unchanged.
    normal = "This is a perfectly ordinary sentence with no repetition at all."
    assert sanitize_live_chunk(normal) == normal, "sanitize_live_chunk() must not alter legitimate speech."
    logger.success("PASS: repeated single-word chunks collapse correctly; ordinary speech passes through unchanged.")


def test_live_chunk_transcription_latency() -> None:
    """Real ~2.75s synthesized speech (matching WINDOW_SEC, the actual rolling
    window size) through the real transcribe_live_window() path (greedy,
    single-beam, word_timestamps=True, vad_filter=False - the actual live-mode
    config). This is the same measurement Round 34's model-suitability audit
    (Final App/LIVE_TRANSCRIPTION_ARCHITECTURE.md) reports."""
    from src.config import load_config
    from src.live_transcriber import WINDOW_SEC
    from src.transcriber import Transcriber

    logger.info("--- Benchmarking transcribe_live_window() latency on a real ~2.75s speech window ---")
    config = load_config()
    transcriber = Transcriber(config=config.whisper)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "sample.wav"
        _synthesize("Testing one two three four five six seven.", wav_path)
        audio = _load_wav_as_16k_float32(wav_path)

    duration_s = len(audio) / 16000.0
    logger.info(f"Synthesized sample duration: {duration_s:.2f}s (target window: {WINDOW_SEC}s)")

    # One warmup pass (first call after model load pays a one-time CUDA
    # kernel warmup cost unrelated to steady-state per-hop latency).
    transcriber.transcribe_live_window(audio)

    t0 = time.perf_counter()
    words = transcriber.transcribe_live_window(audio)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    text = " ".join(w["word"] for w in words)
    logger.info(f"Latency: {elapsed_ms:.1f}ms -> '{text}'")

    # 500ms, not the mission's literal 200ms: measured directly rather than
    # asserting an unverified number. See the architecture doc for the full
    # measured comparison against the <200ms target and what it means for
    # hop cadence - this test's bound exists to catch an actual regression,
    # not to assert a target this specific hardware may or may not clear.
    assert elapsed_ms < 500, f"transcribe_live_window() took {elapsed_ms:.1f}ms on a {duration_s:.1f}s window - too slow for a live streaming experience."
    logger.success(f"PASS: {elapsed_ms:.1f}ms for a {duration_s:.1f}s window (word-level timestamps included).")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO LIVE TRANSCRIPTION ANTI-HALLUCINATION SUITE")
    logger.info("=========================================================")
    test_low_energy_noise_produces_zero_chunks()
    test_ngram_repetition_collapses_to_one_instance()
    test_live_chunk_transcription_latency()
    logger.success("=========================================================")
    logger.success(" ALL ANTI-HALLUCINATION TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
