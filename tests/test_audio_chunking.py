"""
Long-Recording Chunking Verification for Wisperno.
The old test (test_long_dictation.py) proved a long buffer reaches Whisper
untruncated, but used synthetic tones - it couldn't prove real SPEECH content
spread across the recording actually survives transcription. This one
synthesizes real, recognizable speech (Windows SAPI, via PowerShell -
no new Python dependency) with distinct checkpoint keywords at the start,
middle, and end of a 120+ second recording, separated by real silence gaps,
and verifies every checkpoint word survives - the exact shape of the reported
defect ("1m39s+ recording produced only two fragments", i.e. middle content
silently dropped).

Run: python tests/test_audio_chunking.py
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

# 5 checkpoints, spread across the recording, each with a distinctive keyword
# a real ASR model reliably recognizes - if chunking drops a segment, its
# keyword goes missing from the final transcript.
CHECKPOINTS = [
    ("alpha", "Checkpoint alpha. This is the very beginning of a long continuous dictation test. "
              "We are verifying that speech recognition captures every part of a long recording. "
              "The system must not drop any content near the start of the audio. "
              "This first section should be transcribed completely and accurately, word for word."),
    ("bravo", "Checkpoint bravo. This is roughly one quarter of the way through the recording. "
              "If chunking is working correctly this sentence will still be transcribed accurately. "
              "Nothing in the middle of a long dictation should ever be silently discarded. "
              "A real dictation tool has to handle pauses gracefully without losing any words."),
    ("charlie", "Checkpoint charlie. This is the halfway point of the entire test recording. "
               "A naive implementation relying only on a native thirty second window would likely "
               "lose this exact sentence, since it falls well past that boundary. "
               "Reaching the middle of a long recording is exactly where dropped audio would show up."),
    ("delta", "Checkpoint delta. This is roughly three quarters of the way through the recording. "
              "Every checkpoint word in this test must appear somewhere in the final transcript. "
              "Losing this sentence would prove that later chunks are being dropped. "
              "The recording is almost finished, but there is still more content to come after this."),
    ("echo", "Checkpoint echo. This is the very end of the long dictation test. "
             "If you can read this sentence then no audio was dropped anywhere in the recording "
             "from the very beginning all the way through to this final word. "
             "This concludes the full length verification of the long dictation pipeline."),
]

_PS_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = -3
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


def _load_wav_as_float32(path: Path) -> tuple:
    import scipy.io.wavfile as wavfile

    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32)
    from scipy import signal
    import math

    gcd = math.gcd(16000, sr)
    return signal.resample_poly(audio, 16000 // gcd, sr // gcd).astype(np.float32)


def _build_long_recording() -> np.ndarray:
    """Synthesizes each checkpoint via SAPI, resamples to 16kHz mono, and
    concatenates with explicit 1.5s silence gaps - real speech, real silence
    boundaries for VAD to align chunks against, well past CHUNK_TRIGGER_SEC."""
    with tempfile.TemporaryDirectory() as tmp:
        pieces = []
        gap = np.zeros(int(1.5 * 16000), dtype=np.float32)
        for i, (_keyword, text) in enumerate(CHECKPOINTS):
            wav_path = Path(tmp) / f"checkpoint_{i}.wav"
            _synthesize(text, wav_path)
            sr, audio = _load_wav_as_float32(wav_path)
            pieces.append(_resample_to_16k(audio, sr))
            if i < len(CHECKPOINTS) - 1:
                pieces.append(gap)
        return np.concatenate(pieces)


def test_chunk_boundaries_engage_on_long_audio() -> None:
    """Unit-level: _chunk_boundaries() must actually split a >20s buffer at
    real silence gaps, and must leave a normal short dictation untouched."""
    from src.config import load_config
    from src.transcriber import Transcriber
    from src.vocabulary import Vocabulary

    logger.info("--- Testing Transcriber._chunk_boundaries(): splits long audio, leaves short audio alone ---")
    config = load_config()
    transcriber = Transcriber(config=config.whisper, vocabulary=Vocabulary(db=None))

    short_audio = np.zeros(16000 * 5, dtype=np.float32)  # 5s - well under the trigger
    assert transcriber._chunk_boundaries(short_audio) == [(0, len(short_audio))], (
        "A short (5s) buffer was chunked - it should pass through as a single range unchanged."
    )

    long_audio = _build_long_recording()
    duration = len(long_audio) / 16000.0
    logger.info(f"Synthesized {duration:.1f}s of real speech for chunking test.")
    assert duration > 100.0, f"Synthesized audio was only {duration:.1f}s - need >100s to exercise this properly."

    boundaries = transcriber._chunk_boundaries(long_audio)
    logger.info(f"{len(boundaries)} chunk(s): {boundaries}")
    assert len(boundaries) > 1, f"A {duration:.1f}s recording was not split into multiple chunks."
    # Boundaries must cover the whole buffer with no gaps or overlaps.
    assert boundaries[0][0] == 0 and boundaries[-1][1] == len(long_audio)
    for (_, end_a), (start_b, _) in zip(boundaries, boundaries[1:]):
        assert end_a == start_b, f"Chunk boundaries have a gap or overlap: {boundaries}"
    logger.success(f"PASS: {duration:.1f}s recording split into {len(boundaries)} contiguous, VAD-aligned chunks.")
    return long_audio


def test_full_transcript_preserves_all_checkpoints() -> None:
    """End-to-end: every checkpoint keyword, from the very start to the very
    end of a 120+ second recording, must survive in the final transcript -
    the actual reported defect was mid-recording content silently vanishing."""
    from src.config import load_config
    from src.transcriber import Transcriber
    from src.vocabulary import Vocabulary

    logger.info("--- Testing Transcriber.transcribe(): no checkpoint keyword is lost across a 120s+ recording ---")
    config = load_config()
    transcriber = Transcriber(config=config.whisper, vocabulary=Vocabulary(db=None))
    if transcriber.model is None:
        logger.warning("No Whisper model loaded - skipping.")
        return

    long_audio = _build_long_recording()
    duration = len(long_audio) / 16000.0

    t0 = time.perf_counter()
    transcript = transcriber.transcribe(long_audio)
    elapsed = time.perf_counter() - t0
    logger.info(f"Transcribed {duration:.1f}s of audio in {elapsed:.1f}s -> {transcript!r}")

    lowered = transcript.lower()
    missing = [kw for kw, _ in CHECKPOINTS if kw not in lowered]
    assert not missing, (
        f"Checkpoint word(s) {missing} were dropped from the transcript of a {duration:.1f}s recording - "
        f"this is the reported defect. Full transcript: {transcript!r}"
    )
    logger.success(
        f"PASS: all {len(CHECKPOINTS)} checkpoints ({', '.join(kw for kw, _ in CHECKPOINTS)}) survived "
        f"across the full {duration:.1f}s recording."
    )


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO LONG-RECORDING CHUNKING TEST SUITE")
    logger.info("=========================================================")
    test_chunk_boundaries_engage_on_long_audio()
    test_full_transcript_preserves_all_checkpoints()
    logger.success("=========================================================")
    logger.success(" ALL AUDIO CHUNKING TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
