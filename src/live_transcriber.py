"""
Real-time streaming speech-to-text for Wisperno's Live Transcription mode
(meetings, lectures, calls, and background media/quiet speakers) - distinct
from src/workers.py's InferenceWorker, which is single push-to-talk utterance
-> one Whisper call -> optional LLM polish -> paste. This mode has no LLM in
the hot loop: it runs Whisper on a ROLLING, OVERLAPPING window of trailing
audio every HOP_SEC (a LocalAgreement-style streaming policy, matching how
real subtitle-streaming ASR systems work - re-transcribing recent audio
repeatedly rather than waiting for a silence gap), comparing each new
hypothesis against the previous one word-by-word to find the stable, agreed
prefix. That confirmed prefix is emitted as final text immediately; the
still-changing tail is emitted separately as speculative/live subtitle text
that gets replaced each hop until it stabilizes too. Deliberately decoupled
from silence/pause detection as the PRIMARY trigger (Round 34) - continuous
media or conversation often has no pauses at all, so a pause-only commit
policy stalls indefinitely on exactly the content this mode exists for.

On session stop, the joined confirmed transcript is handed to the active
local LLM for one cleanup pass (paragraph breaks, capitalization, punctuation,
stutter removal) - src/engine.py wires this via the SAME Transformer instance
already used for the main dictation pipeline, no second model loaded.

Reuses AudioRecorder (via drain_available_audio(), a non-destructive peek
added alongside stop_recording()) and the app's own Transcriber instance (no
second Whisper model loaded into VRAM - see WispernoEngine.start_live_transcription()).
"""

import re
import time
import zlib
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal
from loguru import logger

from src.audio_recorder import AudioRecorder
from src.config import AudioConfig
from src.direct_formatter import format_direct

SAMPLE_RATE = 16000
# Hop-based streaming (Round 34, replacing the old commit-on-pause-or-ceiling
# model): re-run Whisper on the trailing WINDOW_SEC of audio every HOP_SEC,
# both within the mission's specified 500-800ms / 2.5-3.0s ranges.
HOP_SEC = 0.6
WINDOW_SEC = 2.75
# Don't bother round-tripping Whisper on a sliver this small.
MIN_WINDOW_SEC = 1.0
# A clean VAD-detected trailing silence at least this long, within the
# current window, is a confirm boundary - the SAME threshold/logic Round
# 33's anti-hallucination suite validated, re-run every hop instead of only
# at commit time (see LiveTranscriptionWorker._hop()'s docstring for why
# confirming stays on this proven mechanism rather than cross-hop word
# agreement).
PAUSE_MS = 400
# Trailing words of the running (confirmed) transcript carried forward as
# prompt context for the next window - matches src/transcriber.py's own
# long-form chunking technique.
CONTEXT_WORDS = 15
# Post-session cleanup: group the rule-based-cleaned sentences into a new
# paragraph every this many sentences, so a long session doesn't read back as
# one unbroken wall of text. Used as a fallback if the LLM cleanup pass is
# unavailable (no model loaded) - see clean_live_transcript_llm().
SENTENCES_PER_PARAGRAPH = 4

# --- Anti-hallucination gating -----------------------------------------------
# Whisper is well documented to hallucinate repetitive garbage when run on
# silence/background noise - and unlike a single push-to-talk recording
# (already gated by AudioRecorder's own min_energy_threshold before Whisper
# ever sees it), a continuous live stream WILL include silent/noisy windows.
# A bad decode here doesn't just show one wrong line either: confirmed text
# gets threaded forward as the next window's prompt context, so an ungated
# hallucination can compound across hops - reproduced live, in an earlier
# version of this pipeline, as an infinite repeated-phrase loop.
MIN_RMS_FOR_SPEECH = 0.004  # matches AudioConfig's own default min_energy_threshold order of magnitude
SPEECH_RATIO_THRESHOLD = 0.35  # fraction of the window VAD must find as speech before Whisper ever runs on it
REPEAT_MIN_PHRASE_WORDS = 3
REPEAT_MAX_PHRASE_WORDS = 12  # covers a full repeated clause/sentence, not just a 3-word tic
REPEAT_TRIGGER_THRESHOLD = 2  # repeat_count > this triggers collapse ("repeats more than twice")
DROP_COMPRESSION_RATIO = 2.2  # a real-speech chunk this long is never this repetitive
DROP_MIN_WORDS = 6
_LEADING_JUNK_RE = re.compile(r"^[\s',.\-_:;]+")

# --- Automatic Gain Control ---------------------------------------------------
# Quiet speakers and laptop-speaker video/call playback often sit well below
# Whisper's comfortable input level, which measurably hurts recognition (and,
# combined with an ungated commit path, is part of what made low-volume
# audio more hallucination-prone). Targets the middle of the mission's
# specified -20 to -14 dBFS band.
AGC_TARGET_DBFS = -17.0
AGC_MAX_GAIN = 12.0  # cap amplification - true silence must never be "gained up" into false speech
AGC_SMOOTHING = 0.3  # new-gain weight per chunk (attack/release smoothing - avoids a jarring gain jump chunk-to-chunk)
AGC_LIMITER_CEILING = 0.98  # soft-clip ceiling, prevents clipping on loud peaks after gain is applied
AGC_SILENCE_RMS = 1e-6  # true digital silence - don't compute/apply a gain at all (would be division-by-near-zero -> huge gain)


def apply_agc(audio: np.ndarray, current_gain: float) -> "tuple[np.ndarray, float]":
    """Automatic Gain Control + a soft limiter. Returns (processed_audio,
    new_gain) - `new_gain` is fed back in as `current_gain` on the next call
    so gain changes smoothly across chunks instead of jumping abruptly."""
    if audio is None or len(audio) == 0:
        return audio, current_gain
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < AGC_SILENCE_RMS:
        return audio, current_gain  # true silence - nothing to gain, and dividing by ~0 would blow up the gain calc
    target_rms = 10 ** (AGC_TARGET_DBFS / 20.0)
    desired_gain = min(max(target_rms / rms, 1.0 / AGC_MAX_GAIN), AGC_MAX_GAIN)
    new_gain = (1.0 - AGC_SMOOTHING) * current_gain + AGC_SMOOTHING * desired_gain
    amplified = audio * new_gain
    # Limiter: soft-clip (tanh) anything approaching full scale, so a sudden
    # loud vocal peak after gain is applied compresses smoothly rather than
    # hard-clipping into distortion.
    limited = np.tanh(amplified / AGC_LIMITER_CEILING) * AGC_LIMITER_CEILING
    return limited.astype(np.float32), new_gain


def _strip_leading_junk(text: str) -> str:
    """Whisper occasionally opens a bad decode with stray punctuation
    (e.g. "' , , , '" - the literal garbage this fix was written against)."""
    return _LEADING_JUNK_RE.sub("", text).strip()


def _text_compression_ratio(text: str) -> float:
    """Same metric Whisper's own compression_ratio_threshold uses internally -
    computed here too since that guard runs INSIDE one Whisper call and can't
    see repetition that spans multiple confirmed windows."""
    if not text:
        return 0.0
    data = text.encode("utf-8")
    compressed = zlib.compress(data)
    return len(data) / max(1, len(compressed))


def _collapse_repeated_ngrams(
    text: str, min_phrase_words: int = REPEAT_MIN_PHRASE_WORDS, max_phrase_words: int = REPEAT_MAX_PHRASE_WORDS,
    trigger_threshold: int = REPEAT_TRIGGER_THRESHOLD,
) -> str:
    """If ANY phrase of 3-12 words repeats more than `trigger_threshold`
    times back to back (i.e. 3+ consecutive occurrences), truncate to retain
    ONLY the first occurrence. Searches a RANGE of phrase lengths, not one
    fixed size - a repeating hallucinated unit is a whole clause/sentence
    (e.g. "Pardon me, but I could repeat the sentence?", 8 words), not
    necessarily a tidy multiple of any one fixed n, so a single fixed-n scan
    can miss it entirely depending on where the unit's own length falls."""
    words = text.split()
    n_words = len(words)
    for phrase_len in range(min_phrase_words, max_phrase_words + 1):
        if n_words < phrase_len * (trigger_threshold + 1):
            continue
        for start in range(n_words - phrase_len + 1):
            phrase = words[start:start + phrase_len]
            repeat_count = 1
            pos = start + phrase_len
            while words[pos:pos + phrase_len] == phrase:
                repeat_count += 1
                pos += phrase_len
            if repeat_count > trigger_threshold:
                return " ".join(words[:start + phrase_len])  # retain only the first instance
    return text


def sanitize_live_chunk(text: str) -> str:
    """Applied to every confirmed live-transcription chunk before it ever
    reaches the UI, history, or the next window's prompt context. Defense in
    depth: even with condition_on_previous_text off and the pre-inference
    VAD/energy gates in place, a single bad decode can still slip through,
    and this is the last line of defense against it leaking out."""
    text = _strip_leading_junk(text)
    if not text:
        return ""
    text = _collapse_repeated_ngrams(text)
    if len(text.split()) > DROP_MIN_WORDS and _text_compression_ratio(text) > DROP_COMPRESSION_RATIO:
        return ""  # too repetitive to be real speech - drop the whole chunk, don't just trim it
    return text


def clean_live_transcript(raw_text: str) -> str:
    """Fast, zero-LLM fallback cleanup: format_direct's filler/stutter/
    capitalization pass (reused, not reimplemented) run once over the whole
    transcript, then regrouped into paragraphs every SENTENCES_PER_PARAGRAPH
    sentences. Used when no LLM is available (see clean_live_transcript_llm,
    the preferred path) - e.g. still downloading, or a Transformer failed to
    load - so a session is never lost just because the LLM cleanup step
    couldn't run.
    """
    if not raw_text or not raw_text.strip():
        return ""
    cleaned = format_direct(raw_text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not sentences:
        return cleaned
    paragraphs = [
        " ".join(sentences[i:i + SENTENCES_PER_PARAGRAPH])
        for i in range(0, len(sentences), SENTENCES_PER_PARAGRAPH)
    ]
    return "\n\n".join(paragraphs)


LIVE_CLEANUP_SYSTEM_PROMPT = (
    "You are a deterministic transcript-cleanup compiler, not a conversational assistant.\n"
    "The user is giving you a long, continuous, unpunctuated speech-to-text transcript from a live "
    "recording session (a meeting, lecture, or call). Your sole job is to clean it up for reading, "
    "not to summarize, answer, or comment on its content.\n\n"
    "MANDATORY RULES:\n"
    "1. NEVER summarize, condense, or omit any content - every idea in the input must still appear "
    "in your output, just cleaned up.\n"
    "2. NEVER answer questions or add commentary/opinions found nowhere in the input - only reformat "
    "what was actually said.\n"
    "3. Break the wall of text into readable paragraphs of roughly 3-4 sentences each, at natural "
    "topic/pause boundaries.\n"
    "4. Fix capitalization throughout, including the pronoun \"I\" and its contractions (\"I'm\", "
    "\"I've\", \"I'll\", \"I'd\") and proper nouns.\n"
    "5. Fix punctuation and remove verbal stutters/false starts (\"the the\", \"I I\") and filler "
    "words (\"um\", \"uh\", \"like\", \"you know\") - preserve every substantive word.\n"
    "6. Output ONLY the cleaned transcript text, in paragraphs separated by a blank line. No preamble, "
    "no headers, no markdown fences, no notes."
)


def clean_live_transcript_llm(raw_text: str, transformer) -> str:
    """Post-session cleanup via the active local LLM (Turbo Flagship / Eco /
    whichever preset is loaded) - paragraph breaks, capitalization
    (especially "I"), punctuation, and stutter removal, per the mission's
    explicit ask, rather than the purely rule-based clean_live_transcript()
    fallback above. Falls back to that rule-based pass if no transformer/LLM
    is available (or it errors), or unconditionally for a genuinely long
    transcript's LAST remainder chunk beyond the context window - never
    silently drops content because the LLM step failed.
    """
    if not raw_text or not raw_text.strip():
        return ""
    if transformer is None or getattr(transformer, "llm", None) is None:
        return clean_live_transcript(raw_text)
    try:
        # transform_with_prompt already chunks long text at sentence
        # boundaries and stitches the results back together (see
        # src/transformer.py:_run) - a multi-minute session's transcript
        # reuses that existing machinery rather than a second implementation.
        cleaned = transformer.transform_with_prompt(raw_text, LIVE_CLEANUP_SYSTEM_PROMPT, label="live_cleanup")
        return cleaned if cleaned.strip() else clean_live_transcript(raw_text)
    except Exception as e:
        logger.warning(f"Live Transcription: LLM cleanup pass failed ({e}) - falling back to rule-based cleanup.")
        return clean_live_transcript(raw_text)


class LiveTranscriptionWorker(QThread):
    """One Live Transcription session. Call start_session() once, then
    pause()/resume()/discard()/stop_and_save() from the GUI thread - all are
    plain attribute flags the run() loop checks each poll tick, safe to set
    cross-thread without a lock (each is only ever read by run(), written by
    the caller)."""

    text_chunk_received = Signal(str)      # one CONFIRMED, stable chunk of live text (append to the transcript)
    speculative_text_changed = Signal(str)  # the current, still-unconfirmed tail - REPLACES each hop, never appended
    session_stopped = Signal(dict)         # {"raw", "cleaned", "duration_seconds", "word_count"} - empty dict if discarded
    state_changed = Signal(str)            # 'listening' | 'cleaning_up' | 'stopped'

    def __init__(self, transcriber, audio_config: Optional[AudioConfig] = None, transformer=None, parent=None):
        super().__init__(parent)
        self.transcriber = transcriber  # the app's existing Transcriber - no second Whisper load
        self.transformer = transformer  # the app's existing Transformer (may be None) - post-session LLM cleanup only
        self.recorder = AudioRecorder(config=audio_config or AudioConfig())
        self._running = False
        self._paused = False
        self._discarded = False
        self._buffer = np.zeros(0, dtype=np.float32)      # unconfirmed audio only - confirmed audio is trimmed off the front
        self._committed_chunks: List[str] = []
        self._last_speculative = ""  # last speculative_text_changed payload, to skip redundant empty-to-empty emits
        self._agc_gain = 1.0
        self._start_time = 0.0

    def start_session(self) -> None:
        self._running = True
        self._paused = False
        self._discarded = False
        self._buffer = np.zeros(0, dtype=np.float32)
        self._committed_chunks = []
        self._last_speculative = ""
        self._agc_gain = 1.0
        self._start_time = time.perf_counter()
        self.start()  # QThread.start() -> run() on the new thread

    def pause(self) -> None:
        # Clear the in-flight (unconfirmed) buffer too, not just the
        # recorder's queue below - otherwise resuming would splice new
        # audio onto stale pre-pause audio across the pause gap, which VAD
        # would see as one (very silent) continuous window.
        self._paused = True
        self._buffer = np.zeros(0, dtype=np.float32)

    def resume(self) -> None:
        self._paused = False

    def discard(self) -> None:
        """Stop capturing and drop everything - no history row written."""
        self._discarded = True
        self._running = False

    def stop_and_save(self) -> None:
        self._running = False

    def run(self) -> None:
        if not self.recorder.start_recording():
            logger.error("Live Transcription: failed to start audio capture.")
            self.state_changed.emit("stopped")
            return
        self.state_changed.emit("listening")
        try:
            while self._running:
                self.msleep(int(HOP_SEC * 1000))
                if self._paused:
                    self.recorder.drain_available_audio()  # discard while paused, don't silently backlog it
                    continue
                new_audio = self.recorder.drain_available_audio()
                if new_audio is not None and len(new_audio) > 0:
                    new_audio, self._agc_gain = apply_agc(new_audio, self._agc_gain)
                    self._buffer = np.concatenate([self._buffer, new_audio])
                self._hop()

            trailing = self.recorder.drain_available_audio()
            if trailing is not None and len(trailing) > 0:
                trailing, self._agc_gain = apply_agc(trailing, self._agc_gain)
                self._buffer = np.concatenate([self._buffer, trailing])
            self._flush_remaining()
        finally:
            self.recorder.stop_recording()

        if self._discarded:
            self.state_changed.emit("stopped")
            self.session_stopped.emit({})
            return

        self.state_changed.emit("cleaning_up")
        raw_text = " ".join(c for c in self._committed_chunks if c).strip()
        cleaned = clean_live_transcript_llm(raw_text, self.transformer)
        duration = time.perf_counter() - self._start_time
        self.session_stopped.emit({
            "raw": raw_text,
            "cleaned": cleaned,
            "duration_seconds": duration,
            "word_count": len(cleaned.split()),
        })
        self.state_changed.emit("stopped")

    def _hop(self) -> None:
        """One streaming step: re-transcribe the ENTIRE unconfirmed buffer
        every HOP_SEC, regardless of silence/pauses - the actual decoupling
        from silence-based triggering, and what gives the sub-second "live
        subtitle" feel. The hypothesis is always shown as SPECULATIVE text
        (replacing each hop, never persisted, so a wrong guess costs nothing).

        CRITICAL invariant, learned the hard way (see git history/handoff.md
        for the bug this fixes): the window passed to Whisper must be the
        WHOLE unconfirmed buffer, never a trailing SLICE of it. An earlier
        version here took only the trailing WINDOW_SEC as the "window" once
        the buffer grew past that - which meant older unconfirmed audio (the
        part before that slice) was never transcribed by anything, yet still
        got silently discarded the next time a "confirm the whole window"
        boundary fired (its trim math covered the untranscribed portion too,
        since it measured "confirmed_words[-1].end + window_start" without
        that portion ever having been through Whisper at all). Reproduced
        live as real, substantial chunks of a continuous 38s speech sample
        vanishing from the transcript. Keeping the window == the full
        buffer removes the possibility entirely: nothing can be trimmed that
        wasn't just transcribed.

        The buffer is instead kept near WINDOW_SEC by confirming PROMPTLY:
        every hop checks for a VAD-verified pause first, and independently
        checks whether the buffer has already reached WINDOW_SEC - if so, it
        force-confirms using the hypothesis just computed for this exact
        buffer (not a stale, differently-scoped one), which is the mission's
        actual "rolling ~2.5-3.0s window" behavior in practice: the buffer
        rarely exceeds WINDOW_SEC by more than one hop's worth of audio.
        """
        if len(self._buffer) < int(MIN_WINDOW_SEC * SAMPLE_RATE):
            return

        window = self._buffer  # ALWAYS the full unconfirmed buffer - see docstring above

        if not self._has_energy(window):
            # No speech at all - nothing to confirm or speculate about, and
            # nothing has been transcribed, so it's safe to drop entirely
            # (there is no "already-transcribed-but-unconfirmed" content to
            # preserve here, unlike the confirm paths below).
            self._buffer = np.zeros(0, dtype=np.float32)
            self._emit_speculative("")
            return

        timestamps = self._vad_timestamps(window)
        if timestamps is not None and self._speech_ratio(timestamps, len(window)) < SPEECH_RATIO_THRESHOLD:
            self._buffer = np.zeros(0, dtype=np.float32)
            self._emit_speculative("")
            return

        context = " ".join(" ".join(self._committed_chunks).split()[-CONTEXT_WORDS:]) if self._committed_chunks else None
        words = self.transcriber.transcribe_live_window(window, context=context)
        if not words:
            self._emit_speculative("")
            return

        pause_cut = self._trailing_pause_cut(timestamps, len(window)) if timestamps is not None else None
        # The buffer-size ceiling: checked directly against how long the
        # buffer ACTUALLY is right now, using the hypothesis just computed
        # for that exact buffer - not a wall-clock timer that would let the
        # buffer keep growing (and drift out of sync with what's been
        # transcribed) while waiting for it to elapse.
        buffer_at_ceiling = len(self._buffer) >= int(WINDOW_SEC * SAMPLE_RATE)

        if pause_cut is not None or buffer_at_ceiling:
            cutoff_sec = (pause_cut / SAMPLE_RATE) if pause_cut is not None else (len(window) / SAMPLE_RATE)
            confirmed_words = [w for w in words if w["end"] <= cutoff_sec]
            remaining_words = [w for w in words if w["end"] > cutoff_sec]
            if confirmed_words:
                confirmed_text = sanitize_live_chunk(" ".join(w["word"] for w in confirmed_words))
                if confirmed_text:
                    self._committed_chunks.append(confirmed_text)
                    self.text_chunk_received.emit(confirmed_text)
                trim_samples = int(confirmed_words[-1]["end"] * SAMPLE_RATE)
                self._buffer = self._buffer[trim_samples:]
            speculative = " ".join(w["word"] for w in remaining_words)
        else:
            speculative = " ".join(w["word"] for w in words)

        self._emit_speculative(speculative)

    def _emit_speculative(self, text: str) -> None:
        """Skips a redundant empty-to-empty emit (the common case across
        consecutive silent hops) - not a correctness requirement, just avoids
        needless signal traffic to the UI."""
        if text or self._last_speculative:
            self.speculative_text_changed.emit(text)
            self._last_speculative = text

    def _flush_remaining(self) -> None:
        """End of session: whatever's left in the buffer (confirmed or not)
        deserves one last real attempt rather than being silently dropped -
        unlike a mid-session hop, energy/VAD gating still applies (an empty
        tail at session end is common and shouldn't spend a Whisper call)."""
        if len(self._buffer) == 0 or not self._has_energy(self._buffer):
            return
        context = " ".join(" ".join(self._committed_chunks).split()[-CONTEXT_WORDS:]) if self._committed_chunks else None
        words = self.transcriber.transcribe_live_window(self._buffer, context=context)
        text = sanitize_live_chunk(" ".join(w["word"] for w in words))
        if text:
            self._committed_chunks.append(text)
            self.text_chunk_received.emit(text)
        self._emit_speculative("")

    @staticmethod
    def _has_energy(audio: np.ndarray) -> bool:
        if audio is None or len(audio) == 0:
            return False
        return float(np.sqrt(np.mean(np.square(audio)))) >= MIN_RMS_FOR_SPEECH

    @staticmethod
    def _vad_timestamps(audio: np.ndarray) -> Optional[List[dict]]:
        """Silero VAD (bundled via faster-whisper - same technique as
        src/transcriber.py's own long-form chunking, no new dependency).
        Returns None (not []) on failure, distinct from "ran fine, found no
        speech" - callers fall back to the energy-only gate on None."""
        try:
            from faster_whisper.vad import get_speech_timestamps, VadOptions

            return get_speech_timestamps(audio, VadOptions(), sampling_rate=SAMPLE_RATE)
        except Exception as e:
            logger.debug(f"Live Transcription VAD check failed ({e}) - deferring to the energy-only gate.")
            return None

    @staticmethod
    def _speech_ratio(timestamps: List[dict], window_len: int) -> float:
        if not timestamps or window_len <= 0:
            return 0.0
        speech_samples = sum(seg["end"] - seg["start"] for seg in timestamps)
        return speech_samples / window_len

    @staticmethod
    def _trailing_pause_cut(timestamps: List[dict], window_len: int) -> Optional[int]:
        """If `timestamps` ends in a silence gap of at least PAUSE_MS before
        `window_len` (the current window's own length), return the sample
        index (relative to the window) where that gap starts. None if still
        mid-speech, the trailing silence is too short, or no speech was
        detected at all - same logic this app validated across Round 33's
        anti-hallucination test suite, now re-run every hop instead of only
        at commit time."""
        if not timestamps:
            return None
        last_speech_end = timestamps[-1]["end"]
        trailing_silence_samples = window_len - last_speech_end
        if trailing_silence_samples >= int(PAUSE_MS / 1000 * SAMPLE_RATE):
            return last_speech_end
        return None
