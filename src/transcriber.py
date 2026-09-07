"""
Transcription Engine for Wisperno.
Wraps faster-whisper (CTranslate2) with CUDA acceleration and VAD silence filtering.
"""

import os
import sys
import time
from typing import Optional, List, Tuple
import numpy as np
from loguru import logger

# Defense-in-depth: main.py sets this before any import in the packaged app,
# but this module is also imported directly by tests and the dashboard's
# sandbox preview, which bypass main.py entirely. Without it, huggingface_hub
# probes the network for a revision check even on fully-cached weights and
# hangs for the full socket timeout (~173s, measured) before local_files_only
# forces the fallback - setting it here too closes that gap.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from faster_whisper import WhisperModel

from src.config import WhisperConfig, get_base_dir
from src.vocabulary import Vocabulary

# Above this many seconds of buffered audio, split into VAD-aligned chunks
# before ever calling Whisper, rather than handing it one huge buffer and
# trusting faster-whisper's own internal long-form handling end to end - a
# recording well past Whisper's native ~30s window (reported: 1m39s+
# continuous speech coming back as only two fragments) is exactly the case
# an explicit, app-level chunking pass is meant to make robust: each chunk
# stays comfortably inside Whisper's native window, and a real silence gap
# (not a mid-word cut) is chosen as the boundary every time one exists.
CHUNK_TRIGGER_SEC = 20.0
# A pause has to last at least this long to count as a valid chunk boundary -
# shorter than this and it's collapsed into the surrounding speech segment
# (a normal breath/comma pause, not a real break), matching VAD's own
# min_silence_duration_ms so the two never disagree about what "a pause" is.
CHUNK_MIN_SILENCE_MS = 300
# Trailing words of the previous chunk's own transcript carried forward as
# initial_prompt context for the next chunk, so a sentence split across a
# chunk boundary still reads coherently instead of two disconnected halves.
CHUNK_CONTEXT_WORDS = 15
# Wispr Flow-style paragraph breaks: a mid-dictation pause at least this long
# between two Whisper segments starts a new paragraph ("\n\n") instead of
# just continuing the sentence with a space - the natural-pause range a user
# actually stops and resumes a new thought in, per real dictation behavior.
PARAGRAPH_PAUSE_SEC = 1.3


class Transcriber:
    """
    High-performance Speech-to-Text inference engine powered by faster-whisper.
    """

    def __init__(self, config: Optional[WhisperConfig] = None, vocabulary: Optional[Vocabulary] = None):
        self.config = config or WhisperConfig()
        self.vocabulary = vocabulary or Vocabulary(db=None)
        self._ensure_cuda_dlls()
        self.model = self._load_model()

    def _ensure_cuda_dlls(self) -> None:
        """
        Add PyTorch / CUDA runtime DLL directories to Windows DLL search path if on Windows.
        """
        if sys.platform == "win32":
            try:
                import torch
                torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
                if os.path.exists(torch_lib_path):
                    if hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(torch_lib_path)
                    os.environ["PATH"] = torch_lib_path + os.path.pathsep + os.environ.get("PATH", "")
            except Exception as e:
                logger.debug(f"Could not add torch lib to DLL search path: {e}")

    def _load_model(self) -> WhisperModel:
        """
        Initialize faster-whisper model on GPU with fallback to CPU if needed.
        """
        logger.info(
            f"Loading Whisper model '{self.config.model_name}' on {self.config.device} "
            f"({self.config.compute_type})..."
        )
        t0 = time.perf_counter()
        download_root = str(get_base_dir() / "models")
        try:
            model = WhisperModel(
                self.config.model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root=download_root,
                local_files_only=True,
            )
            load_time = (time.perf_counter() - t0) * 1000
            logger.info(f"Whisper model loaded successfully in {load_time:.1f}ms.")
            return model
        except Exception as e:
            if self.config.device == "cuda":
                logger.warning(f"CUDA initialization failed for Whisper: {e}. Falling back to CPU.")
                model = WhisperModel(
                    self.config.model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=download_root,
                    local_files_only=True,
                )
                return model
            raise

    def transcribe(self, audio_array: np.ndarray) -> str:
        """
        Transcribe in-memory float32 16kHz mono audio array to text.

        Args:
            audio_array: np.ndarray of shape (N,) with float32 samples in range [-1.0, 1.0].

        Returns:
            Cleaned transcription string.
        """
        if audio_array is None or len(audio_array) == 0:
            return ""

        t0 = time.perf_counter()
        biased_prompt = self.vocabulary.build_initial_prompt(self.config.initial_prompt)

        chunks = self._chunk_boundaries(audio_array)
        if len(chunks) <= 1:
            full_text = self._transcribe_one_pass(audio_array, biased_prompt)
        else:
            logger.info(
                f"Long recording ({len(audio_array)/16000:.1f}s) - splitting into {len(chunks)} "
                f"VAD-aligned chunk(s) to stay within Whisper's native window."
            )
            pieces: List[str] = []
            for start, end in chunks:
                # Carry the trailing words of the previous chunk's own transcript
                # forward as context, so a sentence split across the chunk
                # boundary still reads coherently rather than as two disconnected
                # halves - not the same as condition_on_previous_text (off below,
                # per this app's existing anti-hallucination stance), just enough
                # prompt continuity to bridge the cut.
                context_words = " ".join(pieces[-1].split()[-CHUNK_CONTEXT_WORDS:]) if pieces else ""
                chunk_prompt = f"{biased_prompt} {context_words}".strip() if context_words else biased_prompt
                chunk_text = self._transcribe_one_pass(audio_array[start:end], chunk_prompt)
                if chunk_text:
                    pieces.append(chunk_text)
            full_text = " ".join(pieces).strip()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Transcribed {len(audio_array)/16000:.2f}s audio -> '{full_text}' [{elapsed_ms:.1f}ms]")
        return full_text

    def _transcribe_one_pass(self, audio_array: np.ndarray, biased_prompt: str) -> str:
        """One chunk (already known to fit comfortably inside Whisper's native window) through
        the model, with the existing no-VAD retry safety net for a quiet/short/oddly-mic'd chunk
        VAD misjudges as silent."""
        vad_params = {
            "threshold": self.config.vad_threshold,
            "min_speech_duration_ms": self.config.vad_min_speech_duration_ms,
            "min_silence_duration_ms": self.config.vad_min_silence_duration_ms,
            "speech_pad_ms": self.config.vad_speech_pad_ms,
        } if self.config.vad_filter else None

        text, _info = self._run_transcription(audio_array, biased_prompt, vad_filter=self.config.vad_filter, vad_params=vad_params)
        if not text:
            logger.warning("Transcription was empty after VAD filtering - retrying without VAD before giving up.")
            text, _info = self._run_transcription(audio_array, biased_prompt, vad_filter=False, vad_params=None)

        return self.vocabulary.apply_replacements(text.strip())

    def transcribe_live_window(self, audio_array: np.ndarray, context: Optional[str] = None) -> List[dict]:
        """
        Live Transcription mode (src/live_transcriber.py): transcribes one
        ROLLING WINDOW of trailing audio (re-run every hop on overlapping
        content, per the mission's LocalAgreement-style streaming design -
        distinct from the old commit-once-per-pause chunk model), returning
        WORD-LEVEL timestamps (word_timestamps=True) so the caller can align
        successive hypotheses word-by-word to find the stable, agreed-upon
        prefix and know exactly which audio samples that prefix corresponds
        to (for trimming the rolling buffer).

        condition_on_previous_text=False - deliberately, and consistent with
        the main transcribe() path above's own existing anti-hallucination
        stance (see that method's comment). It feeds the model's own prior
        DECODED TOKENS back into itself; on a low-quality/near-silent window
        (a real risk here specifically, since live audio includes room noise
        between utterances, unlike a single push-to-talk recording) that lets
        one bad decode's hallucination compound into the next call -
        reproduced live as an infinite repeated-phrase loop in an earlier
        version of this pipeline. Cross-window coherence is instead carried
        the SAME safe way the main path already does it: `context` threaded
        through as plain `initial_prompt` text.

        No internal VAD pass here (vad_filter=False) - the caller
        (LiveTranscriptionWorker) already gates on VAD speech-ratio before
        ever calling this, and re-filtering would only risk clipping the
        window's edges, not improve it.
        """
        if audio_array is None or len(audio_array) == 0:
            return []
        biased_prompt = self.vocabulary.build_initial_prompt(self.config.initial_prompt)
        prompt = f"{biased_prompt} {context}".strip() if context else biased_prompt
        segments, _info = self.model.transcribe(
            audio_array,
            beam_size=self.config.beam_size,
            best_of=1,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=False,
            initial_prompt=prompt,
            language=self.config.language,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        words: List[dict] = []
        for seg in segments:
            for w in (seg.words or []):
                text = self.vocabulary.apply_replacements(w.word.strip())
                if text:
                    words.append({"word": text, "start": w.start, "end": w.end})
        return words

    def _chunk_boundaries(self, audio_array: np.ndarray) -> List[Tuple[int, int]]:
        """Splits `audio_array` into (start, end) sample ranges covering the whole buffer,
        cutting only at real silence gaps (>= CHUNK_MIN_SILENCE_MS) via faster-whisper's own
        bundled Silero VAD, once accumulated speech would exceed CHUNK_TRIGGER_SEC. A short
        buffer (the overwhelming common case) returns a single range unchanged - this only
        engages for genuinely long recordings."""
        sr = 16000  # AudioRecorder always resamples to this before Whisper ever sees the buffer
        total_samples = len(audio_array)
        target_samples = int(CHUNK_TRIGGER_SEC * sr)
        if total_samples <= target_samples:
            return [(0, total_samples)]

        try:
            from faster_whisper.vad import get_speech_timestamps, VadOptions

            timestamps = get_speech_timestamps(
                audio_array, VadOptions(min_silence_duration_ms=CHUNK_MIN_SILENCE_MS), sampling_rate=sr,
            )
        except Exception as e:
            logger.warning(f"VAD chunk-boundary detection failed ({e}) - transcribing as one pass instead.")
            return [(0, total_samples)]

        if not timestamps:
            return [(0, total_samples)]  # no detected speech at all - let the single-pass no-VAD retry handle it

        boundaries: List[Tuple[int, int]] = []
        chunk_start = 0
        for i, seg in enumerate(timestamps):
            is_last_segment = i == len(timestamps) - 1
            if not is_last_segment and (seg["end"] - chunk_start) >= target_samples:
                # Cut at the midpoint of the silence gap after this segment - never
                # mid-speech, since that gap is exactly what VAD just confirmed is silence.
                cut = (seg["end"] + timestamps[i + 1]["start"]) // 2
                boundaries.append((chunk_start, cut))
                chunk_start = cut
        boundaries.append((chunk_start, total_samples))
        return boundaries

    def _run_transcription(
        self, audio_array: np.ndarray, biased_prompt: str, vad_filter: bool, vad_params: Optional[dict],
        condition_on_previous_text: bool = False,
    ):
        """One Whisper pass; consumes the full segment generator, never just the first segment.
        The three threshold kwargs below are faster-whisper's OWN default values
        (verified directly against its `WhisperModel.transcribe()` signature) -
        passed explicitly rather than left implicit so a future faster-whisper
        upgrade changing its defaults can't silently loosen these anti-
        hallucination gates without someone noticing the diff here."""
        segments, info = self.model.transcribe(
            audio_array,
            beam_size=self.config.beam_size,
            best_of=1,
            temperature=0.0,
            # word_timestamps=True (not the segment-level timestamps a plain
            # transcribe() call gives you) is required for the paragraph-pause
            # join below: verified empirically that faster-whisper's SEGMENT
            # timestamps are contiguous (one segment's start == the previous
            # segment's end) even across a real multi-second silence - only
            # the finer word-level alignment actually reflects real gaps.
            # transcribe_live_window() already relies on this same word-level
            # signal for its own pause detection.
            word_timestamps=True,
            vad_filter=vad_filter,
            vad_parameters=vad_params,
            initial_prompt=biased_prompt,
            language=self.config.language,  # pinned to 'en' by default; null = auto-detect
            condition_on_previous_text=condition_on_previous_text,
            compression_ratio_threshold=2.4,  # reject a segment that's too repetitive to be real speech
            log_prob_threshold=-1.0,          # reject a low-confidence (likely hallucinated) decode
            no_speech_threshold=0.6,          # reject a segment Whisper itself thinks is silence
        )
        # Join words with "\n\n" across a real mid-dictation pause (>=
        # PARAGRAPH_PAUSE_SEC) instead of the word's own natural leading
        # space - Wispr Flow-style paragraph breaks.
        parts: List[str] = []
        prev_end: Optional[float] = None
        for seg in segments:
            for w in seg.words:
                word_text = w.word  # faster-whisper already includes the natural leading space/attachment
                if prev_end is not None and w.start - prev_end >= PARAGRAPH_PAUSE_SEC:
                    parts.append("\n\n" + word_text.lstrip())
                else:
                    parts.append(word_text)
                prev_end = w.end
        return "".join(parts).strip(), info
