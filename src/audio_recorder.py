"""
In-memory Audio Capture Module for Wisperno.
Captures multi-channel audio at hardware-native sample rates, averages channels to mono float32,
and resamples in-memory to 16,000 Hz for Whisper with zero disk I/O.
"""

import math
import queue
import threading
import time
from typing import Callable, Optional, Tuple, Any
import numpy as np
from scipy import signal
import sounddevice as sd
from loguru import logger

from src.config import AudioConfig

# Auto-Stop Silence Detection tuning.
_NOISE_CALIBRATION_SEC = 0.25  # ponytail: assumes a near-silent mic lead-in; a
                                # user who starts talking in the very first 250ms
                                # will slightly inflate the floor - upgrade path:
                                # drop outlier-high calibration blocks if that
                                # turns out to matter in practice.
_NOISE_FLOOR_MULTIPLIER = 1.8


class AudioRecorder:
    """
    Thread-safe in-memory audio recorder compatible with multi-channel Realtek microphone arrays.
    """

    def __init__(
        self, config: Optional[AudioConfig] = None, level_callback: Optional[Callable[[float], None]] = None,
        silence_callback: Optional[Callable[[], None]] = None,
    ):
        self.config = config or AudioConfig()
        self.target_sample_rate = self.config.sample_rate  # 16000 Hz expected by Whisper
        self.device_index = self.config.device_index
        self.min_duration_sec = self.config.min_duration_sec
        self.min_energy_threshold = self.config.min_energy_threshold
        self.level_callback = level_callback  # optional: fired with live RMS per audio block, for UI waveforms
        # Auto-Stop Silence Detection: fired at most once per recording, from
        # the audio callback thread, when speech has been absent for
        # auto_silence_seconds. 0 disables it (manual stop only). Mutable
        # (not baked into a closure) so Settings can change it live.
        self.auto_silence_seconds = self.config.auto_silence_seconds
        self.silence_callback = silence_callback
        self._last_speech_time: float = 0.0
        self._silence_fired = False
        # Adaptive noise floor: a room's steady ambient noise (fan whine, mic
        # hiss) can sit above the static min_energy_threshold, which used to
        # keep resetting _last_speech_time forever and made auto-silence never
        # fire. Calibrated once per recording from its first _NOISE_CALIBRATION_SEC.
        self._noise_floor_rms = 0.0
        self._noise_floor_calibrated = False
        self._calibration_samples: list = []
        self._calibration_start: float = 0.0
        # The actual captured audio length from the last stop_recording() call (computed
        # from the full concatenated buffer, not wall-clock press-to-release time, which
        # runs ~0.3-0.5s long on this hardware from stream start/stop overhead) - callers
        # that need an accurate duration for history/display should read this, not time
        # the press/release themselves.
        self.last_duration_seconds: float = 0.0

        self._audio_queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._is_recording = False
        self._start_time: float = 0.0
        self._lock = threading.Lock()

        # Query hardware capabilities
        self.native_channels, self.native_sample_rate = self._detect_hardware_settings()

    def _detect_hardware_settings(self) -> Tuple[int, int]:
        """
        Query hardware properties to obtain native channels and sample rate.
        Prevents silent audio capture failures on Realtek 4-channel microphone arrays.
        """
        try:
            default_input = sd.default.device[0]
            chosen_device = self.device_index if self.device_index is not None else default_input
            device_info = sd.query_devices(chosen_device, "input")

            native_channels = int(device_info.get("max_input_channels", 1))
            if native_channels < 1:
                native_channels = 1

            native_sr = int(device_info.get("default_samplerate", 44100))
            if native_sr < 8000:
                native_sr = 16000

            logger.info(
                f"Audio Device Initialized: [{chosen_device}] {device_info['name']} "
                f"(Native: {native_channels}ch @ {native_sr}Hz -> Target: 1ch @ {self.target_sample_rate}Hz)"
            )
            return native_channels, native_sr
        except Exception as e:
            logger.warning(f"Could not query audio device properties: {e}. Defaulting to 1ch @ 44.1kHz.")
            return 1, 44100

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        """
        Callback for sounddevice.InputStream.
        Pushes multi-channel raw float32 audio chunks to the synchronized FIFO queue.
        """
        if status:
            logger.warning(f"Audio stream status flag: {status}")
        if self._is_recording:
            self._audio_queue.put(indata.copy())
            rms = float(np.sqrt(np.mean(np.square(indata))))
            if self.level_callback:
                try:
                    self.level_callback(rms)
                except Exception:
                    pass

            # Auto-Stop Silence Detection: a lightweight RMS energy check, not
            # full per-block neural VAD - real-time Silero VAD on every
            # 1024-sample block would add meaningful CPU overhead on this
            # native callback thread for a feature that only needs to notice
            # "speech, then quiet for N seconds", not classify every block
            # precisely. What WAS missing is an adaptive gate: the room's own
            # steady background noise (fan whine, hiss) can sit above the
            # static min_energy_threshold, which kept resetting
            # _last_speech_time forever and made auto-silence never fire.
            now = time.monotonic()
            if not self._noise_floor_calibrated:
                if (now - self._calibration_start) < _NOISE_CALIBRATION_SEC:
                    self._calibration_samples.append(rms)
                else:
                    self._noise_floor_rms = (
                        sum(self._calibration_samples) / len(self._calibration_samples)
                        if self._calibration_samples else 0.0
                    )
                    self._noise_floor_calibrated = True

            speech_gate = max(self.min_energy_threshold, self._noise_floor_rms * _NOISE_FLOOR_MULTIPLIER)
            if rms >= speech_gate:
                self._last_speech_time = now
            elif (
                self.auto_silence_seconds > 0
                and not self._silence_fired
                and self._last_speech_time > 0
                and (now - self._last_speech_time) >= self.auto_silence_seconds
            ):
                self._silence_fired = True  # once per recording - stop_recording() resets it on the next start
                if self.silence_callback:
                    try:
                        self.silence_callback()
                    except Exception:
                        pass

    def start_recording(self) -> bool:
        """
        Start non-blocking audio capture at hardware-native configuration.
        """
        with self._lock:
            if self._is_recording:
                logger.debug("AudioRecorder: start_recording called while already recording.")
                return False

            # Drain any stale audio chunks in queue
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                self._is_recording = True
                self._start_time = time.perf_counter()
                self._last_speech_time = time.monotonic()
                self._silence_fired = False
                self._noise_floor_rms = 0.0
                self._noise_floor_calibrated = False
                self._calibration_samples = []
                self._calibration_start = time.monotonic()

                self._stream = sd.InputStream(
                    samplerate=self.native_sample_rate,
                    channels=self.native_channels,
                    dtype="float32",
                    device=self.device_index,
                    callback=self._audio_callback,
                    blocksize=1024,
                )
                self._stream.start()
                logger.debug(
                    f"Audio capture started ({self.native_channels} channels @ {self.native_sample_rate}Hz)."
                )
                return True
            except Exception as e:
                self._is_recording = False
                logger.error(f"Failed to start audio stream: {e}")
                return False

    def stop_recording(self) -> Optional[np.ndarray]:
        """
        Stop audio capture, average multi-channel array into mono, resample to 16kHz,
        and perform energy/duration validation.
        """
        with self._lock:
            if not self._is_recording:
                return None

            self._is_recording = False
            raw_duration = time.perf_counter() - self._start_time
            # Reset before any of the early-return paths below (empty chunks, too
            # short, too quiet) - without this, a filtered-out recording left
            # last_duration_seconds holding the PREVIOUS successful recording's
            # value, which a caller reading it afterward would see as stale
            # (real value, wrong recording) rather than a correct "nothing here".
            self.last_duration_seconds = 0.0

            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning(f"Error closing audio stream: {e}")
                finally:
                    self._stream = None

        # Collect raw chunks from queue
        chunks = self._drain_queue()

        if not chunks:
            logger.debug("No audio chunks captured in buffer.")
            return None

        audio_16k = self._to_mono_16k(chunks)

        # 4. Calculate duration, RMS energy, and peak amplitude on the final 16kHz buffer
        duration_seconds = len(audio_16k) / float(self.target_sample_rate) if len(audio_16k) > 0 else 0.0
        rms_value = float(np.sqrt(np.mean(np.square(audio_16k)))) if len(audio_16k) > 0 else 0.0
        peak_value = float(np.max(np.abs(audio_16k))) if len(audio_16k) > 0 else 0.0
        self.last_duration_seconds = duration_seconds

        logger.info(
            f"[AUDIO] Duration: {duration_seconds:.2f}s | RMS: {rms_value:.5f} | Peak: {peak_value:.5f} "
            f"(Raw: {len(chunks)} chunks, {self.native_channels}ch @ {self.native_sample_rate}Hz -> 1ch @ 16kHz)"
        )

        # 5. Filter accidental clicks or silence - discard before ever invoking Whisper/LLM
        if duration_seconds < self.min_duration_sec:
            logger.info(
                f"[AUDIO] Ignored: duration ({duration_seconds:.3f}s) below minimum ({self.min_duration_sec}s)."
            )
            return None

        if rms_value < self.min_energy_threshold:
            logger.info(
                f"[AUDIO] Ignored: Signal below speech energy threshold "
                f"(RMS {rms_value:.5f} < {self.min_energy_threshold})."
            )
            return None

        return audio_16k

    def drain_available_audio(self) -> Optional[np.ndarray]:
        """
        Non-destructively pull whatever audio has arrived since the last call,
        WITHOUT stopping the stream or touching is_recording - for continuous
        streaming use (src/live_transcriber.py's Live Transcription mode),
        unlike stop_recording() which ends the whole recording. Returns None
        if nothing new is queued yet, or if not currently recording.
        """
        if not self._is_recording:
            return None
        chunks = self._drain_queue()
        if not chunks:
            return None
        return self._to_mono_16k(chunks)

    def _drain_queue(self) -> list:
        chunks = []
        while not self._audio_queue.empty():
            try:
                chunks.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    def _to_mono_16k(self, chunks: list) -> np.ndarray:
        """Shared by stop_recording() and drain_available_audio(): concatenate
        raw multi-channel chunks, average to mono, resample to 16kHz - the
        exact same conversion regardless of whether it runs once at the end of
        a recording or repeatedly mid-stream."""
        # 1. Concatenate into full multi-channel array: shape (N, Channels) or (N,)
        raw_data = np.concatenate(chunks, axis=0)

        # 2. Convert multi-channel input to mono float32 by averaging across all channels
        if raw_data.ndim > 1 and raw_data.shape[1] > 1:
            mono_data = np.mean(raw_data, axis=1, dtype=np.float32)
        else:
            mono_data = raw_data.flatten().astype(np.float32)

        # 3. Resample from native sample rate to 16,000 Hz if necessary
        if self.native_sample_rate != self.target_sample_rate and len(mono_data) > 0:
            gcd_val = math.gcd(self.target_sample_rate, self.native_sample_rate)
            up = self.target_sample_rate // gcd_val
            down = self.native_sample_rate // gcd_val
            return signal.resample_poly(mono_data, up, down).astype(np.float32)
        return mono_data

    def set_auto_silence_seconds(self, seconds: int) -> None:
        self.auto_silence_seconds = seconds

    @property
    def is_recording(self) -> bool:
        return self._is_recording
