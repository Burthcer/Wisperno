"""
Auto-Stop Silence Detection Verification for Wisperno.
Drives AudioRecorder._audio_callback() directly (the same production code
path sounddevice.InputStream calls) with real loud-then-quiet blocks, so this
exercises the actual live watchdog logic, not a re-implementation of it.

Run: python tests/test_auto_silence.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _feed_amplitude(recorder, seconds: float, amplitude: float, block_sec: float = 0.05) -> None:
    """Drives real time forward while feeding blocks at a specific RMS
    amplitude - auto-silence timing is wall-clock based (time.monotonic()),
    so this can't be faked with synthetic timestamps."""
    block_samples = max(1, int(recorder.native_sample_rate * block_sec))
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        block = (np.random.randn(block_samples, recorder.native_channels) * amplitude).astype(np.float32)
        recorder._audio_callback(block, block_samples, None, None)
        time.sleep(block_sec)


def _feed(recorder, seconds: float, loud: bool, block_sec: float = 0.05) -> None:
    """loud/near-silent convenience wrapper around _feed_amplitude - well
    above / well below min_energy_threshold (0.002 default)."""
    _feed_amplitude(recorder, seconds, 0.2 if loud else 0.0001, block_sec)


def test_auto_silence_fires_once_after_threshold() -> None:
    from src.config import load_config
    from src.audio_recorder import AudioRecorder

    logger.info("--- Testing AudioRecorder: auto-silence fires once, after real speech then a real pause ---")
    config = load_config()
    fired = []
    recorder = AudioRecorder(config=config.audio, silence_callback=lambda: fired.append(time.monotonic()))
    recorder.set_auto_silence_seconds(1)  # fast for the test; production default is 5s

    recorder._is_recording = True
    recorder._last_speech_time = time.monotonic()
    recorder._silence_fired = False
    t_start = time.monotonic()

    _feed(recorder, seconds=0.5, loud=True)
    assert not fired, "Auto-silence fired while speech was still active."

    _feed(recorder, seconds=1.6, loud=False)  # past the 1s threshold
    assert fired, "Auto-silence did not fire after the configured silence threshold elapsed."
    assert len(fired) == 1, f"Auto-silence fired more than once for a single recording: {len(fired)} times."

    # Keep feeding silence - must NOT fire again for the same recording.
    _feed(recorder, seconds=0.5, loud=False)
    assert len(fired) == 1, "Auto-silence fired a second time for the same recording."
    logger.success(f"PASS: fired exactly once, {fired[0] - t_start:.2f}s after recording started.")
    recorder._is_recording = False


def test_auto_silence_ignores_steady_ambient_noise() -> None:
    """Reproduces the reported defect: a room's steady ambient noise (fan
    whine, mic hiss) sitting above the static min_energy_threshold used to
    reset the silence timer on every single block, so auto-silence never
    fired no matter how long the user stayed quiet. Feeds a calibration
    window of that ambient noise, real speech, then the same ambient noise
    again - the adaptive noise-floor gate (AudioRecorder._audio_callback)
    must still let the silence timer accumulate and fire."""
    from src.config import load_config
    from src.audio_recorder import AudioRecorder

    logger.info("--- Testing AudioRecorder: adaptive noise floor ignores steady background noise ---")
    config = load_config()
    fired = []
    recorder = AudioRecorder(config=config.audio, silence_callback=lambda: fired.append(time.monotonic()))
    recorder.set_auto_silence_seconds(1)

    recorder._is_recording = True
    recorder._last_speech_time = time.monotonic()
    recorder._silence_fired = False
    recorder._noise_floor_rms = 0.0
    recorder._noise_floor_calibrated = False
    recorder._calibration_samples = []
    recorder._calibration_start = time.monotonic()

    # Just above the static default threshold (0.002) - the OLD code treated
    # this as continuous "speech" forever, since it never fell below 0.002.
    AMBIENT_NOISE_AMPLITUDE = 0.0028
    _feed_amplitude(recorder, seconds=0.3, amplitude=AMBIENT_NOISE_AMPLITUDE)  # covers the 0.25s calibration window
    _feed(recorder, seconds=0.5, loud=True)  # genuine speech - must still register above the adaptive gate
    assert not fired, "Auto-silence fired while speech was still active."

    _feed_amplitude(recorder, seconds=1.6, amplitude=AMBIENT_NOISE_AMPLITUDE)  # back to the same steady ambient noise
    assert fired, "Auto-silence did not fire through steady ambient background noise (adaptive gate regressed)."
    logger.success(f"PASS: adaptive noise floor ({recorder._noise_floor_rms:.5f} RMS) absorbed steady ambient noise; auto-silence still fired.")
    recorder._is_recording = False


def test_auto_silence_disabled_when_zero() -> None:
    from src.config import load_config
    from src.audio_recorder import AudioRecorder

    logger.info("--- Testing AudioRecorder: auto_silence_seconds=0 disables the watchdog entirely ---")
    config = load_config()
    fired = []
    recorder = AudioRecorder(config=config.audio, silence_callback=lambda: fired.append(True))
    recorder.set_auto_silence_seconds(0)

    recorder._is_recording = True
    recorder._last_speech_time = time.monotonic()
    recorder._silence_fired = False

    _feed(recorder, seconds=1.5, loud=False)
    assert not fired, "Auto-silence fired even though auto_silence_seconds was 0 (disabled)."
    logger.success("PASS: no auto-stop fires when the setting is disabled.")
    recorder._is_recording = False


def test_force_stop_active_recording_resets_both_trigger_kinds() -> None:
    from src.hotkey_manager import HotkeyManager

    logger.info("--- Testing HotkeyManager.force_stop_active_recording(): clears hold and tap bookkeeping ---")
    mgr = HotkeyManager()

    mgr._is_hold_active = True
    mgr._active_trigger = "hold"
    mgr.force_stop_active_recording()
    assert not mgr._is_hold_active and mgr._active_trigger is None, "Hold-to-talk state was not cleared."

    mgr._is_toggled_recording = True
    mgr._active_trigger = "tap"
    mgr.force_stop_active_recording()
    assert not mgr._is_toggled_recording and mgr._active_trigger is None, "Tap-to-toggle state was not cleared."

    logger.success("PASS: force_stop_active_recording() clears whichever trigger kind was active.")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO AUTO-STOP SILENCE DETECTION TEST SUITE")
    logger.info("=========================================================")
    test_auto_silence_fires_once_after_threshold()
    test_auto_silence_ignores_steady_ambient_noise()
    test_auto_silence_disabled_when_zero()
    test_force_stop_active_recording_resets_both_trigger_kinds()
    logger.success("=========================================================")
    logger.success(" ALL AUTO-SILENCE TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
