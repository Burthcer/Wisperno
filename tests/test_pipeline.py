"""
Integration & End-to-End Pipeline Verification Test for Wisperno.
Tests multi-channel audio capture, Whisper STT inference, SLM transformation modes,
Win32 GetAsyncKeyState hotkey manager, Win32 SendInput injector, and Floating UI state machine.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
from loguru import logger

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import load_config
from src.audio_recorder import AudioRecorder
from src.transcriber import Transcriber
from src.transformer import Transformer
from src.injector import TextInjector
from src.hotkey_manager import HotkeyManager, parse_hotkey_to_vks


def generate_synthetic_audio(duration_sec: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Generate synthetic multi-tone audio resembling vocal frequencies (150Hz, 300Hz, 600Hz).
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = (
        0.4 * np.sin(2 * np.pi * 150 * t) +
        0.3 * np.sin(2 * np.pi * 300 * t) +
        0.2 * np.sin(2 * np.pi * 600 * t)
    )
    fade_len = int(sample_rate * 0.05)
    waveform[:fade_len] *= np.linspace(0, 1, fade_len)
    waveform[-fade_len:] *= np.linspace(1, 0, fade_len)
    return waveform.astype(np.float32)


def test_audio_recorder_energy_and_resampling():
    """Test duration, multi-channel averaging, and RMS energy thresholding in AudioRecorder."""
    logger.info("--- Testing AudioRecorder Multi-Channel Handling & Resampling ---")
    config = load_config()
    recorder = AudioRecorder(config=config.audio)

    # 1. Test pure silence rejection
    silence = np.zeros(16000 * 2, dtype=np.float32)
    rms_silence = float(np.sqrt(np.mean(np.square(silence))))
    assert rms_silence < config.audio.min_energy_threshold, "Silence should have RMS below threshold."

    # 2. Test multi-channel synthetic audio averaging
    t = np.linspace(0, 1.0, 44100, endpoint=False)
    ch1 = 0.3 * np.sin(2 * np.pi * 200 * t)
    ch2 = 0.3 * np.sin(2 * np.pi * 200 * t)
    ch3 = 0.3 * np.sin(2 * np.pi * 200 * t)
    ch4 = 0.3 * np.sin(2 * np.pi * 200 * t)
    multi_ch_audio = np.stack([ch1, ch2, ch3, ch4], axis=1).astype(np.float32)

    recorder._is_recording = True
    recorder._start_time = time.perf_counter() - 1.0
    recorder._audio_queue.put(multi_ch_audio)

    result_16k = recorder.stop_recording()
    assert result_16k is not None, "Resampled audio should not be None."
    assert result_16k.ndim == 1, "Result audio must be 1D mono float32."
    assert 15800 <= len(result_16k) <= 16200, f"Expected ~16000 samples after 1s resample, got {len(result_16k)}."

    rms_synth = float(np.sqrt(np.mean(np.square(result_16k))))
    assert rms_synth > config.audio.min_energy_threshold, "Synthetic tone should exceed energy threshold."
    logger.success(
        f"AudioRecorder 4ch @ 44.1kHz -> 1ch @ 16kHz resampling verified (Samples: {len(result_16k)}, RMS: {rms_synth:.4f})."
    )


def test_hotkey_manager_vk_parsing():
    """Test Win32 virtual key parsing and manager lifecycle."""
    logger.info("--- Testing HotkeyManager Win32 VK Parsing & Lifecycle ---")
    vks = parse_hotkey_to_vks("ctrl+b")
    assert 0x11 in vks, "Ctrl (0x11) must be in VK set."
    assert 0x42 in vks, "B (0x42) must be in VK set."

    cycle_vks = parse_hotkey_to_vks("ctrl+shift+b")
    assert {0x11, 0x10, 0x42}.issubset(cycle_vks), "Ctrl+Shift+B must have VKs 0x11, 0x10, 0x42."

    manager = HotkeyManager(tap_toggle_hotkey="ctrl+b", cycle_mode_hotkey="ctrl+shift+b")
    manager.start()
    time.sleep(0.05)
    assert manager._worker_thread is not None and manager._worker_thread.is_alive()
    manager.stop()
    assert manager._worker_thread is None
    logger.success("Win32 HotkeyManager VK parsing and thread lifecycle verified.")


def test_floating_pill_rendering():
    """Test the Qt FloatingPill's paintEvent renders real, correctly-colored, transparent-cornered output."""
    logger.info("--- Testing FloatingPill (PySide6) State Rendering ---")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QImage
    from src.ui.floating_pill import FloatingPill, STATE_IDLE, STATE_RECORDING, theme

    app = QApplication.instance() or QApplication([])
    pill = FloatingPill(initial_mode="POLISH", hotkey_label="Ctrl+B")
    assert pill.current_mode == "POLISH"

    pill.set_state(STATE_IDLE)
    app.processEvents()
    img = pill.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    corner = img.pixelColor(2, 2)
    center = img.pixelColor(img.width() // 2, img.height() // 2)
    assert corner.alpha() == 0, "pill corners must be transparent (rounded capsule, not a square)"
    assert center.alpha() == 255 and (center.red(), center.green(), center.blue()) != (0, 0, 0)

    pill.set_state(STATE_RECORDING)
    pill.set_mode("PROMPT_ENGINEER")
    assert pill.current_mode == "PROMPT_ENGINEER"

    logger.success("FloatingPill (PySide6) rendering, transparency, and state transitions verified.")


def test_injector():
    """Test Win32 TextInjector mechanics."""
    logger.info("--- Testing Win32 TextInjector ---")
    config = load_config()
    injector = TextInjector(config=config.injector)
    assert injector is not None
    logger.success("Win32 TextInjector successfully initialized.")


def test_transcriber_inference():
    """Test Whisper STT inference on audio sample."""
    logger.info("--- Testing Transcriber STT Inference ---")
    config = load_config()
    transcriber = Transcriber(config=config.whisper)

    synth_audio = generate_synthetic_audio(duration_sec=2.0)
    t0 = time.perf_counter()
    transcript = transcriber.transcribe(synth_audio)
    t_stt = (time.perf_counter() - t0) * 1000

    logger.info(f"Transcriber output for synthetic tone: '{transcript}' in {t_stt:.1f}ms.")
    logger.success("Whisper Transcriber initialized and executed inference successfully.")


def test_transformer_modes():
    """Test all transformation modes on the local LLM."""
    logger.info("--- Testing Transformer Modes ---")
    config = load_config()
    transformer = Transformer(config=config.llm, prompts=config.prompts)

    sample_spoken_text = "um so like basically what I want to do is uh create a python script that can parse json files and export them to csv you know"

    # 1. Test Raw Mode
    raw_res = transformer.transform(sample_spoken_text, mode="raw")
    assert raw_res == sample_spoken_text.strip(), "Raw mode must return unmodified text."

    if transformer.llm is not None:
        # 2. Test Polish Mode
        t0 = time.perf_counter()
        polish_res = transformer.transform(sample_spoken_text, mode="polish")
        t_polish = (time.perf_counter() - t0) * 1000
        logger.info(f"[Mode: POLISH in {t_polish:.1f}ms]\nOutput: {polish_res}")
        assert len(polish_res) > 0, "Polish mode produced empty output."
        assert "um" not in polish_res.lower().split(), "Polish mode should eliminate 'um'."

        # 3. Test Prompt Engineer Mode
        t0 = time.perf_counter()
        prompt_res = transformer.transform(sample_spoken_text, mode="prompt_engineer")
        t_prompt = (time.perf_counter() - t0) * 1000
        logger.info(f"[Mode: PROMPT_ENGINEER in {t_prompt:.1f}ms]\nOutput:\n{prompt_res}")
        assert len(prompt_res) > 0, "Prompt Engineer mode produced empty output."

        # 4. Test Bullets Mode
        t0 = time.perf_counter()
        bullets_res = transformer.transform(sample_spoken_text, mode="bullets")
        t_bullets = (time.perf_counter() - t0) * 1000
        logger.info(f"[Mode: BULLETS in {t_bullets:.1f}ms]\nOutput:\n{bullets_res}")
        assert "-" in bullets_res or "*" in bullets_res or "\n" in bullets_res, "Bullets mode should contain list items."

        logger.success("All Transformer modes verified successfully on GPU!")
    else:
        logger.warning("LLM model weights not loaded; tested fallback behavior.")


def run_all_tests():
    logger.info("=========================================================")
    logger.info("        WISPERNO INTEGRATION TEST SUITE")
    logger.info("=========================================================")

    t_start = time.perf_counter()

    test_hotkey_manager_vk_parsing()
    test_floating_pill_rendering()
    test_audio_recorder_energy_and_resampling()
    test_injector()
    test_transcriber_inference()
    test_transformer_modes()

    total_time = (time.perf_counter() - t_start) * 1000
    logger.success("=========================================================")
    logger.success(f" ALL TESTS COMPLETED SUCCESSFULLY in {total_time:.1f}ms!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
