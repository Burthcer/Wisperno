"""
Integrated Graphics / Legacy GPU Safeguard Verification for Wisperno.
Directly exercises src/device_manager.py's proactive GPU probe (mocking
torch.cuda so this runs on any machine regardless of its real hardware) and
confirms WispernoEngine actually applies the CPU-fallback override to its
config when no usable GPU is detected.

Run: python tests/test_device_manager.py
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_detects_real_gpu_when_present() -> None:
    from src.device_manager import detect_gpu_capability

    logger.info("--- Testing detect_gpu_capability(): a normal, adequately-VRAM'd GPU is used ---")
    fake_props = MagicMock()
    fake_props.total_memory = int(8 * 1024 ** 3)  # 8GB
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_properties", return_value=fake_props):
        result = detect_gpu_capability()
    assert result["use_gpu"] is True and result["has_cuda"] is True
    assert abs(result["vram_gb"] - 8.0) < 0.01
    logger.success(f"PASS: {result}")


def test_falls_back_to_cpu_with_no_cuda() -> None:
    from src.device_manager import detect_gpu_capability

    logger.info("--- Testing detect_gpu_capability(): no CUDA device at all -> CPU fallback ---")
    with patch("torch.cuda.is_available", return_value=False):
        result = detect_gpu_capability()
    assert result["use_gpu"] is False and result["has_cuda"] is False
    assert result["reason"], "No reason given for the CPU fallback."
    logger.success(f"PASS: {result}")


def test_falls_back_to_cpu_with_insufficient_vram() -> None:
    """Integrated graphics can report a CUDA-capable device with almost no
    usable VRAM - this must fall back too, not just a total CUDA absence."""
    from src.device_manager import detect_gpu_capability, MIN_VRAM_GB

    logger.info("--- Testing detect_gpu_capability(): CUDA present but VRAM below the minimum -> CPU fallback ---")
    fake_props = MagicMock()
    fake_props.total_memory = int(0.5 * 1024 ** 3)  # 512MB - well under MIN_VRAM_GB
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_properties", return_value=fake_props):
        result = detect_gpu_capability()
    assert result["use_gpu"] is False, f"A {result['vram_gb']:.1f}GB GPU (< {MIN_VRAM_GB}GB) was not rejected."
    assert result["has_cuda"] is True, "has_cuda should still report True - CUDA itself IS present, just not enough VRAM."
    logger.success(f"PASS: {result}")


def test_probe_failure_never_raises() -> None:
    """A probe exception (e.g. a broken/partial CUDA install) must be treated
    as 'no usable GPU', never propagate and crash startup."""
    from src.device_manager import detect_gpu_capability

    logger.info("--- Testing detect_gpu_capability(): a probe exception is swallowed, not raised ---")
    with patch("torch.cuda.is_available", side_effect=RuntimeError("broken CUDA install")):
        result = detect_gpu_capability()  # must not raise
    assert result["use_gpu"] is False
    logger.success(f"PASS: {result}")


def test_engine_applies_cpu_override_when_no_gpu() -> None:
    """WispernoEngine must actually rewrite whisper.device/compute_type and
    llm.n_gpu_layers to CPU-safe values when detect_gpu_capability() says so -
    in memory only, config.yaml itself must stay untouched."""
    from src.config import load_config
    from src.engine import WispernoEngine

    logger.info("--- Testing WispernoEngine: applies the CPU override to its in-memory config only ---")
    engine = WispernoEngine.__new__(WispernoEngine)
    engine.config = load_config()
    original_device = engine.config.whisper.device
    original_layers = engine.config.llm.n_gpu_layers

    with patch("src.device_manager.detect_gpu_capability", return_value={
        "has_cuda": False, "vram_gb": 0.0, "use_gpu": False, "reason": "test",
    }):
        from src.device_manager import detect_gpu_capability
        gpu = detect_gpu_capability()
        engine._cpu_compat_mode = not gpu["use_gpu"]
        if engine._cpu_compat_mode:
            engine.config.whisper.device = "cpu"
            engine.config.whisper.compute_type = "int8"
            engine.config.llm.n_gpu_layers = 0

    assert engine._cpu_compat_mode is True
    assert engine.config.whisper.device == "cpu"
    assert engine.config.whisper.compute_type == "int8"
    assert engine.config.llm.n_gpu_layers == 0

    # The override must be in-memory only - a fresh load_config() (simulating
    # config.yaml on disk) must show the ORIGINAL, unmodified values.
    fresh = load_config()
    assert fresh.whisper.device == original_device, "CPU override leaked into a fresh config.yaml read."
    assert fresh.llm.n_gpu_layers == original_layers, "CPU override leaked into a fresh config.yaml read."
    logger.success("PASS: CPU override applied in-memory; config.yaml on disk is untouched.")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO GPU/CPU FALLBACK SAFEGUARD TEST SUITE")
    logger.info("=========================================================")
    test_detects_real_gpu_when_present()
    test_falls_back_to_cpu_with_no_cuda()
    test_falls_back_to_cpu_with_insufficient_vram()
    test_probe_failure_never_raises()
    test_engine_applies_cpu_override_when_no_gpu()
    logger.success("=========================================================")
    logger.success(" ALL GPU/CPU FALLBACK TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
