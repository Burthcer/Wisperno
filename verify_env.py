"""
Environment Verification Script for Wisperno.
Confirms PyTorch CUDA, GPU device capabilities, llama-cpp CUDA offload, and audio devices.
"""

import sys
from loguru import logger


def verify_python():
    logger.info(f"Python Version: {sys.version.split()[0]} ({'64-bit' if sys.maxsize > 2**32 else '32-bit'})")


def verify_cuda_torch():
    try:
        import torch
        logger.info(f"PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available in PyTorch: {cuda_available}")
        if cuda_available:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.success(f"GPU {i}: {device_name} ({total_mem_gb:.2f} GB VRAM)")
        else:
            logger.error("CUDA is NOT available in PyTorch!")
    except ImportError as e:
        logger.error(f"PyTorch not installed: {e}")


def verify_faster_whisper():
    try:
        import ctranslate2
        import faster_whisper
        logger.info(f"faster-whisper Version: {faster_whisper.__version__}")
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        logger.info(f"CTranslate2 CUDA Compute Types: {list(cuda_types)}")
        logger.success("faster-whisper and CTranslate2 are available.")
    except Exception as e:
        logger.error(f"faster-whisper / CTranslate2 check failed: {e}")


def verify_llama_cpp():
    try:
        import llama_cpp
        logger.info(f"llama-cpp-python Version: {llama_cpp.__version__}")
        # Test if GPU support is compiled in
        try:
            supports_gpu = llama_cpp.llama_supports_gpu_offload()
            logger.info(f"llama-cpp GPU Offload Supported: {supports_gpu}")
            if supports_gpu:
                logger.success("llama-cpp is compiled with GPU acceleration.")
            else:
                logger.warning("llama-cpp does NOT support GPU offload (running on CPU).")
        except AttributeError:
            logger.info("llama_supports_gpu_offload() not present; checking backend.")
    except Exception as e:
        logger.error(f"llama-cpp-python check failed: {e}")


def verify_audio():
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        logger.info(f"Total Audio Devices: {len(devices)}, Default Input Index: {default_in}")
        if default_in is not None and default_in >= 0:
            in_info = sd.query_devices(default_in, "input")
            logger.success(f"Default Microphone: '{in_info['name']}' (Channels: {in_info['max_input_channels']})")
        else:
            logger.warning("No default audio input device detected.")
    except Exception as e:
        logger.error(f"Audio device check failed: {e}")


def verify_clipboard_and_keys():
    try:
        import pyperclip
        import ctypes

        assert ctypes.windll.user32.GetAsyncKeyState is not None
        logger.success("pyperclip and Win32 GetAsyncKeyState are available.")
    except Exception as e:
        logger.error(f"Clipboard / Win32 key-state check failed: {e}")


def main():
    logger.info("=" * 60)
    logger.info("Wisperno Environment & Hardware Verification")
    logger.info("=" * 60)

    verify_python()
    verify_cuda_torch()
    verify_faster_whisper()
    verify_llama_cpp()
    verify_audio()
    verify_clipboard_and_keys()

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
