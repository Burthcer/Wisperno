"""
Interactive Subsystem Diagnostic Script for Wisperno.
Tests real hardware and OS interactions: Microphone -> Whisper STT -> LLM Transform -> Native Win32 Injection.
"""

import math
import os
import sys
import time
import numpy as np
from scipy import signal
import sounddevice as sd
import pyperclip

# Ensure CUDA runtime DLLs are found by ctypes on Windows
if sys.platform == "win32":
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.path.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

from faster_whisper import WhisperModel
from llama_cpp import Llama
from src.config import load_config
from src.injector import TextInjector


def test_mic():
    print("\n--- [1/4] TESTING MICROPHONE ---")
    print("Speak loudly into your mic for 3 seconds...")
    device_info = sd.query_devices(kind="input")
    sr = int(device_info["default_samplerate"])
    channels = int(device_info["max_input_channels"])
    print(f"Device: {device_info['name']} | Channels: {channels} | Native SR: {sr}")

    audio = sd.rec(int(3 * sr), samplerate=sr, channels=channels, dtype="float32")
    sd.wait()

    # Downmix multi-channel to mono
    mono = np.mean(audio, axis=1) if channels > 1 else audio.flatten()

    # Resample to 16kHz for Whisper if needed
    if sr != 16000 and len(mono) > 0:
        gcd_val = math.gcd(16000, sr)
        up = 16000 // gcd_val
        down = sr // gcd_val
        mono_16k = signal.resample_poly(mono, up, down).astype(np.float32)
    else:
        mono_16k = mono.astype(np.float32)

    rms = float(np.sqrt(np.mean(mono_16k**2)))
    peak = float(np.max(np.abs(mono_16k)))
    print(f"Captured: 3.0s | RMS: {rms:.5f} | Peak: {peak:.5f}")

    if rms < 0.003:
        print("❌ CRITICAL: Microphone captured pure silence or near-zero amplitude! Check Windows Mic Privacy settings.")
        return None
    print("✅ Microphone capture SUCCESSFUL.")
    return mono_16k


def test_stt(audio_data):
    print("\n--- [2/4] TESTING WHISPER STT ---")
    if audio_data is None:
        print("Skipping STT due to silent audio.")
        return None
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16", download_root="models")
    segments, _ = model.transcribe(audio_data, beam_size=1, vad_filter=True)
    text = " ".join([seg.text for seg in segments]).strip()
    print(f'Transcript: "{text}"')
    if not text:
        print("❌ CRITICAL: Whisper returned an empty string.")
        return None
    print("✅ Whisper STT SUCCESSFUL.")
    return text


def test_llm(raw_text):
    print("\n--- [3/4] TESTING LLM TRANSFORM ---")
    if not raw_text:
        raw_text = "um hello like i am testing this offline ai tool wait no dictation app"
    print(f'Input text: "{raw_text}"')
    llm = Llama(
        model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a text cleanup engine. Remove filler words and stutters. Fix punctuation. Output ONLY the polished text with no notes or extra commentary.",
            },
            {"role": "user", "content": raw_text},
        ],
        temperature=0.1,
        max_tokens=256,
    )
    polished = response["choices"][0]["message"]["content"].strip()
    print(f'Polished Output: "{polished}"')
    print("✅ LLM Transform SUCCESSFUL.")
    return polished


def test_paste(text):
    print("\n--- [4/4] TESTING OS CLIPBOARD & INJECTION (Win32 SendInput) ---")
    if not text:
        text = "Hello from Wisperno offline dictation test!"
    print("Switch to Notepad or any text field NOW! Pasting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"Injecting in {i}...", end="\r", flush=True)
        time.sleep(1)
    
    injector = TextInjector()
    success = injector.inject_text(text)
    if success:
        print("\n✅ Native Win32 SendInput Ctrl+V dispatched successfully.")
    else:
        print("\n❌ Failed to dispatch text injection.")


if __name__ == "__main__":
    audio = test_mic()
    transcript = test_stt(audio)
    polished = test_llm(transcript)
    test_paste(polished)
