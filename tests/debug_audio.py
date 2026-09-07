"""
Isolated diagnostic for the microphone -> Whisper pipeline, bypassing
Wisperno's own code entirely (raw sounddevice + faster_whisper calls) so a
regression in src/audio_recorder.py or src/transcriber.py can't hide the
root cause of a "no transcription" report.

Run: python tests/debug_audio.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from src.config import get_base_dir  # noqa: E402

# faster-whisper/huggingface_hub probe the Hub for a revision check even on
# fully-cached weights unless told not to - without this, this standalone
# script hangs/fails exactly like the ~173s stall documented in handoff.md
# Round 8, or (with local_files_only=True but no matching download_root)
# fails outright with LocalEntryNotFoundError because it looks in the
# default ~/.cache/huggingface location instead of Wisperno's models/ folder.
import os  # noqa: E402
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

print("=== 1. HARDWARE QUERY ===")
device_info = sd.query_devices(kind='input')
sr = int(device_info['default_samplerate'])
channels = int(device_info['max_input_channels'])
print(f"Default Input Device: {device_info['name']}")
print(f"Native Channels: {channels} | Native Samplerate: {sr}")

print("\n=== 2. RECORDING 3 SECONDS OF LIVE AUDIO ===")
print(">>> SPEAK CLEARLY INTO YOUR MICROPHONE NOW <<<")
raw_audio = sd.rec(int(3 * sr), samplerate=sr, channels=channels, dtype='float32')
sd.wait()

print("\n=== 3. AUDIO BUFFER ANALYSIS ===")
print(f"Raw shape: {raw_audio.shape} | dtype: {raw_audio.dtype}")
if channels > 1:
    mono_audio = np.mean(raw_audio, axis=1)
else:
    mono_audio = raw_audio.flatten()

peak = float(np.max(np.abs(mono_audio)))
rms = float(np.sqrt(np.mean(mono_audio**2)))
print(f"Peak Amplitude: {peak:.5f} | RMS Energy: {rms:.5f}")

if rms < 0.002:
    print("XX ERROR: Audio buffer is near zero/pure silence! Hardware stream failed or mic is muted.")
else:
    print("OK: Microphone capture SUCCESSFUL.")

print("\n=== 4. RESAMPLING & WHISPER STT TEST ===")
from scipy.signal import resample_poly
from math import gcd
g = gcd(16000, sr)
audio_16k = resample_poly(mono_audio, 16000 // g, sr // g).astype(np.float32)
print(f"Resampled to 16kHz: {audio_16k.shape} samples")

print("Loading Whisper model...")
# download_root must point at Wisperno's own models/ folder (src/config.py's
# get_base_dir()/"models") - without it, local_files_only=True looks in the
# default HF cache location instead and raises LocalEntryNotFoundError even
# though the model IS present, just somewhere else. This is the exact thing
# to check first if this script reports Whisper "won't load" at all.
model = WhisperModel(
    "large-v3-turbo", device="cuda", compute_type="int8_float16",
    local_files_only=True, download_root=str(get_base_dir() / "models"),
)
segments, info = model.transcribe(audio_16k, beam_size=1, vad_filter=False)
text = " ".join([seg.text.strip() for seg in segments if seg.text])
print(f'Raw Whisper Output (vad_filter=False): "{text}"')

segments_vad, _ = model.transcribe(audio_16k, beam_size=1, vad_filter=True)
text_vad = " ".join([seg.text.strip() for seg in segments_vad if seg.text])
print(f'Whisper Output (vad_filter=True): "{text_vad}"')
