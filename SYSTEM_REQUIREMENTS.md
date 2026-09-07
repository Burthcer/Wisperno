# System Requirements

## Operating System

- Windows 11, 64-bit (primary target).
- Windows 10, 64-bit: expected to work, not the primary tested platform.

## GPU / VRAM Tiers

Wisperno runs Whisper `large-v3-turbo` (STT) and a quantized local LLM
(polish/transform) simultaneously on the GPU. Combined VRAM by preset,
measured directly via `nvidia-smi`/`pynvml` (see `tests/benchmark_models.py`
and `handoff.md`), not estimated:

| Preset | Measured VRAM | Minimum GPU VRAM to run comfortably |
|---|---|---|
| Turbo Flagship (default) | ~3.5 GB | 4GB |
| Eco | ~3.6 GB | 4GB |
| Standard | ~3.7 GB | 4GB |
| Flagship | ~5.1 GB | 6GB |

| VRAM available | Recommended preset |
|---|---|
| 6GB+ | Any preset, including Flagship |
| 4-5GB | Turbo Flagship, Eco, or Standard (default: Turbo Flagship) |
| 2-3GB | Turbo Flagship only, minimal headroom - close other GPU apps |
| < 2GB / no CUDA GPU | CPU-compatibility fallback (automatic) - see below |

Any NVIDIA GPU with a current driver and CUDA support works (GeForce GTX
10-series and newer, RTX series, or equivalent Quadro/professional cards).

## CPU-Only Fallback

`src/device_manager.py` probes VRAM once at startup. Below ~2GB VRAM (or no
CUDA device at all - integrated graphics, non-NVIDIA GPUs), Wisperno
switches Whisper to `device="cpu"`/`compute_type="int8"` and the LLM to
`n_gpu_layers=0` for that launch only - noticeably slower (multi-second
transcription/polish instead of sub-second), but functional. Requires a
modern CPU with **AVX2** support (any x86-64 CPU from roughly the last
decade). This override is never written back to `config.yaml`.

## Disk Space

- ~8.5GB total for the app + all model weights (Whisper + one LLM preset).
- The portable download itself (no models) is a few GB (PyTorch/CUDA/
  CTranslate2 runtime).

## RAM

- 8GB system RAM minimum, 16GB recommended (leaves headroom for whatever
  else is running alongside the CPU-side portion of inference).

## Microphone

- Any Windows-recognized input device. Multi-channel arrays (e.g. laptop
  4-channel Realtek arrays) are downmixed and resampled to 16kHz mono
  automatically - no special configuration needed.
