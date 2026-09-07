# SKILL.md - Developer & System Operational Guide

This document contains persistent execution guidelines, hardware specifications, installation recipes, and solutions to known OS/GPU pitfalls for the Wisperno daemon.

---

## 1. Development & Execution Rules
- **Code Style & Formatting**: Strict adherence to PEP 8 standards with comprehensive Python 3.10+ type annotations (`typing`, `Optional`, `Union`, `Dict`, `Any`).
- **Structured Logging**: Use `loguru` (or formatted `logging`) with timestamps, log levels, and millisecond-accurate execution timers across all pipeline steps.
- **Thread Safety**: Audio recording runs on a dedicated thread with a synchronized FIFO queue (`queue.Queue`). All access to audio buffers must be thread-safe.
- **Zero Disk Latency**: Active audio buffers and intermediate strings must never touch disk during normal push-to-talk cycles. Use in-memory numpy float32 arrays and StringIO/BytesIO if needed.
- **Fail-Safe Operation**: If the LLM transformation fails or encounters an exception, the system must gracefully fall back to injecting the raw transcription rather than failing silently or crashing the daemon.

---

## 2. Hardware Profile & Target Environment
- **Operating System**: Microsoft Windows 11 64-bit (Build 22631+).
- **Target GPU**: NVIDIA GeForce RTX series (e.g., RTX 4070 Laptop GPU 8GB GDDR6 / RTX 30/40/50 Desktop GPUs).
- **Compute Stack**: NVIDIA CUDA 12.x / 13.x driver with cuDNN support.
- **Audio Capture**: Windows Core Audio (WASAPI / DirectSound / MME) via `sounddevice` input streams at 16,000 Hz, 1 channel (mono), 32-bit float.

---

## 3. CUDA & Environment Installation Recipes

### Virtual Environment Creation
Use Python 3.12 (standard for PyTorch CUDA & precompiled wheels):
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### PyTorch CUDA 12.x Installation
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### llama-cpp-python with CUDA Acceleration on Windows
Install precompiled CUDA wheels for Windows 64-bit and CUDA 12.x without requiring MSVC build tools:
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```
*(Or install via GitHub release assets for cu124/cu122/cu121 if needed).*

### Core Requirements
```powershell
pip install faster-whisper sounddevice soundfile numpy pynput pyperclip pyyaml huggingface_hub loguru scipy
```

---

## 4. Known Pitfalls & Engineered Workarounds

### Pitfall 1: `pynput` Modifier Key Release Bugs on Windows
- **Issue**: On Windows, when combinations like `Ctrl+Space` or `Alt+Key` are held, Windows API sometimes intercepts or suppresses the modifier `Key.release` event if focus shifts or another application hooks the keyboard.
- **Workaround**:
  - Implement a stateful key tracker in `hotkey_manager.py` that handles single-key hotkeys (like `Key.caps_lock`, `Key.f8`, `Key.scroll_lock`) or multi-key chords (`Ctrl+Space`).
  - Maintain a debounce window and active recording flag. If a release event is missed, provide a safety timeout or double-tap toggle fallback.

### Pitfall 2: Windows Audio Device Indexing in `sounddevice`
- **Issue**: Windows frequently re-indexes audio endpoint devices or selects virtual audio endpoints (e.g., monitor speakers, virtual cables) instead of the primary microphone when `device=None`.
- **Workaround**:
  - Automatically query `sounddevice.query_devices()` and select the default input device (`sounddevice.default.device[0]`).
  - Validate device channels, sample rate support (16kHz), and provide explicit device selection in `config.yaml`.

### Pitfall 3: Clipboard Race Conditions during Text Injection
- **Issue**: Calling `pyperclip.copy(text)` and immediately dispatching `Ctrl+V` often results in pasting the *old* clipboard content because the Windows clipboard manager (or active app) hasn't finished synchronizing the clipboard handle. Additionally, restoring the clipboard too soon overwrites the clipboard before the target app reads it.
- **Workaround**:
  - Follow the strict sequence in `injector.py`:
    1. Read and preserve `prev_clipboard = pyperclip.paste()`.
    2. Write new text: `pyperclip.copy(polished_text)`.
    3. Sleep `30ms` - `50ms` (allows OS clipboard sync).
    4. Simulate `Key.ctrl_l` down + `'v'` down -> `'v'` up -> `Key.ctrl_l` up via `pynput.keyboard.Controller`.
    5. Sleep `100ms` (allows target application window message loop to read paste).
    6. (Optional) Restore `pyperclip.copy(prev_clipboard)`.

### Pitfall 4: Hugging Face Model Download Timeouts
- **Issue**: Large model downloads (>1.5GB) over unstable connections can fail or hang during standard `requests` downloads.
- **Workaround**:
  - Use `huggingface_hub.hf_hub_download` with `resume_download=True` and `local_dir_use_symlinks=False` in `download_models.py`.
  - Check file size and SHA256 checksums if needed before proceeding.

### Pitfall 5: CTranslate2 / CUDA DLL Resolution on Windows
- **Issue**: `faster-whisper` depends on `ctranslate2`, which requires NVIDIA CUDA and cuDNN runtime DLLs (`cublas64_12.dll`, `cudnn64_9.dll` or `cudnn_ops_infer64_8.dll`).
- **Workaround**:
  - In `transcriber.py` or entry point, ensure PyTorch's `torch/lib` and NVIDIA CUDA path are added to Windows DLL search directory via `os.add_dll_directory()` if DLL load errors arise.
