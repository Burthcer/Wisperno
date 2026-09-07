"""
Assembles Final App/git versions/: a clean, portable-style distribution meant
to be attached to a GitHub Release (not committed to git history - it's still
a multi-GB runtime). Same exe/runtime bundling as package_portable.py's
_copy_exe_and_runtime(), reused rather than duplicated, but deliberately
DOESN'T bundle any model weights (models/ ships as an empty, documented
folder instead) - a GitHub Release download shouldn't force an 8GB+ transfer
before a user even knows which preset they want; download_models.py (or a
preset switch in Settings, which downloads on demand) fetches weights after
install instead. Also excludes the Inno Setup installer/.iss entirely, per
the explicit ask: this is the "run it anywhere" portable folder, not the
Windows-installer path.

Requires `python build_exe.py` to have already produced Final App/v{VERSION}/.

Usage: python scripts/package_git_release.py
"""

import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.__version__ import __version__  # noqa: E402
from package_portable import _copy_exe_and_runtime, BUILT_VERSION_DIR, BUILT_CONFIG_PATH  # noqa: E402

RELEASE_DIR = BASE_DIR / "Final App" / "git versions"

MODELS_MANIFEST = """\
# Model Weights (not included in this download)

This folder ships without any model weights - the runtime alone is already
several GB, and forcing an extra 5-10GB download before you've even chosen a
preset isn't a good first-run experience. Get the weights one of two ways:

## Option A: Run the downloader (recommended)

From this folder:

```
Wisperno.exe
```

On first launch with an empty `models/`, Wisperno logs a warning and falls
back to plain (un-polished) dictation - it still works, just without AI
cleanup. To get the full experience, fetch the models the app expects with
the project's downloader (needs the source checkout, not just this portable
folder - see INSTALL.md's "Building from source" section) or manually place
the files below.

## Option B: Manual download

Whisper speech-to-text (required for all presets):
- Model: `large-v3-turbo` (via `faster-whisper` / CTranslate2, auto-fetched
  from Hugging Face on first run if you have internet access on that machine -
  no manual step needed for this one).

Local LLM (pick ONE preset to match `config/config.yaml`'s `model_preset`):

| Preset | Hugging Face repo | Filename | Size | VRAM |
|---|---|---|---|---|
| **Turbo Flagship (default)** | `bartowski/Qwen2.5-1.5B-Instruct-GGUF` | `Qwen2.5-1.5B-Instruct-Q8_0.gguf` | ~1.6 GB | ~3.5 GB |
| Standard | `bartowski/Qwen2.5-3B-Instruct-GGUF` | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` | ~1.9 GB | ~3.7 GB |
| Flagship | `bartowski/Qwen2.5-3B-Instruct-GGUF` | `Qwen2.5-3B-Instruct-Q8_0.gguf` | ~3.3 GB | ~5.1 GB |
| Eco | `bartowski/Qwen2.5-3B-Instruct-GGUF` | `Qwen2.5-3B-Instruct-IQ4_XS.gguf` (rename to `wisperno-custom-v1.gguf`) | ~1.7 GB | ~3.6 GB |

`config/config.yaml` in this folder already ships pre-configured for the
default (Turbo Flagship) - the table above only matters if you switch
presets or the models/ folder needs re-populating.

Download the file for your chosen preset from Hugging Face and place it
directly in this `models/` folder, next to this file. Also grab the fallback
model used automatically if your primary file is missing:
`bartowski/Llama-3.2-3B-Instruct-GGUF` -> `Llama-3.2-3B-Instruct-Q4_K_M.gguf`.

Whichever preset you use, `config/config.yaml`'s `model_preset` /
`llm.model_path` must point at the matching filename - switch presets from
the app's own Settings screen rather than hand-editing this if you're not
sure, and it will tell you what to download.
"""

LICENSE_TEXT = """\
MIT License

Copyright (c) 2026 Wisperno

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

README_TEXT = f"""\
# Wisperno

**100% offline, GPU-accelerated push-to-talk dictation for Windows.**

Hold a hotkey, speak, release - Wisperno transcribes your voice locally
(`faster-whisper large-v3-turbo` on CUDA) and cleans it up with a local
quantized LLM before pasting the result into whichever app has focus. No
audio or text ever leaves the machine.

**Ships with Turbo Flagship as the default engine** - `Qwen2.5-1.5B-Instruct`
at ~91 tok/s decode, ~3.5GB combined VRAM with Whisper, and the fastest
measured latency of the four presets (see the table below). Switch presets
any time from Settings if you'd rather trade speed for the Flagship tier's
extra fidelity headroom.

## Core Features

- **Two dictation engines**: instant, zero-LLM Direct Dictation (rule-based
  formatting, sub-second latency) and on-demand LLM Polish for grammar/filler
  cleanup - pick per-trigger, not an all-or-nothing switch.
- **Sub-second latency** end to end on the default Turbo Flagship preset.
- **100% private, fully offline** - no network calls once models are on disk.
- **Four engine presets** trading VRAM for fidelity/speed (measured, not
  estimated - see `tests/benchmark_models.py`):

  | Preset | VRAM | Decode speed | Notes |
  |---|---|---|---|
  | **Turbo Flagship (default)** | ~3.5 GB | ~91 tok/s | Fastest tier, instant text replacement |
  | Eco | ~3.6 GB | ~80 tok/s | Compact footprint, Standard-level fidelity |
  | Standard | ~3.7 GB | ~72 tok/s | Balanced, previous default |
  | Flagship | ~5.1 GB | ~53 tok/s | Q8_0 weights + fp16 KV-cache, highest fidelity |

- **Contextual transform modes**: Polish, Prompt Engineer, Bullet Points, Raw
  verbatim, plus a user-defined Transforms hub for custom prompts.
- **Vocabulary/dictionary** biasing for names and jargon Whisper wouldn't
  otherwise get right.

## Default Hotkeys

| Hotkey | Action |
|---|---|
| `Ctrl+Alt` | Tap to toggle Direct Dictation (tap to start, tap to stop) |
| `Ctrl+Space` | Hold to talk (disabled by default - enable in Settings) |
| `Ctrl+Shift+B` | Cycle active transformation mode |
| `Ctrl+,` | Open Settings & Dictionary |
| `Ctrl+Shift+H` | Toggle the dashboard window |

Any transform's own hotkey also polishes the current text selection if
nothing is spoken - a separate "Polish Selected Text" shortcut isn't needed.

## Quick Start

See [INSTALL.md](INSTALL.md) for full setup steps. Short version:

```
Wisperno.exe
```

First launch takes ~20-30s while Windows SmartScreen scans the new,
unsigned binary - every launch after is fast (a few seconds).

## Requirements

See [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) - short version: Windows
11 64-bit, an NVIDIA GPU with 4GB+ VRAM (2GB minimum triggers an automatic
CPU-compatibility fallback), ~8.5GB free disk space.

## License

[MIT](LICENSE) - v{__version__}.
"""

INSTALL_TEXT = """\
# Installing Wisperno (Portable)

This folder is a self-contained, portable Wisperno build - no installer, no
admin rights, nothing written outside this folder except your per-user
config database (`%APPDATA%\\Wisperno\\wisperno.db`).

## 1. Prerequisites

- **Windows 11** (64-bit). Windows 10 64-bit likely works but isn't the
  primary tested target.
- **NVIDIA GPU + driver** with CUDA support, 4GB+ VRAM recommended (2GB is
  the hard minimum before Wisperno automatically falls back to a slower
  CPU-only mode - see SYSTEM_REQUIREMENTS.md). Install the latest driver from
  https://www.nvidia.com/Download/index.aspx - the CUDA runtime itself is
  already bundled in this folder's `_internal\\`, you only need the driver.
- No Python install needed - this is a compiled, self-contained `.exe`.

## 2. Directory placement

Put this whole folder anywhere with write access (Desktop, a games/apps
drive, a USB drive for a fully portable setup) - it does not need to be
"installed" to a specific system location. Avoid paths with unusual
characters; a normal folder name is safest.

## 3. Get the model weights

This distribution ships WITHOUT model weights to keep the download small -
see `models/MODELS.md` for exactly what to fetch and where to put it, or
just launch `Wisperno.exe` once and read the log's warning if you skip this
step (dictation still works without an LLM, just without AI polish/cleanup).

## 4. Run it

```
Wisperno.exe
```

A small pill indicator appears near the bottom of your screen once ready
(a few seconds after Whisper/LLM load). Hold/tap `Ctrl+Alt` to dictate.

## 5. CPU-only fallback (no compatible GPU)

If Wisperno detects no CUDA-capable GPU (or under ~2GB VRAM), it
automatically switches to CPU inference for that launch - slower, but still
functional, and never silently corrupts your saved GPU preset for later.
A toast on the floating pill confirms when this happens.

## 6. Building from source (optional)

Only needed if you want to modify Wisperno or run `download_models.py`
directly instead of a manual download:

```
git clone <this repository>
cd Wisperno
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt
python download_models.py
python main.py
```

## Uninstalling

Delete the folder. Your dictation history/dictionary/settings live in
`%APPDATA%\\Wisperno\\` - delete that too for a full removal.
"""

SYSTEM_REQUIREMENTS_TEXT = """\
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
"""


def stage_release() -> Path:
    if not BUILT_VERSION_DIR.exists():
        print(f"[ERROR] '{BUILT_VERSION_DIR}' not found - run `python build_exe.py` first.")
        sys.exit(1)

    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True)

    print("Copying Wisperno.exe + PyInstaller runtime dependencies...")
    _copy_exe_and_runtime(RELEASE_DIR)
    # build_exe.py's generic README.txt is superseded by this script's own
    # README.md/INSTALL.md below - drop it so the release folder doesn't ship two.
    (RELEASE_DIR / "README.txt").unlink(missing_ok=True)

    print("Writing config template (from the built app's own config)...")
    dest_config = RELEASE_DIR / "config"
    dest_config.mkdir(exist_ok=True)
    shutil.copy2(BUILT_CONFIG_PATH, dest_config / "config.yaml")
    shutil.copy2(BUILT_VERSION_DIR / "config" / "dictionary.json", dest_config / "dictionary.json")

    print("Copying assets (icons)...")
    shutil.copytree(BUILT_VERSION_DIR / "assets", RELEASE_DIR / "assets", dirs_exist_ok=True)

    print("Writing empty models/ scaffold + manifest (weights intentionally excluded)...")
    dest_models = RELEASE_DIR / "models"
    dest_models.mkdir(exist_ok=True)
    (dest_models / ".gitkeep").write_text("", encoding="utf-8")
    (dest_models / "MODELS.md").write_text(MODELS_MANIFEST, encoding="utf-8")

    (RELEASE_DIR / "logs").mkdir(exist_ok=True)
    (RELEASE_DIR / "logs" / ".gitkeep").write_text("", encoding="utf-8")

    print("Writing documentation (README / INSTALL / SYSTEM_REQUIREMENTS / LICENSE)...")
    (RELEASE_DIR / "README.md").write_text(README_TEXT, encoding="utf-8")
    (RELEASE_DIR / "INSTALL.md").write_text(INSTALL_TEXT, encoding="utf-8")
    (RELEASE_DIR / "SYSTEM_REQUIREMENTS.md").write_text(SYSTEM_REQUIREMENTS_TEXT, encoding="utf-8")
    (RELEASE_DIR / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")

    return RELEASE_DIR


def main() -> None:
    print("=" * 70)
    print(f" Staging GitHub release distribution: v{__version__}")
    print("=" * 70)
    release_dir = stage_release()
    size_gb = sum(f.stat().st_size for f in release_dir.rglob("*") if f.is_file()) / 1e9
    print(f"\nRelease folder ready: {release_dir} ({size_gb:.2f} GB, no model weights)")
    print("=" * 70)


if __name__ == "__main__":
    main()
