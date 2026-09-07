# Wisperno

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Windows%2011-0078D6?logo=windows&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-required-76B900?logo=nvidia&logoColor=white)
![Offline](https://img.shields.io/badge/network%20calls-zero-success)
![License](https://img.shields.io/badge/license-MIT-green)

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
- **Live Transcription** (`Ctrl+Shift+L`): a separate streaming mode for
  meetings/lectures/calls - continuous Whisper transcription re-decoded every
  0.6s against the entire unconfirmed audio buffer (never a trailing slice -
  an earlier sliding-window design was found by testing to silently drop
  real content, and was replaced), shown in a floating overlay with Pause /
  Stop & Save / Discard. Measured **RTF ~0.074** (~13.5x faster than
  real-time) with no second model loaded - see
  [LIVE_TRANSCRIPTION_ARCHITECTURE.md](LIVE_TRANSCRIPTION_ARCHITECTURE.md)
  for the full pipeline, including the content-loss bug that shaped the
  current design.
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
| `Ctrl+Shift+L` | Start/stop a Live Transcription session |
| `Alt+C` (alias `Alt+P`) | Polish - the default transform |
| `Alt+X` | Prompt Engineer transform |
| `Alt+V` | Detailed Code Fix Prompt transform |
| `Ctrl+Shift+B` | Cycle active transformation mode |
| `Ctrl+,` | Open Settings & Dictionary |
| `Ctrl+Shift+H` | Toggle the dashboard window |

Any transform's own hotkey also polishes the current text selection if
nothing is spoken - a separate "Polish Selected Text" shortcut isn't needed.
Additional transforms get their own configurable shortcut from the
Transforms hub.

## What's New

Summarized from this build's own update notes, verified against source
where checkable (e.g. the hotkeys above are read from `src/database.py`/
`src/engine.py`, not copied from prose):

- **Real-Time Streaming Live Transcription** - a whole new mode (`Ctrl+Shift+L`),
  separate from push-to-talk dictation. See the Core Features entry above
  and [LIVE_TRANSCRIPTION_ARCHITECTURE.md](LIVE_TRANSCRIPTION_ARCHITECTURE.md)
  for the full design, including a real content-loss bug found by its own
  test suite and fixed before shipping.
- **Wispr Flow parity improvements**: pause-based paragraph splitting (a
  1.3s+ mid-dictation pause starts a new paragraph, using Whisper's
  word-level timestamps - segment-level timestamps were tested and found
  unable to detect the pause at all), a zero-LLM rule formatter
  (`src/direct_formatter.py`) for pronoun capitalization/filler
  stripping/list detection in well under 10ms, spoken command parsing
  ("slash compact" -> `/compact`), and expanded vocabulary biasing
  (now includes "Claude Code" and a "screenshot" bias entry).
- **UI redesign**: a jet-black/purple visual pass across the whole dashboard
  (`src/ui/theme.py` - `#08080C` canvas, `#A855F7` accent), a redesigned
  History tab (word/time/WPM/dictation-count summary row, search, filter
  pills), a Hardware Telemetry tab showing GPU VRAM and system RAM with
  dual-layer attribution (Wisperno's own usage vs. total system usage), and
  a floating pill with a live mic-driven waveform. See Interface &
  Ergonomics below.
- **Windows integration fixes**: correct taskbar icon/grouping via an
  explicit `AppUserModelID`, no startup hang (heavy model/device loading
  moved off the UI thread), and an installer that upgrades in place instead
  of duplicating the Start Menu entry.

## Interface & Ergonomics

No screenshots ship in this repo yet. A prior round embedded UI images that
turned out to include unreviewed personal content, so they were removed
outright rather than swapped for something "safer" - see `HANDOFF.md` for
the incident record. A real, deliberately-captured and reviewed screenshot
(cropped to just the app window, no desktop/taskbar/background) can be
added by hand later; until then, the dashboard's tab layout is
`History | Transforms | Dictionary | Snippets | Settings | System Usage`,
and the floating status pill is the only UI element visible outside the
dashboard window during normal use.

## Quick Start

This repository is source only (see [Why no .exe here](INSTALL.md#why-no-exe-here)).
See [INSTALL.md](INSTALL.md) for full setup steps. Short version:

```
git clone https://github.com/Burthcer/Wisperno.git
cd Wisperno
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python download_models.py
python main.py
```

## Requirements

See [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) - short version: Windows
11 64-bit, an NVIDIA GPU with 4GB+ VRAM (2GB minimum triggers an automatic
CPU-compatibility fallback), ~8.5GB free disk space.

## Measured Performance

Not projected - run this session on the actual shipping hardware target
(RTX 4070 Laptop GPU, 8GB VRAM) via `tests/benchmark_models.py`,
`tests/test_startup_time.py`, and a real (TTS-generated, not synthetic-tone)
speech sample transcribed end to end through `faster-whisper`. Reproduce it
yourself with the same two scripts - both ship in this repo.

**All four presets, back-to-back, Whisper + LLM loaded together** (combined
GPU memory via `pynvml`, process RAM via `psutil`, decode speed and 100-word
latency via streamed token timing, fidelity = output/input word-count ratio
on a 250-word rambling sample, run through the real conversational-trap
guardrail suite):

| Preset | VRAM | System RAM | Decode Speed | 100-word Latency | Fidelity | Guardrail |
|---|---|---|---|---|---|---|
| **Turbo Flagship (default)** | 3,493 MB | 2,306 MB | 89.3 tok/s | 1.04s | 98.0% | PASS (5/5 adversarial prompts) |
| Eco | 3,621 MB | 2,315 MB | 71.8 tok/s | 1.24s | 97.6% | PASS |
| Standard | 3,693 MB | 2,551 MB | 69.3 tok/s | 1.34s | 99.2% | PASS |
| Flagship | 5,121 MB | 3,865 MB | 52.8 tok/s | 1.77s | 99.2% | PASS |

Every preset lands under the 5GB system RAM budget with over 2GB of
headroom. Only Flagship exceeds the 4GB VRAM target (5.1GB) - a disclosed,
deliberate trade-off for its near-lossless Q8_0 weights, not an oversight;
the shipped default (Turbo Flagship) stays at 3.5GB. The guardrail column
isn't a formality: this run's raw Turbo Flagship model actually drifted on
one adversarial prompt ("can you explain python pointers to me") into a
291-word off-topic explanation - `sanity_check()` caught it and the
delivered output still passed clean. That's the safety net working, not a
result being hidden.

**Cold start** (`tests/test_startup_time.py`, all 5 assertions passing):
Whisper (`large-v3-turbo`, CUDA, int8_float16) loads in **2.8-5.2s** across
repeated runs (well under the 10s regression ceiling this test guards - a
prior bug made this take ~173s via an unwanted Hugging Face network probe).
The LLM loads from its local GGUF in **~1.1-1.2s**, consistent across all
four presets.

**Real speech transcription (not a synthetic tone):** a genuine
TTS-generated 9.75s spoken sample was transcribed by the actual
`faster-whisper` pipeline in 0.945s - **RTF 0.097** (~10.3x faster than
real-time) - and came back **word-for-word verbatim** against the known
script, punctuation and filler words included, before any LLM polish is
even applied.

**Resource ceiling during active inference** (paired `psutil`/`pynvml`
sampling through one live LLM transform call): peak process CPU 92.4%,
peak GPU compute utilization 79% - consistent with the 77% observed during
the full four-preset run above.

## How It Compares

Wispr Flow (the product Wisperno takes after) is cloud-only: it has no
offline mode at any pricing tier, and every word dictated is sent to a
subprocessor stack (OpenAI, Anthropic, Baseten, AWS) for transcription and
cleanup, even under its "zero data retention" privacy setting. A
self-hosted full Whisper (`large-v3`) is private but heavy - roughly 10GB
VRAM to run comfortably. Wisperno's own numbers below are the ones already
measured and shipped in `config/config.yaml`/`SYSTEM_REQUIREMENTS.md`, not
projections.

| | Wispr Flow | Local Whisper `large-v3` | Wisperno |
|---|---|---|---|
| Where audio is processed | Cloud (OpenAI + subprocessors) | On-device | On-device |
| Works with no internet | Never | Yes | Yes, after first model fetch |
| STT backbone | OpenAI Whisper (cloud) + fine-tuned Llama | `large-v3` (full) | `faster-whisper large-v3-turbo` (CTranslate2, ~7x faster decode than full `large-v3`) |
| VRAM floor | None (any device + internet) | ~10GB | 4GB (2GB triggers automatic CPU fallback) |
| Marginal cost per use | Subscription | None (hardware only) | None (hardware only) |
| Text cleanup/formatting | Cloud LLM | None built-in | Local quantized LLM (llama.cpp, 4 VRAM presets) |

## Credits & Third-Party Models

Wisperno is glue code and product design around other people's models and
runtimes - full credit belongs to them:

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** (MIT) /
  **[CTranslate2](https://github.com/OpenNMT/CTranslate2)** (MIT) - the STT
  inference engine.
- **[Whisper](https://github.com/openai/whisper)** (`large-v3-turbo`, MIT) -
  OpenAI's speech recognition model.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT) /
  **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** (MIT) -
  the local LLM inference engine.
- **[Qwen2.5](https://github.com/QwenLM/Qwen2.5)** (Apache 2.0, Alibaba
  Cloud) - the default Standard/Flagship/Turbo Flagship/Eco LLM presets, via
  [bartowski's GGUF quantizations](https://huggingface.co/bartowski).
- **Llama 3.2** (Meta's [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)) -
  the fallback LLM preset. Built with Llama.
- **[PySide6](https://doc.qt.io/qtforpython-6/)** (LGPLv3) - the tray/dashboard UI.
- **[PyAV](https://github.com/PyAV-Org/PyAV)** (BSD-3-Clause) - audio I/O.

## License

Wisperno's own code is [MIT](LICENSE) - v1.0.0. The third-party models and
runtimes above keep their own licenses; see each project for terms,
particularly Meta's Llama 3.2 Community License if you redistribute a build
that bundles the Llama-3.2 fallback weights.
