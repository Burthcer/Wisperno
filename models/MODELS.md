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
