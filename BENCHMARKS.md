# Benchmarks

Measured directly on the dev machine (RTX-class NVIDIA GPU, CUDA) via
`tests/benchmark_models.py`, not estimated. Whisper `large-v3-turbo`
(int8_float16) loaded throughout, matching the real app's combined VRAM
footprint. Last measured 2026-09-09 against `llama-cpp-python==0.3.26`.

| Preset | VRAM | Process RAM | Avg decode speed | 100-word latency | Fidelity |
|---|---|---|---|---|---|
| **Turbo Flagship (default)** | 3491 MB | 2311 MB | **113.8 tok/s** | 0.82s | 97.6% |
| Eco (IQ4_XS) | 3621 MB | 2319 MB | 103.3 tok/s | 0.89s | 97.6% |
| Standard (Q4_K_M, 3B) | 3691 MB | 2559 MB | 91.7 tok/s | 1.04s | 98.8% |
| Flagship (Q8_0, 3B, full-precision KV) | 5119 MB | 3871 MB | 63.1 tok/s | 1.48s | 98.8% |

"Fidelity" on Turbo/Eco is measured on raw model output and bypasses the
app's own guardrail (`sanity_check()`) by design - real usage always goes
through it, which is why the conversational-trap suite (adversarial prompts
designed to make the model answer back instead of cleaning dictation) still
passes 100% on every tier above despite the lower raw-fidelity score on the
smaller models.

## Why these numbers moved since the last release

This build's `llama-cpp-python` inference engine was upgraded twice this
session, `0.2.90` -> `0.3.4` -> `0.3.26` (see `handoff.md` in the source repo,
Rounds 39-40, for the full investigation - version-bisected against a range
of releases that crash on this hardware to find the newest one that doesn't).
Net effect versus the previous release's published numbers: **every preset
decodes roughly 20-30% faster** with identical output fidelity and unchanged
VRAM - a pure engine-level win, not a model or quantization change. Cold-start
latency (time to first token after a mode switch) also improved separately
via the `0.2.90` -> `0.3.4` half of that upgrade (~2x faster, not reflected in
the steady-state numbers above).

## Reproducing this benchmark

From a full source checkout (not this portable download - see INSTALL.md's
"Building from source" section for setup):

```
python tests/benchmark_models.py
```

Requires all four preset model files present in `models/` (Turbo and Eco/
Standard/Flagship's weights are not all bundled in the same distribution -
see `models/MODELS.md` in this folder for where to get each one).
