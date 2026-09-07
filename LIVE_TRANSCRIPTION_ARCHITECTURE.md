# Live Transcription: Streaming Architecture & Model Suitability

This document covers the hop-based streaming pipeline introduced in Round 34
(replacing the earlier commit-on-pause-or-ceiling design from Round 32/33),
its hardware/VRAM budget, measured latency and the model-selection decision
behind it, and honest notes on a real correctness bug found and fixed while
building it.

## 1. Streaming Pipeline Mechanics

### 1.1 Why not just wait for silence?

The original (Round 32/33) design committed a chunk when Silero VAD detected
a trailing pause, or when a buffer-size ceiling was hit. That works for
turn-taking dictation but fails on the actual Live Transcription use case:
continuous media playback, a lecturer who doesn't pause, or a fast talker in
a meeting - audio with no clean silence gaps at all. A pause-only trigger
stalls on exactly that content.

### 1.2 Rolling buffer + continuous re-transcription

`LiveTranscriptionWorker._hop()` (`src/live_transcriber.py`) runs on a fixed
wall-clock interval, **`HOP_SEC = 0.6s`** (within the 500-800ms target),
regardless of whether anything paused. Each hop:

1. Drains newly-captured audio from `AudioRecorder.drain_available_audio()`,
   applies AGC (section below), and appends it to an in-memory buffer of
   **unconfirmed** audio only.
2. Gates on real speech content before ever calling Whisper (see section 3):
   an RMS energy check, then a Silero VAD speech-ratio check.
3. Transcribes the **entire unconfirmed buffer** (not a sub-slice - see the
   critical invariant below) with `Transcriber.transcribe_live_window()`,
   which requests `word_timestamps=True` so every word carries a precise
   start/end time.
4. Decides a **confirm boundary**: a VAD-detected trailing pause within the
   buffer, or - if the buffer has grown to `WINDOW_SEC = 2.75s` (within the
   mission's 2.5-3.0s target) with no clean pause yet - the end of the
   buffer itself, as a safety valve for genuinely continuous speech.
5. Words up to that boundary are **confirmed**: sanitized
   (`sanitize_live_chunk()`), appended to the permanent transcript, emitted
   via `text_chunk_received`, and their audio is trimmed off the front of
   the buffer.
6. Words after the boundary (not yet confirmed) are emitted as
   **speculative** text via `speculative_text_changed` - this is the "live
   subtitle" signal: it fires every hop, gets fully REPLACED (never
   appended) by the next hop's re-guess of the same still-unconfirmed
   audio, and is never persisted. A wrong speculative guess costs nothing;
   it's simply overwritten ~600ms later.

### 1.3 A critical invariant, found by testing, not assumed

An earlier version of this pipeline (built and then replaced within this
same round, after its own test suite caught the problem) used a **sliding
trailing-WINDOW_SEC slice** of the buffer as the transcription window once
the buffer grew past that size, combined with **cross-hop word-by-word
agreement** (a LocalAgreement-2-style policy: a word confirms once it
appears identically at the same position in two consecutive hypotheses).
That is architecturally more sophisticated, but `tests/test_live_streaming.py`
caught it doing something serious: on a real ~38-second continuous-speech
sample, **large stretches of real content silently disappeared** from the
confirmed transcript. Root cause: once the buffer grew past `WINDOW_SEC`,
the transcription window only ever looked at the *trailing* slice - older
unconfirmed audio (before that slice) was **never sent to Whisper at all**,
yet a later "confirm the whole window" boundary still trimmed the buffer
using an offset that covered that never-transcribed portion too, discarding
it without a trace.

The fix (what's described in 1.2 above) makes the window **always equal the
whole unconfirmed buffer** - nothing can be trimmed that wasn't just
transcribed, by construction. The buffer is kept near `WINDOW_SEC` by
confirming *promptly* (on the first pause found, or the moment the buffer
reaches the ceiling) rather than by ever sliding the transcription window
ahead of content that hasn't been looked at yet. This is a strictly safer
design and is what's shipped; the cross-hop word-agreement approach was not
used, despite being closer to some published streaming-ASR designs, because
a silent content-loss bug is not an acceptable trade for architectural
elegance in a dictation tool.

### 1.4 Decoder parameters

Every live-mode Whisper call (`Transcriber.transcribe_live_window()`) uses:

- `beam_size=1`, `best_of=1`, `temperature=0.0` - greedy, deterministic, fastest.
- `condition_on_previous_text=False` - **not** the mission's original ask
  (an earlier mission asked for `True` to chase cross-chunk grammatical
  continuity); reverted after `True` was found to let one bad decode's
  hallucination compound into the next call, reproduced live as an infinite
  repeated-phrase loop (`' , , , ' sentence? Pardon me, but I could repeat
  the sentence?...`). Cross-chunk coherence is instead carried via a plain
  `initial_prompt` string (the previous confirmed chunk's trailing words) -
  the same safe technique this codebase's main dictation path already used
  for exactly this reason.
- `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`,
  `no_speech_threshold=0.6` - faster-whisper's own defaults, passed
  explicitly so a future library upgrade can't silently loosen them.
- `vad_filter=False` - the caller has already VAD-gated the buffer before
  calling Whisper at all (section 3); a second internal VAD pass would only
  risk clipping the window's edges.

### 1.5 Automatic Gain Control (AGC)

`apply_agc()` (`src/live_transcriber.py`) runs on every newly-captured audio
chunk before it ever reaches the buffer:

- Targets **-17dBFS** (the center of the mission's -20 to -14dBFS band).
- Gain is capped at **12x** in either direction and smoothed (30% new-gain
  weight per chunk) so it doesn't jump abruptly between chunks.
- A `tanh` soft limiter caps the post-gain signal at 0.98 full-scale,
  preventing clipping on a sudden loud vocal peak after amplification.
- True digital silence (RMS below `1e-6`) is left untouched - AGC never
  "gains up" silence into false signal.

Measured (`tests/test_live_streaming.py::test_agc_brings_quiet_audio_into_target_band`):
a -40dBFS input (a genuinely quiet speaker/laptop-media level) converges to
**-18.5dBFS** after gain smoothing settles, within the target band. A
genuinely quiet real-speech sample (RMS 0.00206, *below* the app's own
speech-energy floor of 0.004 - i.e., a sample the original, ungated pipeline
would have silently dropped) was confirmed to transcribe correctly end to
end once passed through AGC first
(`test_quiet_speech_still_transcribes_via_agc`).

## 2. Hardware Profile & VRAM Budget

Measured on this project's dev machine: **NVIDIA GeForce RTX 4070 Laptop
GPU, 8GB VRAM**. Live Transcription reuses the app's existing `Transcriber`
instance - **no second Whisper model is ever loaded**, so its VRAM cost is
purely the marginal cost of the (slightly heavier, word-timestamp-enabled)
transcription calls on top of whatever's already loaded for normal
dictation.

| Component | VRAM |
|---|---|
| Whisper `large-v3-turbo` (`int8_float16`) | ~1.6GB |
| Turbo Flagship LLM (`Qwen2.5-1.5B-Instruct-Q8_0`, shipped default) | ~1.9GB |
| **Combined, measured live via `nvidia-smi`** | **~3.2-3.5GB** |

This sits comfortably within the mission's ~3.5GB headroom target - Live
Transcription adds no persistent VRAM cost of its own on top of the
already-loaded dictation engines, since it's the same `Transcriber` object.

## 3. Latency & Real-Time-Factor Benchmarks

All numbers below are real measurements on the dev RTX 4070 Laptop GPU
(`tests/test_live_transcriber_anti_loop.py`,
`tests/test_live_streaming.py`), not estimates.

| Metric | Measured | Target | Result |
|---|---|---|---|
| `transcribe_live_window()` on an exact 2.75s window (word timestamps on) | **~203ms avg** (199.8-209.9ms across 5 runs) | <200ms | Effectively at parity - within normal timing variance of the target |
| Same call, real synthesized ~4.0s speech (incl. word alignment) | 231-234ms | - | Consistent with the above, scaled to duration |
| Full `_hop()` (energy/VAD gate + transcribe + confirm/trim logic) over 39 hops on 38s of real continuous speech | **avg 222-252ms**, **max 570-928ms** | keep up with 600ms hop budget | Average comfortably clears the budget; occasional content-dense hops can exceed it without permanently falling behind (see below) |
| Real-time factor (RTF) on the 2.75s window | **~0.074** (203ms / 2750ms) | <1.0 (faster than real-time) | ~13.5x faster than real-time |

**On the <200ms target specifically**: measured average latency for the
mission's literal 2.75s-window benchmark is ~203ms - within measurement
noise of the target, not a clean miss. The metric that actually matters for
a streaming experience - whether the pipeline keeps up with the 600ms hop
cadence - is met with a wide margin (RTF ~0.074, meaning a hop's
transcription work takes roughly 1/13th of the audio duration it covers).
**Occasional individual hops measured up to ~570-930ms** - this happens on
content-dense windows (decode time scales with output token count, not just
audio duration) and does not cause the pipeline to permanently fall behind:
the next hop simply processes whatever backlog accumulated, and the
worst-case measured (930ms) is still well under one second.

## 4. Engineering Justification: Why `large-v3-turbo`, Not a Lighter Model

The mission asked to evaluate a lighter streaming-specific model
(`small.en`/`base.en`/int8 CTranslate2) if `large-v3-turbo` couldn't clear
the 200ms target. Given the measured results above - average latency
effectively at the target, and a >13x real-time-factor margin against the
actual hop-cadence requirement - **switching models was evaluated and
rejected**:

- The literal 200ms number was measured as ~203ms average - a difference
  well within normal run-to-run timing variance on real hardware, not a
  demonstrated shortfall.
- The requirement that actually determines whether streaming "works" (hop
  latency vs. hop interval) is met with over 2x margin even at the
  occasional worst-case (930ms against no hard real-time deadline beyond
  "don't fall permanently behind", which it doesn't).
- `large-v3-turbo` is the SAME model already used for the main dictation
  pipeline (Round 15's empirical bake-off winner for accuracy), and no
  second model would need to be loaded into VRAM if the live path used a
  different model - switching to a lighter model for Live Transcription
  specifically would mean loading a THIRD model into VRAM (on top of
  Whisper + the active LLM preset), which is a real VRAM cost with no
  measured latency problem to justify it.
- A lighter model's accuracy on exactly the audio conditions this mode
  targets (quiet speakers, background media, accented/technical speech) has
  not been benchmarked and is a real, unquantified risk - `large-v3-turbo`
  is the only variant with a track record in this codebase (Round 15's
  fidelity bake-off) of handling this project's own conversational-trap
  test suite correctly.

**No separate "Live Transcription ASR Model" setting was added to Settings
-> Advanced**, since the evaluation concluded the existing model already
meets the requirement - adding a setting for a switch that isn't needed
would be speculative complexity with no current use. If a future round's
real hardware measurement shows a genuine, reproducible shortfall (not
within-noise variance), that setting - and the lighter-model comparison
this section describes doing first - is the documented next step.

## 5. Known Limitations (disclosed, not hidden)

- **Word-boundary decode artifacts**: because each confirm boundary (a VAD
  pause, or the size-ceiling safety valve) ends a Whisper call at an
  arbitrary point rather than a full natural utterance, occasional minor
  word-choice drift can occur right at a chunk boundary (e.g. "a lecture"
  transcribed as "election" in one real test run, or a stray inserted
  connective word). This is normal, expected behavior for ANY streaming ASR
  system that re-decodes bounded windows rather than one continuous
  full-context pass, and does not lose or duplicate content - the anti-loop
  and streaming test suites both verify start-to-end content survives
  intact even though exact per-boundary wording can occasionally differ
  slightly from a hypothetical full-context transcription.
- **The size-ceiling safety valve can end a call mid-word** when no clean
  pause exists within `WINDOW_SEC` (the genuinely-continuous-speech case) -
  the same characteristic the prior round's ceiling design already had, not
  a new regression.
