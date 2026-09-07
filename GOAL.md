# GOAL.md - Wisperno: Offline Desktop Dictation Engine ("Wispr Flow Clone")

## 1. Project Mission
Wisperno is a local, low-latency (<500ms post-release), 100% offline push-to-talk dictation engine and contextual text-transformation daemon for Windows 11 with NVIDIA GPU acceleration. It listens globally for a push-to-talk hotkey, captures audio in-memory, transcribes speech with state-of-the-art accuracy, applies contextual LLM transformations (polishing, formatting, prompt engineering), and injects the polished text directly into whichever desktop application currently holds focus.

## 2. Pipeline Architecture
```
[User Holds Hotkey] (pynput)
         │
         ▼
[In-Memory Audio Capture] (sounddevice: 16kHz, Mono Float32/PCM, thread-safe queue)
         │
[User Releases Hotkey]
         │
         ▼
[VAD / Silence Trimming] (silero-vad / built-in faster-whisper VAD)
         │
         ▼
[Local Speech-to-Text] (faster-whisper: large-v3-turbo / medium.en on CUDA FP16)
         │
         ▼
[Contextual Polish & Transform] (llama-cpp-python: Llama-3.2-3B-Instruct-Q4_K_M on CUDA)
         │
         ▼
[OS-Level Text Injection] (pyperclip + simulated Ctrl+V via pynput.keyboard.Controller)
```

### Detailed Pipeline Steps:
1. **Global Hotkey Trigger (`pynput`)**:
   - Push-to-talk state machine tracks key down and key up events.
   - Mode cycling shortcut allows quick switching between transformation modes.
2. **In-Memory Audio Capture (`sounddevice`)**:
   - Captures non-blocking 16kHz mono audio into an in-memory buffer / queue.
   - Zero disk writes for active audio to guarantee zero I/O latency.
   - RMS energy & duration check discards accidental clicks (<0.2s or silent taps).
3. **Pre-Transcription Silence Trimming**:
   - Silero-VAD filter removes leading, trailing, and inter-phrase silence before inference.
4. **Local Speech-to-Text (`faster-whisper`)**:
   - Powered by CTranslate2 on NVIDIA CUDA with FP16 precision.
   - Configured with `beam_size=1` (or `3`), `vad_filter=True`, and tailored initial prompt.
5. **Contextual Polish & Transform (`llama-cpp-python`)**:
   - Local quantized SLM (Llama 3.2 3B Instruct Q4_K_M or Qwen 2.5 3B Instruct Q4_K_M) fully offloaded to GPU (`n_gpu_layers=-1`).
   - Removes fillers, repairs stutters, formats punctuation, converts to bullet points, or engineers prompts based on active mode.
6. **OS-Level Text Injection (`pyperclip` + `pynput`)**:
   - Safely saves current clipboard content, copies formatted text, delays 30-50ms for Windows clipboard manager synchronization, dispatches simulated `Ctrl+V`, delays 100ms, and restores the previous clipboard.

## 3. Supported Modes
- **Default Polish (`polish`)**: Eliminates filler words (um, uh, like, you know), cleans stutters, adds natural capitalization and punctuation without altering the speaker's core intent or tone.
- **Prompt Engineer (`prompt_engineer`)**: Converts informal, rambling speech into a structured, high-efficiency prompt for ChatGPT / Claude / Gemini / Cursor.
- **Bullet Points (`bullets`)**: Synthesizes spoken thoughts into structured, concise Markdown bullet points.
- **Raw (`raw`)**: Directly outputs the exact literal transcription without LLM processing for fastest possible throughput.

## 4. Latency & Resource Targets
- **Post-Release Processing Latency**: <500ms for standard 5-10 second utterances on RTX 40-series GPUs.
- **Total VRAM Footprint**: <6.0 GB VRAM total allocation:
  - faster-whisper `large-v3-turbo`: ~1.5 - 2.0 GB VRAM (FP16).
  - Llama 3.2 3B Instruct Q4_K_M: ~2.0 - 2.5 GB VRAM (`n_ctx=2048`).
  - Total system footprint leaves plenty of headroom on 8GB+ GPUs.
- **Disk I/O**: Zero disk writes during recording, transcription, and transformation cycles.
