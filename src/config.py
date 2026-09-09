"""
Configuration module for Wisperno.
Loads and validates settings from config.yaml with full typing and default fallbacks.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Union
import yaml
from pydantic import BaseModel, Field


def get_base_dir() -> Path:
    """
    Resolve the project root, whether running from source or as a frozen
    PyInstaller .exe (where __file__ points inside a temp extraction dir,
    not next to the actual binary).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_config_dir() -> Path:
    """Resolve the config/ directory (config.yaml + dictionary.json) next to the app root."""
    return get_base_dir() / "config"


def get_app_data_dir() -> Path:
    """
    Resolve %APPDATA%\\Wisperno (or ~/Wisperno on non-Windows), creating it if
    needed. Per-user, survives the app's install folder being moved/reinstalled/
    deleted - unlike a path beside the .exe, which is fine until the user drags
    a new build over the old one or the distribution folder gets wiped.
    """
    base = Path(os.environ.get("APPDATA", str(Path.home())))
    data_dir = base / "Wisperno"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_db_path() -> Path:
    """
    Single source of truth for wisperno.db's location - never a relative/temp
    path. One-time migration: if a user is upgrading from a build that stored
    the db beside the exe or in the project root (true before this was moved
    to %APPDATA%), copy their existing history forward instead of silently
    starting them over with an empty database.
    """
    db_path = get_app_data_dir() / "wisperno.db"
    if not db_path.exists():
        for legacy in (
            get_base_dir() / "wisperno.db",
            get_base_dir() / "data" / "wisperno.db",
            get_base_dir() / "Final App" / "v1.0.0" / "wisperno.db",
        ):
            if legacy.exists() and legacy != db_path:
                import shutil
                shutil.copy2(legacy, db_path)
                break
    return db_path


class AudioConfig(BaseModel):
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz (Whisper expects 16000)")
    channels: int = Field(default=1, description="Number of audio channels (1 = mono)")
    min_duration_sec: float = Field(default=0.4, description="Minimum audio duration to process in seconds (below this, discard as a ghost trigger)")
    min_energy_threshold: float = Field(default=0.002, description="RMS energy threshold to filter out background noise / silence - kept low so quiet real speech isn't discarded as silence")
    device_index: Optional[int] = Field(default=None, description="Audio input device index (null = default)")
    auto_silence_seconds: int = Field(
        default=5,
        description="Auto-stop recording (and commit/paste) after this many seconds of continuous "
                     "silence following real speech. 0 = disabled (manual stop only). Applies to "
                     "whichever dictation trigger started the recording.",
    )


class WhisperConfig(BaseModel):
    model_name: str = Field(default="large-v3-turbo", description="Whisper model size or path")
    device: str = Field(default="cuda", description="Device: 'cuda' or 'cpu'")
    compute_type: str = Field(default="int8_float16", description="Compute precision: 'float16', 'int8_float16', 'int8' - int8_float16 halves VRAM (~850MB vs ~1.6GB) at equivalent WER")
    beam_size: int = Field(default=1, description="Beam size for decoding (1 for fastest speed)")
    language: Optional[str] = Field(default="en", description="Force decoding language ('en'); null = auto-detect")
    vad_filter: bool = Field(default=True, description="Enable Silero VAD silence trimming")
    vad_threshold: float = Field(default=0.35, description="Speech probability threshold (lower = more lenient, catches quieter speech; faster-whisper's own default is 0.5)")
    vad_min_speech_duration_ms: int = Field(default=150, description="Shortest speech chunk VAD will keep - low so short utterances ('yes', 'no') survive")
    vad_min_silence_duration_ms: int = Field(default=1200, description="Minimum silence duration to trim - kept generous so natural mid-sentence pauses survive intact")
    vad_speech_pad_ms: int = Field(default=500, description="Padding added around detected speech so word onsets/tails aren't clipped")
    initial_prompt: str = Field(
        default="Hello, welcome. Please transcribe with clear punctuation, commas, and sentence structure.",
        description=(
            "Initial prompt fed to Whisper for formatting context. Whisper's initial_prompt "
            "conditions the decoder via token continuation, not instruction-following - an "
            "example-shaped prompt with real commas primes punctuated output measurably "
            "better than an instruction-shaped one (verified directly: 0 vs 2 commas on the "
            "identical captured audio, comparing this wording against a purely instructional "
            "one)."
        ),
    )


class LLMConfig(BaseModel):
    # Winner of tests/benchmark_models.py's empirical SLM bake-off (Qwen2.5-1.5B/3B-
    # Instruct, Llama-3.2-3B-Instruct; Qwen3-4B-Instruct-2507 excluded - its GGUF
    # architecture isn't loadable by the pinned llama-cpp-python 0.2.90, and the
    # only newer build tried, 0.3.35, hard-crashed loading *every* model with
    # WinError 0xc000001d on this machine - reverted, not worth the regression risk).
    # 67.5 tok/s avg decode, 98% verbatim-fidelity retention, structured
    # prompt-engineer output, 3615MB combined VRAM with Whisper (100% GPU offload,
    # zero PCIe/CPU split) - and, decisively, the ONLY candidate that survived
    # Wisperno's conversational-trap suite with zero drift. Llama-3.2-3B-Instruct
    # scored higher on raw throughput (71.9 tok/s) and Sample B fidelity (100%)
    # but directly answered "can you explain python pointers to me" instead of
    # cleaning it (so did Qwen2.5-1.5B) - a correctness violation of Wisperno's
    # core "never answer, only clean" mandate that no throughput number outweighs.
    repo_id: str = Field(default="bartowski/Qwen2.5-3B-Instruct-GGUF", description="Hugging Face repo ID")
    filename: str = Field(default="Qwen2.5-3B-Instruct-Q4_K_M.gguf", description="GGUF model filename")
    model_path: str = Field(default="models/Qwen2.5-3B-Instruct-Q4_K_M.gguf", description="Local GGUF model path")
    n_gpu_layers: int = Field(default=-1, description="Layers offloaded to GPU VRAM; -1 = all layers (100% GPU, zero PCIe/CPU offload - the model comfortably fits VRAM budget at this size)")
    n_threads: int = Field(default=16, description="CPU threads - moot at n_gpu_layers=-1 (nothing runs on CPU), kept for the fallback model / a user manually lowering the GPU-layer slider")
    n_ctx: int = Field(default=8192, description="Context window length in tokens - headroom for long-form dictation/selection chunks; costs ~76MB VRAM over 4096 on a 3B Q4_K_M model, measured")
    n_batch: int = Field(
        default=1024,
        description="Prompt batch size - larger cuts prompt-prefill time (TTFT) on a cold/prompt-switch call; "
        "measured directly on this GPU (Qwen2.5-1.5B-Q8_0, 1202-token system prompt): 512->1024 cut cold TTFT "
        "~180ms->~134ms with zero change to steady-state tok/s and zero measurable VRAM delta. 2048 measured "
        "within noise of 1024 (~129ms) for no further benefit, so not worth its slightly larger compute buffer.",
    )
    kv_cache_quantization: bool = Field(default=True, description="Q8_0-quantize the KV cache (type_k/type_v=8) to save VRAM; requires flash attention")
    temperature: float = Field(default=0.0, description="Sampling temperature (0.0 = greedy, deterministic)")
    max_tokens: int = Field(default=4096, description="Maximum generated output tokens")
    flash_attn: bool = Field(
        default=True,
        description="Enable llama.cpp flash attention. Must be False for Gemma-2-family models specifically - "
        "their attention/logit softcapping is architecturally incompatible with flash attention (llama.cpp "
        "silently disables it for that architecture regardless of this flag, but setting it explicitly here "
        "keeps kv_cache_quantization - which itself requires flash_attn - from being wired on by mistake "
        "for a model where it can't actually apply).",
    )
    chat_style: str = Field(
        default="standard",
        description="'standard' sends a real system-role message via create_chat_completion (works for "
        "Qwen/Llama-family GGUFs used elsewhere in this app). 'gemma' is required for Gemma-2-family models: "
        "they have no system role at all - llama-cpp-python's own built-in gemma chat-format handler "
        "silently DROPS a system message rather than erroring (verified directly against this app's pinned "
        "llama-cpp-python==0.2.90), so the 'gemma' style instead folds the system prompt into the first user "
        "turn itself before formatting, matching Gemma's native <start_of_turn>user/<end_of_turn> template.",
    )

    # Lightweight fallback used automatically when the primary model file isn't
    # present on disk (e.g. still downloading). Llama-3.2-3B-Instruct - NOT
    # immune to the same conversational-trap drift as the primary (see above),
    # but it's a temporary stopgap only, not the shipped default.
    fallback_repo_id: str = Field(default="bartowski/Llama-3.2-3B-Instruct-GGUF", description="Hugging Face repo ID for the fallback model")
    fallback_filename: str = Field(default="Llama-3.2-3B-Instruct-Q4_K_M.gguf", description="Fallback GGUF model filename")
    fallback_model_path: str = Field(default="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf", description="Local fallback GGUF model path")


class InjectorConfig(BaseModel):
    pre_paste_delay_ms: int = Field(default=40, description="Delay before sending Ctrl+V for clipboard synchronization")
    post_paste_delay_ms: int = Field(default=80, description="Delay after paste before restoring previous clipboard")
    restore_clipboard: bool = Field(default=True, description="Whether to restore prior clipboard content")
    auto_copy_to_clipboard: bool = Field(
        default=False,
        description="Leave the transcribed/polished text on the clipboard after typing it, instead of restoring "
        "whatever was there before - overrides restore_clipboard's own restore step when enabled",
    )


class AppConfig(BaseModel):
    # Two fully independent dictation triggers, each with its own hotkey and
    # on/off switch - a user can run either, both (with different chords), or
    # neither. Replaces the old single hotkey+trigger_mode pair (see
    # load_config()'s migration below for configs saved before this existed).
    # "Polish Selected Text" has no dedicated hotkey field anymore: it lives
    # entirely in the Transforms hub now - any transform's own hotkey already
    # falls back to polishing the current selection when no speech is
    # captured (see WispernoEngine._on_transform_release), so a separate
    # global shortcut for the same behavior was redundant and confusing.
    tap_toggle_hotkey: str = Field(default="ctrl+alt", description="Tap-to-toggle dictation hotkey: tap once to start, tap again to stop")
    tap_toggle_enabled: bool = Field(default=True, description="Whether the tap-to-toggle dictation trigger is active")
    hold_to_talk_hotkey: str = Field(default="ctrl+space", description="Hold-to-talk dictation hotkey: hold to record, release to stop")
    hold_to_talk_enabled: bool = Field(default=False, description="Whether the hold-to-talk dictation trigger is active")
    cycle_mode_hotkey: str = Field(default="ctrl+shift+b", description="Hotkey to cycle transformation modes")
    settings_hotkey: str = Field(default="ctrl+,", description="Hotkey to open the Settings & Dictionary window")
    dashboard_hotkey: str = Field(default="ctrl+shift+h", description="Hotkey to toggle the full desktop dashboard window")
    live_transcribe_hotkey: str = Field(default="ctrl+shift+l", description="Hotkey to start/stop a Live Transcription session")
    writing_styles_hotkey: str = Field(default="alt+v", description="Hotkey to open the Writing Styles HUD on the current text selection")
    minimize_to_tray: bool = Field(default=True, description="Closing the dashboard/pill minimizes to tray instead of quitting")
    pill_always_on_top: bool = Field(default=True, description="Keep the floating pill above other windows, including fullscreen apps/games")
    audio_cues_enabled: bool = Field(default=False, description="Play a short tone when dictation starts and stops")
    db_retention_policy: str = Field(default="never", description="History auto-prune window: '3_months', '6_months', '1_year', or 'never'")
    active_mode: str = Field(default="polish", description="Active transformation mode ('polish', 'prompt_engineer', 'bullets', 'raw')")
    dictionary_path: str = Field(default="config/dictionary.json", description="Custom word-replacement dictionary path")
    profanity_filter: str = Field(
        default="allow",
        description="How spoken profanity is handled in polished output: 'allow' (verbatim, default), "
                     "'censor' (first letter + asterisks, e.g. 'fuck' -> 'f***'), 'remove' (dropped entirely).",
    )
    auto_llm_polish: bool = Field(
        default=False,
        description="Default OFF: the main dictation hotkey uses the instant, zero-LLM "
                     "src/direct_formatter.py (filler/stutter stripping, spoken-list detection, "
                     "capitalization) for sub-second latency. When True, dictation is routed through "
                     "the active LLM preset instead (src/transformer.py), same as before this setting "
                     "existed. The dedicated Polish Selected Text hotkey always uses the LLM regardless.",
    )
    model_preset: str = Field(
        default="standard",
        description=(
            "Selected Engine Preset key ('eco' | 'standard' | 'flagship' | 'turbo' - see "
            "src/ui/settings_tab.py:MODEL_PRESETS). The llm.* fields below are the actual "
            "source of truth for what loads; this just remembers which named preset they "
            "came from so the Settings dropdown re-selects the right entry on next launch, "
            "rather than falling back to matching llm.model_path against the presets table."
        ),
    )
    audio: AudioConfig = Field(default_factory=AudioConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    injector: InjectorConfig = Field(default_factory=InjectorConfig)
    prompts: Dict[str, str] = Field(default_factory=dict)


def _migrate_legacy_trigger_fields(data: dict) -> None:
    """In-place: maps a pre-dual-trigger config's single `hotkey`/`trigger_mode`
    pair onto the new independent tap_toggle_*/hold_to_talk_* fields, so an
    existing user's chosen hotkey and mode survive the upgrade instead of
    silently resetting to defaults. `polish_selection_hotkey` (removed - see
    AppConfig's comment) needs no migration, it's simply dropped."""
    if "tap_toggle_hotkey" in data or "hotkey" not in data:
        return  # already on the new schema, or a fresh/default config
    old_hotkey = data.pop("hotkey", "ctrl+alt")
    old_mode = data.pop("trigger_mode", "hold_to_talk")
    if old_mode == "tap_to_toggle":
        data["tap_toggle_hotkey"] = old_hotkey
        data["tap_toggle_enabled"] = True
        data["hold_to_talk_enabled"] = False
    else:
        data["hold_to_talk_hotkey"] = old_hotkey
        data["hold_to_talk_enabled"] = True
        data["tap_toggle_enabled"] = False


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """
    Load configuration from YAML file with fallback to defaults. Sanitized:
    a corrupt/unparseable YAML file or a value that fails Pydantic's type
    validation (AppConfig's field types ARE the schema) falls back to safe
    defaults instead of raising - a malformed config.yaml must never be a
    silent, unrecoverable boot failure.
    """
    if config_path is None:
        config_path = get_config_dir() / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return AppConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _migrate_legacy_trigger_fields(data)
        return AppConfig(**data)
    except Exception as e:
        try:
            from loguru import logger
            logger.error(
                f"config.yaml at '{config_path}' failed to parse/validate ({e}) - "
                "falling back to safe defaults for this launch rather than failing to start."
            )
        except Exception:
            pass
        return AppConfig()


def save_config(config: AppConfig, config_path: Optional[Union[str, Path]] = None) -> None:
    """Persist an AppConfig back to config.yaml, preserving multi-line prompt formatting."""
    if config_path is None:
        config_path = get_config_dir() / "config.yaml"
    else:
        config_path = Path(config_path)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False, allow_unicode=True)
