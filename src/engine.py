"""
Central orchestration controller for Wisperno's Qt UI.

Owns config/database/vocabulary immediately (cheap); everything expensive -
audio device probing, Whisper, the local LLM, and the Win32 hotkey poller - is
constructed on a background thread via load_heavy_engines() so QApplication's
event loop and the floating pill are interactive from the first frame.

Bridges between plain-Python callbacks (HotkeyManager, AudioRecorder) and the
Qt world purely through Signals, which PySide6 marshals thread-safely onto
whichever thread owns the connected slot.
"""

import threading
import time
from typing import Dict, Optional

from PySide6.QtCore import QObject, Signal
from loguru import logger

from src import audio_cues
from src.__version__ import __version__
from src.config import AppConfig, get_app_data_dir, get_base_dir, get_db_path, load_config, save_config
from src.database import WispernoDB
from src.vocabulary import Vocabulary
from src.workers import AudioWorker, HotkeyWorker, InferenceWorker, SelectionPolishWorker, WritingStylesWorker

MODES = ["polish", "prompt_engineer", "bullets", "code", "raw"]

# Muscle-memory alias: "alt+p" ("Polish") triggers the same "polish" transform
# as its configured shortcut, regardless of what that shortcut is - a second
# way in, not a replacement for the user's own binding.
POLISH_HOTKEY_ALIAS = "alt+p"


def _build_transform_hotkeys(db) -> Dict[str, str]:
    hotkeys = {t["id"]: t["shortcut"] for t in db.list_transforms(active_only=True) if t["shortcut"]}
    if "polish" in hotkeys and POLISH_HOTKEY_ALIAS not in hotkeys["polish"].split("|"):
        hotkeys["polish"] += f"|{POLISH_HOTKEY_ALIAS}"
    return hotkeys

# Ceiling for the one-time engine load (Whisper + local LLM). Not the mission's
# literal 15s: a normal combined load on this hardware is already 7-12s, and
# first-run disk-cache misses or a slower GPU eat into that margin fast - 15s
# would fire falsely on hardware only slightly slower than this dev machine.
# 45s leaves real headroom while still catching a genuine hang in well under a
# session, let alone the hours a user waited before reporting this.
INIT_TIMEOUT_SEC = 45.0


class WispernoEngine(QObject):
    state_changed = Signal(str)       # 'loading' | 'idle' | 'recording' | 'processing' | 'typing' | 'success' |
                                       # 'error' | 'selection_processing' | 'selection_success' | 'fallback_warning' |
                                       # 'cpu_mode' (non-blocking startup toast, no dedicated GPU detected)
    level_changed = Signal(float)
    mode_changed = Signal(str)
    history_added = Signal(dict)
    engines_ready = Signal(bool)
    settings_requested = Signal()
    dashboard_toggle_requested = Signal()
    hotkey_changed = Signal(str)
    live_transcribe_toggle_requested = Signal()   # hotkey/sidebar asked to start OR stop a session
    live_transcribe_state_changed = Signal(str)   # 'listening' | 'cleaning_up' | 'stopped'
    live_transcribe_chunk_received = Signal(str)
    live_transcribe_speculative_changed = Signal(str)  # replaces (never appends) - the current unconfirmed tail
    live_transcribe_session_saved = Signal(dict)  # the finished history row (empty raw = discarded)
    writing_styles_no_selection = Signal()         # Alt+V pressed with nothing highlighted - pill toast only
    writing_styles_selection_ready = Signal(str)   # (original_text) - HUD should appear now
    writing_styles_style_ready = Signal(str, str)  # (style_id, styled_text)
    writing_styles_all_done = Signal()

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.config_path = config_path  # remembered so every save_config() call below writes back to
                                         # THIS engine's config file, not silently falling back to the
                                         # default project config.yaml regardless of what was passed in
        self.config: AppConfig = load_config(config_path)
        self.active_mode: str = self.config.active_mode
        self._engines_ready = False
        self._processing = False
        self._ptt_press_time = 0.0
        self._active_transform_id: Optional[str] = None  # set by a per-transform hotkey press, read at release
        self._inference_worker: Optional[InferenceWorker] = None
        self._live_worker = None  # type: Optional["LiveTranscriptionWorker"]
        self._writing_styles_worker = None  # type: Optional["WritingStylesWorker"]
        self._writing_styles_selection = ""  # the raw text a chosen style gets saved against in history
        self._beta_fallback_occurred = False  # set by _load_transformer_with_beta_fallback() this boot

        db_path = get_db_path()
        self.db = WispernoDB(db_path)
        self.db.migrate_dictionary_json(get_base_dir() / self.config.dictionary_path)
        self.db.seed_additional_corrections()
        self.db.seed_additional_corrections_v3()
        self.db.free_up_alt_v_shortcut()
        self.db.seed_grammar_correct_transform()
        self.db.seed_ai_relay_transform()
        self.vocabulary = Vocabulary(db=self.db)

        self.audio_worker: Optional[AudioWorker] = None
        self.hotkey_worker: Optional[HotkeyWorker] = None
        self.injector = None
        self.transcriber = None
        self.transformer = None
        self._init_done_event = threading.Event()
        self._cpu_compat_mode = False  # set by _load_heavy_engines() if no usable GPU was detected

    # --- Startup --------------------------------------------------------------

    @staticmethod
    def _boot_guard_path():
        return get_app_data_dir() / "boot_in_progress.flag"

    def _clear_boot_guard(self) -> None:
        try:
            self._boot_guard_path().unlink(missing_ok=True)
        except Exception:
            pass

    def _recover_from_stuck_boot(self) -> None:
        """
        Cross-launch recovery: if the LAST launch's engine load never reached
        its success point (crashed, hung past the watchdog, or was force-quit
        mid-load), the guard file this method finds left over from that
        attempt is proof the CURRENT model/preset config is what's broken -
        a plain relaunch would retry the identical config and hit the same
        failure again, which is exactly the reported "relaunching does not
        clear the hang." Force the known-good Standard preset for this launch
        and persist it, so the loop breaks instead of repeating indefinitely.
        """
        logger.error(
            "Previous launch did not finish initializing (crash, hang, or force-quit mid-load) - "
            "falling back to the Standard preset for this launch instead of retrying the same config."
        )
        self.config.model_preset = "standard"
        self.config.llm.repo_id = "bartowski/Qwen2.5-3B-Instruct-GGUF"
        self.config.llm.filename = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
        self.config.llm.model_path = "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
        self.config.llm.kv_cache_quantization = True
        try:
            save_config(self.config, self.config_path)
        except Exception as e:
            logger.warning(f"Could not persist the safe-mode preset fallback: {e}")

    def start(self) -> None:
        guard_path = self._boot_guard_path()
        try:
            if guard_path.exists():
                self._recover_from_stuck_boot()
            else:
                guard_path.parent.mkdir(parents=True, exist_ok=True)
            guard_path.touch()
        except Exception as e:
            logger.warning(f"Boot-guard check failed (non-fatal): {e}")

        self.state_changed.emit("loading")
        self._init_done_event.clear()
        threading.Thread(target=self._load_heavy_engines, daemon=True).start()
        threading.Thread(target=self._init_watchdog, daemon=True).start()

    def _init_watchdog(self) -> None:
        """
        Best-effort safety net, not the primary fix: the real bug (a native
        CTranslate2/llama.cpp write blocking forever on an invalid stdout/stderr
        fd in the frozen build) is fixed at its source in main.py, before any
        native library is imported. This just guarantees the pill never sits on
        "Starting..." with zero explanation if init is ever slow for some other
        reason - it can't forcibly abort a stuck native call (unsafe in Python),
        it only makes sure the failure is visible instead of silent. Cross-launch
        recovery (see _recover_from_stuck_boot) is what actually breaks a repeat-
        hang loop; this only guarantees THIS launch's failure is visible, logged,
        and left for the next launch to detect via the still-present guard file.
        """
        if not self._init_done_event.wait(timeout=INIT_TIMEOUT_SEC):
            logger.error(
                f"Engine initialization has not completed after {INIT_TIMEOUT_SEC:.0f}s "
                "(see the log above for the last successful step). It may still finish in the "
                "background, but something is abnormally slow or stuck. If you restart the app, "
                "it will automatically fall back to the Standard preset instead of retrying this."
            )
            self.state_changed.emit("error")

    def _load_heavy_engines(self) -> None:
        try:
            from src.audio_recorder import AudioRecorder  # noqa: F401 (import-time cost check)
            from src.device_manager import detect_gpu_capability
            from src.injector import TextInjector
            from src.transcriber import Transcriber
            from src.transformer import Transformer

            # Proactive hardware check, before either model is loaded: integrated
            # graphics or a VRAM-starved GPU can otherwise crash Whisper/llama.cpp's
            # own CUDA init outright rather than falling back cleanly. This override
            # is in-memory only for THIS launch - never written back to config.yaml,
            # so a saved GPU preset/layer-count survives intact for the next launch
            # on hardware that actually has the VRAM for it.
            gpu = detect_gpu_capability()
            self._cpu_compat_mode = not gpu["use_gpu"]
            if self._cpu_compat_mode:
                self.config.whisper.device = "cpu"
                self.config.whisper.compute_type = "int8"
                self.config.llm.n_gpu_layers = 0

            logger.info("Initializing multi-channel audio capture...")
            self.audio_worker = AudioWorker(self.config.audio)
            self.audio_worker.level_changed.connect(self.level_changed)
            self.audio_worker.auto_silence_triggered.connect(self._on_auto_silence)

            logger.info("Initializing Win32 text injector...")
            self.injector = TextInjector(config=self.config.injector)

            logger.info("Initializing Whisper STT engine (this can take a while)...")
            self.transcriber = Transcriber(config=self.config.whisper, vocabulary=self.vocabulary)

            logger.info("Initializing local SLM transform engine...")
            self.transformer = self._load_transformer_with_beta_fallback()

            logger.info("Initializing Win32 GetAsyncKeyState hotkey manager...")
            transform_hotkeys = _build_transform_hotkeys(self.db)
            self.hotkey_worker = HotkeyWorker(self.config, transform_hotkeys=transform_hotkeys)
            self.hotkey_worker.recording_started.connect(self._on_ptt_press)
            self.hotkey_worker.recording_stopped.connect(self._on_ptt_release)
            self.hotkey_worker.mode_cycled.connect(self._on_mode_cycle)
            self.hotkey_worker.settings_requested.connect(self.settings_requested)
            self.hotkey_worker.dashboard_toggle_requested.connect(self.dashboard_toggle_requested)
            self.hotkey_worker.live_transcribe_toggle_requested.connect(self.toggle_live_transcription)
            self.hotkey_worker.writing_styles_triggered.connect(self.trigger_writing_styles)
            self.hotkey_worker.transform_press_requested.connect(self._on_transform_press)
            self.hotkey_worker.transform_release_requested.connect(self._on_transform_release)
            self.hotkey_worker.start()

        except Exception as e:
            logger.error(f"Fatal error loading AI engines: {e}", exc_info=True)
            logger.error("The pill is running, but dictation is disabled until this is fixed.")
            self._clear_boot_guard()  # a fast, logged failure isn't the silent-hang case the guard exists for
            self._init_done_event.set()  # cleared before signaling "done" - no window where a waiter can observe a stale guard
            self.state_changed.emit("error")
            return

        self._engines_ready = True
        self._clear_boot_guard()
        self._init_done_event.set()
        self.engines_ready.emit(True)
        self.state_changed.emit("idle")
        if self._cpu_compat_mode:
            # Non-blocking "toast": the pill briefly shows this, then auto-collapses
            # back to idle on its own (see floating_pill.py's STATE_CPU_MODE) -
            # nothing in the app is blocked waiting on it being dismissed.
            self.state_changed.emit("cpu_mode")
        if self._beta_fallback_occurred:
            self.state_changed.emit("engine_fallback")

        triggers = []
        if self.config.tap_toggle_enabled:
            triggers.append(f"{self.config.tap_toggle_hotkey.upper()} (tap-to-toggle)")
        if self.config.hold_to_talk_enabled:
            triggers.append(f"{self.config.hold_to_talk_hotkey.upper()} (hold-to-talk)")
        logger.success("=" * 70)
        logger.success(f" Wisperno v{__version__} is ACTIVE.")
        logger.success(f" Dictation trigger(s): {', '.join(triggers) if triggers else '(none enabled)'}.")
        logger.success(f" Press [{self.config.cycle_mode_hotkey.upper()}] to cycle transformation modes.")
        logger.success(f" Press [{self.config.settings_hotkey.upper()}] to open the dashboard.")
        logger.success(f" Initial Mode: [{self.active_mode.upper()}]")
        logger.success("=" * 70)

    def _load_transformer_with_beta_fallback(self) -> "Transformer":
        """
        Constructs the Transformer for whatever preset is currently configured.
        If that preset is the experimental 'gemma_beta' tier and it fails to
        produce a usable LLM (bad GGUF, incompatible architecture, VRAM
        overflow, or any other load-time exception), this reverts the LIVE
        config back to Turbo Flagship, persists that reversion so the NEXT
        launch doesn't repeat the same failed load, and retries construction
        once. Only Python-level failures are catchable this way - a genuine
        native crash inside llama.cpp's C code (an access violation, not a
        raised exception) can still take the whole process down; this is a
        real limit of wrapping a ctypes-bound native library in try/except,
        not something a Python-side try/except can close.
        """
        from src.transformer import Transformer

        try:
            transformer = Transformer(
                config=self.config.llm, prompts=self.config.prompts,
                profanity_filter=self.config.profanity_filter,
            )
            load_failed = transformer.llm is None
        except Exception as e:
            logger.error(f"Transformer construction raised for preset '{self.config.model_preset}': {e}", exc_info=True)
            transformer = None
            load_failed = True

        if not (load_failed and self.config.model_preset == "gemma_beta"):
            return transformer

        logger.error(
            "Experimental 'Gemma 2 2B (Beta)' preset failed to load - reverting to Turbo Flagship "
            "and persisting that reversion so the next launch doesn't retry the same failed config."
        )
        from src.ui.settings_tab import MODEL_PRESETS  # lazy: avoids engine.py importing the UI layer at module scope

        turbo = MODEL_PRESETS["turbo"]
        self.config.model_preset = "turbo"
        self.config.llm.repo_id = turbo["repo_id"]
        self.config.llm.filename = turbo["filename"]
        self.config.llm.model_path = turbo["model_path"]
        self.config.llm.kv_cache_quantization = turbo["kv_cache_quantization"]
        self.config.llm.n_ctx = turbo["n_ctx"]
        self.config.llm.flash_attn = turbo["flash_attn"]
        self.config.llm.chat_style = turbo["chat_style"]
        self.save_config()
        self._beta_fallback_occurred = True

        return Transformer(
            config=self.config.llm, prompts=self.config.prompts,
            profanity_filter=self.config.profanity_filter,
        )

    # --- Pipeline --------------------------------------------------------------

    def _on_ptt_press(self) -> None:
        if not self._engines_ready or self._processing or self._live_worker is not None:
            return
        self._ptt_press_time = time.perf_counter()
        # The badge must reflect what THIS recording will actually do, not
        # just which active_mode preset happens to be selected: with
        # auto_llm_polish off (the default), the main dictation hotkey always
        # runs the zero-LLM src/direct_formatter.py regardless of active_mode,
        # so showing "[POLISH]" there falsely implied an LLM polish pass that
        # never happens.
        badge = "DIRECT" if not self.config.auto_llm_polish else self.active_mode.upper()
        logger.info(f"[RECORDING STARTED] (Mode: {badge})...")
        self.state_changed.emit(f"recording:{badge}")
        if self.config.audio_cues_enabled:
            audio_cues.play_start_cue()
        self.audio_worker.start_recording()

    def _on_ptt_release(self) -> None:
        if not self._engines_ready:
            return
        if self.config.audio_cues_enabled:
            audio_cues.play_stop_cue()
        audio_array = self.audio_worker.stop_recording()
        # The captured-audio buffer's own length, not press-to-release wall-clock
        # time (which runs ~0.3-0.5s long from stream start/stop overhead and would
        # otherwise be what ends up stored as the history row's duration).
        record_duration_ms = self.audio_worker.recorder.last_duration_seconds * 1000
        if audio_array is None or len(audio_array) == 0:
            logger.info("[RECORDING CANCELLED] Audio duration or energy below threshold.")
            self.state_changed.emit("idle")
            return

        if self._processing:
            logger.warning("Pipeline is already processing a previous utterance. Skipping.")
            return
        self._processing = True

        snippets: Dict[str, str] = {
            row["trigger_phrase"]: row["expansion_text"] for row in self.db.list_snippets(active_only=True)
        }

        worker = InferenceWorker(
            audio_array, self.transcriber, self.transformer, self.injector, self.db,
            snippets, self.active_mode, record_duration_ms,
            use_direct_formatter=not self.config.auto_llm_polish,
        )
        worker.state_changed.connect(self.state_changed)
        worker.result_ready.connect(self.history_added)
        worker.finished.connect(self._on_inference_finished)
        self._inference_worker = worker
        worker.start()

    def _on_inference_finished(self) -> None:
        self._processing = False
        self._inference_worker = None

    def _on_auto_silence(self) -> None:
        """Auto-Stop Silence Detection fired (AudioRecorder's own live RMS
        watchdog) - the user paused past their configured threshold. Ends the
        recording and runs the pipeline exactly like a manual stop would,
        through whichever release handler actually owns this recording (the
        main dictation hotkey, or a Transforms-hub hotkey), then tells the
        hotkey manager that trigger is no longer "active" so a later physical
        release/tap doesn't misfire a second, redundant stop on top of this one."""
        if not self._engines_ready or not self.audio_worker.recorder.is_recording:
            return
        logger.info("[AUTO-SILENCE] Pause threshold reached - auto-stopping and committing dictation.")
        if self.hotkey_worker:
            self.hotkey_worker.manager.force_stop_active_recording()
        if self._active_transform_id is not None:
            self._on_transform_release(self._active_transform_id)
        else:
            self._on_ptt_release()

    def _on_transform_press(self, transform_id: str) -> None:
        """A per-transform hotkey (e.g. Alt+X) was pressed - same as the main PTT press, just tagged."""
        if not self._engines_ready or self._processing or self._live_worker is not None:
            return
        self._active_transform_id = transform_id
        self._ptt_press_time = time.perf_counter()
        logger.info(f"[RECORDING STARTED] (Transform: {transform_id.upper()})...")
        self.state_changed.emit("recording")
        if self.config.audio_cues_enabled:
            audio_cues.play_start_cue()
        self.audio_worker.start_recording()

    def _on_transform_release(self, transform_id: str) -> None:
        """
        Released a per-transform hotkey. If real speech was captured, dictate
        with that transform's own prompt (Section 4.1). If nothing usable was
        recorded - the hallmark of a quick tap rather than a held recording -
        fall back to polishing whatever's currently selected with that same
        transform instead of silently discarding (Section 4.2's "highlight
        text, tap the transform's hotkey" behavior), reusing the exact
        no-selection/short-selection guard SelectionPolishWorker already has.
        """
        if not self._engines_ready:
            return
        self._active_transform_id = None
        audio_array = self.audio_worker.stop_recording()
        if self.config.audio_cues_enabled and audio_array is not None and len(audio_array) > 0:
            audio_cues.play_stop_cue()  # skipped on a quick tap (no real recording) - see the selection-polish fallback below
        # The captured-audio buffer's own length, not press-to-release wall-clock time - see _on_ptt_release.
        record_duration_ms = self.audio_worker.recorder.last_duration_seconds * 1000
        transform = self.db.get_transform(transform_id)
        if not transform:
            logger.warning(f"Transform '{transform_id}' no longer exists - ignoring.")
            self.state_changed.emit("idle")
            return

        if self._processing:
            logger.warning("Pipeline is already processing a previous utterance. Skipping.")
            return
        self._processing = True

        if audio_array is not None and len(audio_array) > 0:
            snippets: Dict[str, str] = {
                row["trigger_phrase"]: row["expansion_text"] for row in self.db.list_snippets(active_only=True)
            }
            worker = InferenceWorker(
                audio_array, self.transcriber, self.transformer, self.injector, self.db,
                snippets, transform_id, record_duration_ms, system_prompt=transform["system_prompt"],
                status_label=transform["title"],
            )
        else:
            logger.info(f"[TRANSFORM] No speech captured for '{transform_id}' - trying selected text instead.")
            worker = SelectionPolishWorker(
                self.injector, self.transformer, self.db, mode=transform_id,
                system_prompt=transform["system_prompt"],
            )

        worker.state_changed.connect(self.state_changed)
        worker.result_ready.connect(self.history_added)
        worker.finished.connect(self._on_inference_finished)
        self._inference_worker = worker
        worker.start()

    def refresh_transform_hotkeys(self) -> None:
        """Re-read the DB's transform shortcuts into the live hotkey poller (after the Transforms tab edits one)."""
        if self.hotkey_worker:
            self.hotkey_worker.set_transform_hotkeys(_build_transform_hotkeys(self.db))

    # --- Live Transcription (src/live_transcriber.py) --------------------------
    # Mutually exclusive with the normal dictation pipeline, deliberately: both
    # would otherwise call the SAME shared Transcriber/WhisperModel instance
    # concurrently from different threads, and faster-whisper's CTranslate2
    # backend is not documented as safe for that. _on_ptt_press()/
    # _on_transform_press() already refuse to start while self._live_worker is
    # set (above); the guard here is the mirror image for the other direction.

    def toggle_live_transcription(self) -> None:
        if self._live_worker is not None:
            self.stop_live_transcription()
        else:
            self.start_live_transcription()

    def start_live_transcription(self) -> None:
        if not self._engines_ready or self._processing or self._live_worker is not None:
            logger.warning("Live Transcription: cannot start - engines not ready or another pipeline is active.")
            return
        from src.live_transcriber import LiveTranscriptionWorker

        logger.info("[LIVE] Starting Live Transcription session...")
        worker = LiveTranscriptionWorker(self.transcriber, self.config.audio, transformer=self.transformer)
        worker.text_chunk_received.connect(self.live_transcribe_chunk_received)
        worker.speculative_text_changed.connect(self.live_transcribe_speculative_changed)
        worker.state_changed.connect(self.live_transcribe_state_changed)
        worker.session_stopped.connect(self._on_live_session_stopped)
        worker.finished.connect(self._on_live_worker_finished)
        self._live_worker = worker
        worker.start_session()

    def pause_live_transcription(self) -> None:
        if self._live_worker is not None:
            self._live_worker.pause()

    def resume_live_transcription(self) -> None:
        if self._live_worker is not None:
            self._live_worker.resume()

    def stop_live_transcription(self) -> None:
        if self._live_worker is not None:
            self._live_worker.stop_and_save()

    def discard_live_transcription(self) -> None:
        if self._live_worker is not None:
            self._live_worker.discard()

    # --- Writing Styles (src/writing_styles.py) ---------------------------------
    # Uses the same shared Transformer.llm instance InferenceWorker/SelectionPolishWorker
    # do, so it's gated by the same self._processing single-flight guard those use -
    # not a separate flag, to actually prevent a concurrent LLM call, not just look like it.

    def trigger_writing_styles(self) -> None:
        if not self._engines_ready or self._processing or self._live_worker is not None:
            logger.warning("Writing Styles: cannot start - engines not ready or another pipeline is active.")
            return
        self._processing = True
        worker = WritingStylesWorker(self.injector, self.transformer)
        worker.no_selection.connect(self._on_writing_styles_no_selection)
        worker.selection_captured.connect(self._on_writing_styles_selection_captured)
        worker.style_ready.connect(self.writing_styles_style_ready)
        worker.all_styles_done.connect(self.writing_styles_all_done)
        worker.finished.connect(self._on_writing_styles_worker_finished)
        self._writing_styles_worker = worker
        worker.start()

    def _on_writing_styles_no_selection(self) -> None:
        logger.info("[WRITING STYLES] No text selected - nothing to show.")
        self.writing_styles_no_selection.emit()

    def _on_writing_styles_selection_captured(self, text: str) -> None:
        self._writing_styles_selection = text
        self.writing_styles_selection_ready.emit(text)

    def _on_writing_styles_worker_finished(self) -> None:
        self._writing_styles_worker = None
        self._processing = False

    def select_writing_style(self, style_id: str, styled_text: str) -> None:
        """The user picked a style card (click or number-key) - paste it over
        the original selection and record it to history, tagged so History's
        badge reads e.g. '[Style: Professional]'."""
        from src.writing_styles import WRITING_STYLES_BY_ID

        injected = self.injector.inject_text(styled_text)
        title = WRITING_STYLES_BY_ID[style_id].title if style_id in WRITING_STYLES_BY_ID else style_id
        try:
            self.db.add_history(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                duration_seconds=0.0,
                raw_transcript=self._writing_styles_selection,
                polished_transcript=styled_text,
                mode_used=f"style_{style_id}",
                latency_ms=0.0,
                source="writing_style",
            )
        except Exception as e:
            logger.warning(f"Could not write history row for writing style: {e}")
        self.history_added.emit({
            "raw_transcript": self._writing_styles_selection, "polished_transcript": styled_text,
            "mode_used": f"style_{style_id}", "latency_ms": 0.0,
        })
        logger.info(f"[WRITING STYLES] Applied '{title}' style (injected={injected}).")

    def dismiss_writing_styles(self) -> None:
        if self._writing_styles_worker is not None:
            self._writing_styles_worker.cancel()

    def _on_live_session_stopped(self, result: dict) -> None:
        if not result or not result.get("raw", "").strip():
            logger.info("[LIVE] Session discarded or empty - nothing saved.")
            self.live_transcribe_session_saved.emit({})
            return
        try:
            history_id = self.db.add_live_transcript(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                duration_seconds=result["duration_seconds"],
                raw_transcript=result["raw"],
                polished_transcript=result["cleaned"],
            )
            logger.success(
                f"[LIVE] Session saved (#{history_id}, {result['duration_seconds']:.0f}s, {result['word_count']} words)."
            )
        except Exception as e:
            logger.error(f"[LIVE] Could not write live-transcript history row: {e}")
        self.history_added.emit({
            "raw_transcript": result["raw"], "polished_transcript": result["cleaned"],
            "mode_used": "live", "latency_ms": 0.0,
        })
        self.live_transcribe_session_saved.emit(result)

    def _on_live_worker_finished(self) -> None:
        self._live_worker = None

    def _on_mode_cycle(self, mode: str) -> None:
        self.active_mode = mode
        self.mode_changed.emit(mode)
        logger.info(f">>> Active Mode Changed to: [{self.active_mode.upper()}] <<<")

    # --- Controller interface (tray / settings / dashboard) --------------------

    def get_active_mode(self) -> str:
        return self.active_mode

    def set_active_mode(self, mode: str) -> None:
        self._on_mode_cycle(mode)

    def cycle_active_mode(self) -> None:
        idx = MODES.index(self.active_mode) if self.active_mode in MODES else 0
        self._on_mode_cycle(MODES[(idx + 1) % len(MODES)])

    def _dictation_hotkey_label(self) -> str:
        """Idle-pill label reflecting whichever dictation trigger(s) are enabled."""
        labels = []
        if self.config.tap_toggle_enabled:
            labels.append(self.config.tap_toggle_hotkey.upper())
        if self.config.hold_to_talk_enabled:
            labels.append(self.config.hold_to_talk_hotkey.upper())
        return " / ".join(labels) if labels else "No dictation hotkey enabled"

    # --- Dictation trigger settings: each auto-saves immediately (Settings ->
    # General has no manual Save button - see settings_tab.py) and rebinds the
    # live hotkey poller on the fly, no engine restart needed. ---

    def set_tap_toggle_hotkey(self, hotkey_str: str) -> None:
        self.config.tap_toggle_hotkey = hotkey_str.lower()
        if self.hotkey_worker:
            self.hotkey_worker.set_tap_toggle_hotkey(self.config.tap_toggle_hotkey)
        self.hotkey_changed.emit(self._dictation_hotkey_label())
        self.save_config()

    def set_tap_toggle_enabled(self, enabled: bool) -> None:
        self.config.tap_toggle_enabled = enabled
        if self.hotkey_worker:
            self.hotkey_worker.set_tap_toggle_enabled(enabled)
        self.hotkey_changed.emit(self._dictation_hotkey_label())
        self.save_config()

    def set_hold_to_talk_hotkey(self, hotkey_str: str) -> None:
        self.config.hold_to_talk_hotkey = hotkey_str.lower()
        if self.hotkey_worker:
            self.hotkey_worker.set_hold_to_talk_hotkey(self.config.hold_to_talk_hotkey)
        self.hotkey_changed.emit(self._dictation_hotkey_label())
        self.save_config()

    def set_hold_to_talk_enabled(self, enabled: bool) -> None:
        self.config.hold_to_talk_enabled = enabled
        if self.hotkey_worker:
            self.hotkey_worker.set_hold_to_talk_enabled(enabled)
        self.hotkey_changed.emit(self._dictation_hotkey_label())
        self.save_config()

    def set_minimize_to_tray(self, enabled: bool) -> None:
        self.config.minimize_to_tray = enabled
        self.save_config()

    def set_audio_device(self, device_index: Optional[int]) -> None:
        self.config.audio.device_index = device_index
        if self.audio_worker:
            self.audio_worker.set_device_index(device_index)
        self.save_config()

    def set_auto_silence_seconds(self, seconds: int) -> None:
        self.config.audio.auto_silence_seconds = seconds
        if self.audio_worker:
            self.audio_worker.set_auto_silence_seconds(seconds)
        self.save_config()

    def set_profanity_filter(self, mode: str) -> None:
        self.config.profanity_filter = mode
        if self.transformer:
            self.transformer.set_profanity_filter(mode)
        self.save_config()

    def set_auto_llm_polish(self, enabled: bool) -> None:
        """Read fresh on every dictation release - no restart needed, see _on_ptt_release."""
        self.config.auto_llm_polish = enabled
        self.save_config()

    def set_audio_cues_enabled(self, enabled: bool) -> None:
        self.config.audio_cues_enabled = enabled
        self.save_config()

    def set_auto_copy_to_clipboard(self, enabled: bool) -> None:
        self.config.injector.auto_copy_to_clipboard = enabled
        self.save_config()

    def save_config(self) -> None:
        save_config(self.config, self.config_path)

    def restart_engines(self) -> None:
        logger.info("Restarting AI engines...")
        self._engines_ready = False
        if self.hotkey_worker:
            self.hotkey_worker.stop()
        self.transcriber = None
        self.transformer = None
        self.state_changed.emit("loading")
        self._init_done_event.clear()
        threading.Thread(target=self._load_heavy_engines, daemon=True).start()
        threading.Thread(target=self._init_watchdog, daemon=True).start()

    def stop(self) -> None:
        if self.hotkey_worker:
            self.hotkey_worker.stop()
        if self._live_worker is not None:
            # Shutdown, not a user-requested stop: discard() is the only safe
            # choice here (stop_and_save() still needs the worker thread to
            # run its cleanup pass and emit session_stopped, which nothing
            # can wait on mid-teardown) - and wait for the native audio
            # stream it owns to actually close, same reasoning as the
            # audio_worker check below.
            self._live_worker.discard()
            self._live_worker.wait(2000)
        if self._writing_styles_worker is not None:
            self._writing_styles_worker.cancel()
            self._writing_styles_worker.wait(2000)
        # A recording left open at shutdown (app closed mid-dictation, before
        # the hotkey was released) leaves sounddevice's native PortAudio
        # callback thread running - that thread is outside Python's threading
        # module entirely, so nothing about `daemon=True` stops it, and it can
        # keep the OS process alive (observed as "Suspended" in Task Manager)
        # long after every Python thread has exited. Best-effort: closing it
        # here is cheap and safe even if nothing was recording.
        if self.audio_worker and self.audio_worker.recorder.is_recording:
            try:
                self.audio_worker.stop_recording()
            except Exception as e:
                logger.warning(f"Error stopping in-flight recording during shutdown: {e}")
        self.db.close()
        logger.info("Wisperno engine stopped gracefully.")
