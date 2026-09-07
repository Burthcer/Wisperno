"""
Qt worker layer for Wisperno. Keeps the GUI thread free for 60fps rendering by
running audio capture, hotkey polling, and the STT->LLM->injection pipeline
off the main thread, communicating back via Qt signals (thread-safe: PySide6
auto-marshals a signal emitted on a worker thread into a queued call on the
thread that owns the connected slot).

AudioWorker and HotkeyWorker wrap the existing, independently-tested
AudioRecorder / HotkeyManager (sounddevice's callback and the GetAsyncKeyState
poll already run on their own non-Qt threads) rather than re-implementing that
logic as QThread subclasses - the win from QThread here is signal delivery,
not the threading itself, which was already correct.

InferenceWorker (the main dictation hotkey) runs one of two pipelines after
Whisper: the instant, zero-LLM src/direct_formatter.py (Mode 1 - default), or
the full src/transformer.py LLM polish (opt-in via config.auto_llm_polish, or
always for a Transforms-hub hotkey's own prompt). SelectionPolishWorker
(the dedicated Polish Selected Text hotkey) always uses the LLM.
"""

import time
from typing import Dict, Optional

from PySide6.QtCore import QObject, QThread, Signal
from loguru import logger

from src.audio_recorder import AudioRecorder
from src.direct_formatter import format_direct
from src.hotkey_manager import HotkeyManager
from src.config import AppConfig
from src.swear_filter import apply_profanity_filter


class AudioWorker(QObject):
    """Wraps AudioRecorder; emits level_changed(float) on every audio block during recording,
    and auto_silence_triggered() once per recording if the user pauses past their configured
    auto-stop threshold. Both callbacks fire from PortAudio's own native callback thread - Qt
    signals are safe to emit cross-thread (auto-queued onto the receiving thread)."""

    level_changed = Signal(float)
    auto_silence_triggered = Signal()

    def __init__(self, config, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.recorder = AudioRecorder(
            config=config, level_callback=self.level_changed.emit,
            silence_callback=self.auto_silence_triggered.emit,
        )

    def start_recording(self) -> bool:
        return self.recorder.start_recording()

    def stop_recording(self):
        return self.recorder.stop_recording()

    def set_device_index(self, device_index: Optional[int]) -> None:
        """Live device swap for Settings -> General's auto-save (no engine restart
        needed - the next start_recording() call picks it up)."""
        self.recorder.device_index = device_index

    def set_auto_silence_seconds(self, seconds: int) -> None:
        self.recorder.set_auto_silence_seconds(seconds)


class HotkeyWorker(QObject):
    """Wraps HotkeyManager; translates its callbacks into Qt signals."""

    recording_started = Signal()
    recording_stopped = Signal()
    mode_cycled = Signal(str)
    settings_requested = Signal()
    dashboard_toggle_requested = Signal()
    live_transcribe_toggle_requested = Signal()
    transform_press_requested = Signal(str)
    transform_release_requested = Signal(str)

    def __init__(self, config: AppConfig, transform_hotkeys: Optional[Dict[str, str]] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.manager = HotkeyManager(
            tap_toggle_hotkey=config.tap_toggle_hotkey,
            tap_toggle_enabled=config.tap_toggle_enabled,
            hold_to_talk_hotkey=config.hold_to_talk_hotkey,
            hold_to_talk_enabled=config.hold_to_talk_enabled,
            cycle_mode_hotkey=config.cycle_mode_hotkey,
            settings_hotkey=config.settings_hotkey,
            dashboard_hotkey=config.dashboard_hotkey,
            live_transcribe_hotkey=config.live_transcribe_hotkey,
            transform_hotkeys=transform_hotkeys or {},
            on_press=self.recording_started.emit,
            on_release=self.recording_stopped.emit,
            on_mode_cycle=self.mode_cycled.emit,
            on_settings_open=self.settings_requested.emit,
            on_dashboard_toggle=self.dashboard_toggle_requested.emit,
            on_live_transcribe_toggle=self.live_transcribe_toggle_requested.emit,
            on_transform_press=self.transform_press_requested.emit,
            on_transform_release=self.transform_release_requested.emit,
            available_modes=["polish", "prompt_engineer", "bullets", "code", "raw"],
            initial_mode=config.active_mode,
        )

    def start(self) -> None:
        self.manager.start()

    def stop(self) -> None:
        self.manager.stop()

    def set_tap_toggle_hotkey(self, hotkey_str: str) -> None:
        self.manager.set_tap_toggle_hotkey(hotkey_str)

    def set_tap_toggle_enabled(self, enabled: bool) -> None:
        self.manager.set_tap_toggle_enabled(enabled)

    def set_hold_to_talk_hotkey(self, hotkey_str: str) -> None:
        self.manager.set_hold_to_talk_hotkey(hotkey_str)

    def set_hold_to_talk_enabled(self, enabled: bool) -> None:
        self.manager.set_hold_to_talk_enabled(enabled)

    def set_transform_hotkeys(self, transform_hotkeys: Dict[str, str]) -> None:
        self.manager.set_transform_hotkeys(transform_hotkeys)


class InferenceWorker(QThread):
    """
    Runs one full pipeline pass off the GUI thread:
    audio -> Whisper STT -> vocabulary correction -> snippet expansion (bypasses
    formatting entirely when a trigger matches) -> direct_formatter OR SLM
    transform (see use_direct_formatter) -> Win32 paste injection -> SQLite
    history insert.
    """

    state_changed = Signal(str)          # 'processing' | 'success' | 'idle'
    result_ready = Signal(dict)          # history row payload, for the History tab
    error = Signal(str)

    def __init__(
        self, audio_array, transcriber, transformer, injector, db, snippets,
        mode: str, record_duration_ms: float, system_prompt: Optional[str] = None,
        use_direct_formatter: bool = False, status_label: Optional[str] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.audio_array = audio_array
        self.transcriber = transcriber
        self.transformer = transformer
        self.injector = injector
        self.db = db
        self.snippets = snippets  # dict[str, str]: lowercase trigger -> expansion
        self.mode = mode
        self.record_duration_ms = record_duration_ms
        # When set (a Transforms-hub entry triggered this recording directly,
        # e.g. Alt+X), use its own prompt instead of looking `mode` up in
        # config.yaml - lets DB-defined transforms drive dictation directly.
        self.system_prompt = system_prompt
        # A Transforms-hub entry's display title (e.g. "Prompt Engineer") -
        # carried on the "processing" pill state as "processing:<label>" so
        # the pill shows "Prompt Engineering..." instead of a generic message.
        self.status_label = status_label
        # Mode 1 (Instant Direct Dictation): skip the LLM entirely and run
        # src/direct_formatter.py instead - set by the engine from the
        # `auto_llm_polish` config toggle, only for the main dictation hotkey
        # (never for a Transforms-hub hotkey, which is an explicit LLM gesture).
        self.use_direct_formatter = use_direct_formatter

    def run(self) -> None:
        try:
            if self.use_direct_formatter:
                self.state_changed.emit("typing")
            elif self.status_label:
                self.state_changed.emit(f"processing:{self.status_label}")
            else:
                self.state_changed.emit("processing")
            t0 = time.perf_counter()

            raw_text = self.transcriber.transcribe(self.audio_array)
            if not raw_text or not raw_text.strip():
                logger.info("No speech detected in audio segment.")
                self.state_changed.emit("idle")
                return

            # Snippet expansion bypasses formatting entirely when the transcript matches a trigger.
            normalized = raw_text.strip().lower().rstrip(".!? ")
            if normalized in self.snippets:
                final_text = self.snippets[normalized]
                logger.info(f"Snippet matched '{normalized}' -> bypassing formatting.")
            elif self.use_direct_formatter:
                # Mode 1: instant, zero-LLM rule-based cleanup (see src/direct_formatter.py).
                profanity_mode = getattr(self.transformer, "profanity_filter", "allow")
                final_text = apply_profanity_filter(format_direct(raw_text), profanity_mode)
            elif self.system_prompt:
                final_text = self.transformer.transform_with_prompt(raw_text, self.system_prompt, label=self.mode)
            else:
                final_text = self.transformer.transform(raw_text, mode=self.mode)

            injected = self.injector.inject_text(final_text)
            latency_ms = (time.perf_counter() - t0) * 1000

            # A Transforms-hub hotkey drove this recording (status_label set) vs
            # the plain dictation hotkey (active mode, no transform) - see
            # database.py's add_history() docstring for what each source means.
            history_source = "transform" if self.status_label else "dictation"
            if self.db:
                try:
                    self.db.add_history(
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        duration_seconds=self.record_duration_ms / 1000.0,
                        raw_transcript=raw_text,
                        polished_transcript=final_text,
                        mode_used=self.mode,
                        latency_ms=latency_ms,
                        source=history_source,
                    )
                except Exception as e:
                    logger.warning(f"Could not write history row: {e}")

            self.result_ready.emit({
                "raw_transcript": raw_text,
                "polished_transcript": final_text,
                "mode_used": self.mode,
                "latency_ms": latency_ms,
            })
            # A soft guardrail fallback (rare now - see sanity_check()'s narrowed
            # scope) still injected working, correctly-capitalized text, just not
            # LLM-polished - shown as a normal success, not a warning pill. The
            # warning pill is reserved for the except block below (an actual
            # unhandled failure), per the mission's own "don't flash an annoying
            # warning pill unless something actually broke" instruction.
            # [FALLBACK_DEBUG] in the log is where a soft fallback stays visible.
            self.state_changed.emit("success" if injected else "idle")

        except Exception as e:
            logger.error(f"InferenceWorker pipeline error: {e}", exc_info=True)
            self.error.emit(str(e))
            self.state_changed.emit("idle")


class SelectionPolishWorker(QThread):
    """
    "Polish Selected Text" pipeline, off the GUI thread: Ctrl+C the current
    selection -> SLM polish -> Ctrl+V it back in place -> SQLite history insert
    (mode 'text_polish'). Shares the engine's single-flight guard with
    InferenceWorker since both drive the same non-thread-safe Llama instance.
    """

    state_changed = Signal(str)   # 'selection_processing' | 'selection_success' | 'idle'
    result_ready = Signal(dict)
    error = Signal(str)

    def __init__(
        self, injector, transformer, db, mode: str = "polish",
        system_prompt: Optional[str] = None, parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.injector = injector
        self.transformer = transformer
        self.db = db
        self.mode = mode
        self.system_prompt = system_prompt  # set -> a specific Transforms-hub entry drove this, not config.yaml's mode

    def run(self) -> None:
        try:
            self.state_changed.emit("selection_processing")
            t0 = time.perf_counter()

            selected_text = self.injector.copy_selection()
            if not selected_text or len(selected_text.strip()) < 2:
                logger.info("Polish Selected Text: nothing usable was selected - dismissing.")
                self.state_changed.emit("idle")
                return

            if self.system_prompt:
                polished_text = self.transformer.transform_with_prompt(selected_text, self.system_prompt, label=self.mode)
            else:
                polished_text = self.transformer.transform(selected_text, mode=self.mode)
            injected = self.injector.inject_text(polished_text)
            latency_ms = (time.perf_counter() - t0) * 1000

            if self.db:
                try:
                    self.db.add_history(
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        duration_seconds=0.0,
                        raw_transcript=selected_text,
                        polished_transcript=polished_text,
                        mode_used="text_polish",
                        latency_ms=latency_ms,
                        source="text_polish",
                    )
                except Exception as e:
                    logger.warning(f"Could not write history row for selection polish: {e}")

            self.result_ready.emit({
                "raw_transcript": selected_text,
                "polished_transcript": polished_text,
                "mode_used": "text_polish",
                "latency_ms": latency_ms,
            })
            # See InferenceWorker.run() for why a soft guardrail fallback shows
            # a normal success state instead of a warning pill.
            self.state_changed.emit("selection_success" if injected else "idle")

        except Exception as e:
            logger.error(f"SelectionPolishWorker pipeline error: {e}", exc_info=True)
            self.error.emit(str(e))
            self.state_changed.emit("idle")
