"""
Main Entry Point for Wisperno (PySide6 desktop app).

Startup order is deliberate:
  0. OS-level fd 1/2 redirection (see below) - before ANY other import, since
     the native libraries loaded later (CTranslate2, llama.cpp) write straight
     to the OS file descriptors, not through Python's sys.stdout.
  1. Logging (file sink always; console sink only if a console exists) - set up
     before anything else so a failure during heavy imports is never silent.
  2. Single-instance mutex - before any model, audio device, or GUI work.
  3. QApplication + the floating pill, using only light imports, so the pill
     paints before torch/faster-whisper/llama-cpp are ever touched (those take
     15-25s in a frozen build; doing that at import time left users staring at
     nothing and double-clicking repeatedly).
  4. WispernoEngine.start() loads everything heavy on a background thread.
"""

import sys
import os

# --- 0. OS-level stdout/stderr fd redirection --------------------------------
# In a PyInstaller --noconsole build there is no console, so sys.stdout/stderr
# are None - but CTranslate2 (faster-whisper) and llama.cpp are C/C++ libraries
# that write initialization telemetry straight to OS file descriptors 1/2, not
# through Python's sys module. When those fds are left invalid/unattached (no
# console to receive them), the underlying WriteFile call can block for a very
# long time instead of failing fast: reproduced live as a 173-SECOND stall
# loading Whisper (normally ~5s) with zero errors logged, at the exact point
# CTranslate2 starts talking to the GPU. Redirecting fd 1/2 to a real, always-
# writable NUL device before any native library is imported gives those writes
# somewhere valid to land instantly instead of blocking.
if getattr(sys, "frozen", False) and sys.stdout is None:
    try:
        _devnull = open(os.devnull, "w")
        sys.stdout = _devnull
        sys.stderr = _devnull
        _null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(_null_fd, 1)
        os.dup2(_null_fd, 2)
        os.close(_null_fd)
    except Exception:
        pass

# --- 0b. Force offline mode before any AI library is imported ----------------
# faster-whisper/huggingface_hub still probe the Hub for a metadata/revision
# check even when the weights are already cached on disk, and that HTTPS
# request has no fast-fail on this machine - it hangs for the full Windows
# socket timeout (~173s, measured) before falling back to the local cache.
# Setting these before `faster_whisper`/`transformers` are ever imported skips
# that network probe entirely; confirmed via isolated timing test (174s -> 4.2s).
# NOTE: deliberately NOT setting CT2_CUDA_ALLOCATOR here - "cuda" is not a
# valid CTranslate2 allocator value (only "cuda_malloc"/"cuda_malloc_async"
# are), and setting it made CTranslate2 reject CUDA outright and silently
# fall back to CPU during a live rebuild verification. The offline env vars
# above are the actual, verified fix; this one was unverified and harmful.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from loguru import logger
from src.config import get_base_dir  # cheap import - no heavy ML deps

# Every path in this app already resolves off get_base_dir() (sys.executable's
# own directory when frozen, never os.getcwd() - see config.py), so nothing
# here actually depends on the process's working directory. This is a
# defensive belt-and-suspenders pin anyway: a shortcut launched with no
# explicit "Start in" (or a third-party native dependency that assumes cwd
# for some auxiliary lookup) should never be able to put this process
# somewhere unexpected.
try:
    os.chdir(str(get_base_dir()))
except OSError:
    pass

logger.remove()
if sys.stderr is not None:
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

log_dir = get_base_dir() / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    str(log_dir / "wisperno.log"),
    rotation="10 MB",
    retention="3 days",
    level="DEBUG",
    encoding="utf-8",
)
logger.info("Wisperno starting - logging initialized.")

from src import single_instance  # noqa: E402

single_instance.acquire()

# Windows groups taskbar entries and resolves the "this process's icon" by
# Application User Model ID rather than executable path alone - without an
# explicit AppUserModelID, Explorer falls back to a generic icon for windows
# from an unrecognized/ungrouped process. Must be set before QApplication
# creates the first native window.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Wisperno.Desktop.1.0")
    except Exception:
        pass

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.engine import WispernoEngine  # noqa: E402
from src.ui.floating_pill import FloatingPill  # noqa: E402
from src.ui.live_window import LiveTranscriptionWindow  # noqa: E402
from src.ui.writing_styles_window import WritingStylesWindow  # noqa: E402
from src.ui.tray import TrayIcon  # noqa: E402
from src.ui.theme import get_app_icon  # noqa: E402

logger.info("Light imports complete - showing UI before loading engines.")


class WispernoApp:
    """Top-level wiring: QApplication + pill + tray + engine + lazily-created dashboard."""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)  # closing the dashboard minimizes to tray, not exits
        self.app.setWindowIcon(get_app_icon())  # fallback icon for any window that doesn't set its own

        # Placeholder text color isn't a stylable QSS subcontrol (no placeholder-text-color
        # property) - it comes from QPalette::PlaceholderText, so it's set here once, app-wide,
        # rather than per-widget in every form that has a placeholder.
        from PySide6.QtGui import QPalette, QColor
        from src.ui import theme as _theme
        palette = self.app.palette()
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(_theme.TEXT_PLACEHOLDER))
        self.app.setPalette(palette)

        self.engine = WispernoEngine()
        self.pill = FloatingPill(
            initial_mode=self.engine.active_mode,
            hotkey_label=self.engine._dictation_hotkey_label(),
            always_on_top=self.engine.config.pill_always_on_top,
        )
        self.main_window = None

        self.live_window = LiveTranscriptionWindow()
        self.live_window.pause_clicked.connect(self.engine.pause_live_transcription)
        self.live_window.resume_clicked.connect(self.engine.resume_live_transcription)
        self.live_window.stop_save_clicked.connect(self.engine.stop_live_transcription)
        self.live_window.discard_clicked.connect(self.engine.discard_live_transcription)
        self.engine.live_transcribe_chunk_received.connect(self.live_window.append_chunk)
        self.engine.live_transcribe_speculative_changed.connect(self.live_window.set_speculative_text)
        self.engine.live_transcribe_state_changed.connect(self._on_live_state_changed)

        self.writing_styles_window = WritingStylesWindow()
        self.writing_styles_window.style_chosen.connect(self.engine.select_writing_style)
        self.writing_styles_window.dismissed.connect(self.engine.dismiss_writing_styles)
        self.engine.writing_styles_no_selection.connect(lambda: self.pill.set_state("no_selection"))
        self.engine.writing_styles_selection_ready.connect(self.writing_styles_window.show_selection)
        self.engine.writing_styles_style_ready.connect(self.writing_styles_window.set_style_ready)

        self.tray = TrayIcon(self)
        self.tray.open_dashboard_requested.connect(self.open_dashboard)

        self.engine.state_changed.connect(self.pill.set_state)
        self.engine.level_changed.connect(self.pill.set_level)
        self.engine.mode_changed.connect(self._on_mode_changed)
        self.engine.hotkey_changed.connect(self.pill.set_hotkey_label)
        self.engine.engines_ready.connect(lambda ok: self.tray.refresh())
        self.engine.settings_requested.connect(self.open_dashboard)
        self.engine.dashboard_toggle_requested.connect(self.toggle_dashboard)
        self.pill.dashboard_requested.connect(self.open_dashboard)
        self.pill.cycle_mode_requested.connect(self.engine.cycle_active_mode)
        self.pill.hide_pill_requested.connect(lambda: self.set_pill_visible(False))
        self.pill.close_requested.connect(self._on_pill_close_clicked)
        self.pill.quit_requested.connect(self.shutdown_application)

        self._shutting_down = False

        self.engine.start()

        # A direct double-click launch shows both the pill and the dashboard;
        # a login-triggered launch (autostart registers --background) starts
        # minimized so it doesn't pop a window in the user's face unprompted.
        if "--background" not in sys.argv:
            self.open_dashboard()

    def _on_mode_changed(self, mode: str) -> None:
        self.pill.set_mode(mode)
        self.tray.refresh()

    def open_dashboard(self) -> None:
        if self.main_window is None:
            from src.ui.main_window import MainWindow

            self.main_window = MainWindow(self.engine, self)
        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()

    def toggle_dashboard(self) -> None:
        if self.main_window is not None and self.main_window.isVisible():
            self.main_window.hide()
        else:
            self.open_dashboard()

    def toggle_live_transcription(self) -> None:
        """Single entry point for both the global hotkey (engine.toggle_live_transcription,
        wired straight to the hotkey manager) and the dashboard's sidebar button
        (this method) - the window itself only reacts to engine.live_transcribe_state_changed
        (see _on_live_state_changed), so both call sites stay in sync automatically."""
        self.engine.toggle_live_transcription()

    def _on_live_state_changed(self, state: str) -> None:
        if state == "listening":
            self.live_window.start_session_ui()
        elif state == "cleaning_up":
            self.live_window.show_cleaning_up()
        elif state == "stopped":
            self.live_window.close_session_ui()

    # --- Controller interface used by the tray + Settings tab ------------------

    def get_active_mode(self) -> str:
        return self.engine.get_active_mode()

    def set_active_mode(self, mode: str) -> None:
        self.engine.set_active_mode(mode)

    @property
    def config(self):
        return self.engine.config

    def set_tap_toggle_enabled(self, enabled: bool) -> None:
        self.engine.set_tap_toggle_enabled(enabled)
        self.tray.refresh()

    def set_hold_to_talk_enabled(self, enabled: bool) -> None:
        self.engine.set_hold_to_talk_enabled(enabled)
        self.tray.refresh()

    def toggle_pill(self) -> None:
        self.set_pill_visible(not self.pill.isVisible())

    def set_pill_visible(self, visible: bool) -> None:
        self.pill.set_pill_visible(visible)

    def set_pill_always_on_top(self, enabled: bool) -> None:
        self.pill.set_always_on_top(enabled)
        self.engine.config.pill_always_on_top = enabled
        self.engine.save_config()

    def _on_pill_close_clicked(self) -> None:
        if self.engine.config.minimize_to_tray:
            self.set_pill_visible(False)
        else:
            self.shutdown_application()

    def shutdown_application(self) -> None:
        """
        Single, idempotent teardown path for every way the app can be asked to
        quit (tray Exit, pill's Quit menu item, or the dashboard's close button
        when "Minimize to Tray" is off) - explicitly closes every top-level Qt
        window before quitting so nothing is ever left orphaned on screen with
        no way to reach it except Task Manager.
        """
        if self._shutting_down:
            return
        self._shutting_down = True

        logger.info("Executing graceful application shutdown...")
        self.engine.stop()
        single_instance.release()
        if self.main_window is not None:
            self.main_window.close()
        self.live_window.close()
        self.writing_styles_window.close()
        self.pill.close()
        self.tray.icon.hide()
        self.app.quit()

    def stop(self) -> None:
        """Fallback teardown for the normal app.exec()-returned-without-an-explicit-quit path."""
        self.shutdown_application()

    def run(self) -> int:
        return self.app.exec()


def main() -> None:
    wapp = WispernoApp()
    exit_code = wapp.run()
    wapp.stop()
    # os._exit(), not sys.exit(): sys.exit() only raises SystemExit on the
    # calling thread and waits for every OS thread to end before the process
    # actually terminates - but llama.cpp/CTranslate2/PortAudio can each hold
    # a native thread that Python's daemon-thread flag has no power over
    # (blocked in a C call, or PortAudio's own callback thread if a recording
    # was left open - see engine.stop()). That's what left Wisperno
    # "Suspended" in Task Manager needing multiple End Task clicks. By this
    # point wapp.stop() has already run every graceful step it can (hotkey
    # poller stopped, in-flight recording closed, DB WAL-checkpointed and
    # closed) - os._exit() then guarantees the process actually ends instead
    # of quietly hanging on whatever native thread refuses to.
    os._exit(exit_code)


if __name__ == "__main__":
    main()
