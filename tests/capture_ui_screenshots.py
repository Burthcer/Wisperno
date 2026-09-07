"""
Renders each dashboard tab offscreen and saves a PNG per tab, for visual
review against the redesign mockups (assets/screenshots/new assets/) - this
does NOT do automated pixel-diffing against those mockups (no reference
baseline images are checked into the repo to diff against, and a naive
pixel-diff would be brittle against font-rendering/DPI differences anyway);
it produces real rendered output a human (or a future round) can actually
look at side by side with the mockups.

Run: python tests/capture_ui_screenshots.py [output_dir]
(defaults to tests/ui_screenshots/)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _build_main_window():
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication
    from src.config import load_config, get_db_path
    from src.database import WispernoDB
    from src.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])

    class FakeController:
        def set_pill_visible(self, v):
            pass

        def shutdown_application(self):
            pass

        def toggle_live_transcription(self):
            pass

    class FakeEngine(QObject):
        history_added = Signal(dict)

        def __init__(self):
            super().__init__()
            self.config = load_config()
            self.db = WispernoDB(get_db_path())
            self.transformer = None
            self.transcriber = None
            self.audio_worker = None

        def set_tap_toggle_hotkey(self, h): pass
        def set_tap_toggle_enabled(self, e): pass
        def set_hold_to_talk_hotkey(self, h): pass
        def set_hold_to_talk_enabled(self, e): pass
        def set_audio_device(self, i): pass
        def set_profanity_filter(self, m): pass
        def set_auto_llm_polish(self, e): pass
        def set_minimize_to_tray(self, e): pass
        def save_config(self): pass
        def restart_engines(self): pass
        def refresh_transform_hotkeys(self): pass

    engine = FakeEngine()
    win = MainWindow(engine, FakeController())
    win.resize(1100, 720)
    win.show()
    win.system_usage_tab._timer.stop()  # outlives this script otherwise - see test_ui_layout.py's own note
    app.processEvents()
    return app, win


def capture_all(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    app, win = _build_main_window()

    for row in range(win.nav.count()):
        win.nav.setCurrentRow(row)
        app.processEvents()
        label = win.nav.item(row).text()
        pixmap = win.grab()
        out_path = output_dir / f"{row:02d}_{label.replace(' ', '_')}.png"
        pixmap.save(str(out_path))
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (BASE_DIR / "tests" / "ui_screenshots")
    print(f"Capturing dashboard tabs to {target}...")
    capture_all(target)
    print("Done.")
