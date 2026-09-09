"""
Optional audio feedback for dictation start/stop (Settings -> General ->
Audio Cues). Uses winsound.Beep() - a synthesized tone, stdlib, Windows-only
(this app's only target platform already) - no audio asset files or new
dependency needed for two short tones.
"""

import sys
import threading

from loguru import logger

START_FREQ_HZ = 880   # A5 - a clean "listening started" tick
STOP_FREQ_HZ = 440    # A4 - a full octave down, "stopped" reads as lower/final
DURATION_MS = 90


def _beep(freq_hz: int) -> None:
    if sys.platform != "win32":
        return
    try:
        import winsound

        winsound.Beep(freq_hz, DURATION_MS)
    except Exception as e:
        logger.debug(f"Audio cue playback failed: {e}")


def play_start_cue() -> None:
    """Fire-and-forget - winsound.Beep() blocks its calling thread for
    DURATION_MS, so this must never run on the hotkey callback thread."""
    threading.Thread(target=_beep, args=(START_FREQ_HZ,), daemon=True).start()


def play_stop_cue() -> None:
    threading.Thread(target=_beep, args=(STOP_FREQ_HZ,), daemon=True).start()
