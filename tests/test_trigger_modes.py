"""
Trigger Mechanic Verification for Wisperno.
Directly drives HotkeyManager's TAP_TO_TOGGLE state machine with a synthetic
raw key-state sequence (no real GetAsyncKeyState needed) to prove:
  1. A single tap starts recording and it stays started (no auto-stop).
  2. A momentary key-bounce glitch immediately after the first tap (raw signal
     flickers down/up within the anti-bounce window) does NOT count as a
     second tap - this was the reported "immediately stopped listening" bug,
     caused by using the shared hold-to-talk press-debounce for tap-mode edge
     detection instead of the raw signal.
  3. A genuine second tap, after the anti-bounce window, stops recording.

Run: python tests/test_trigger_modes.py
"""

import sys
import time
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.hotkey_manager import HotkeyManager, parse_hotkey_alternatives


def test_tap_to_toggle_stays_listening_until_second_tap() -> None:
    logger.info("--- Testing TAP_TO_TOGGLE: single tap starts and stays started ---")
    events = []
    mgr = HotkeyManager(
        on_press=lambda: events.append("press"),
        on_release=lambda: events.append("release"),
        tap_toggle_debounce_sec=0.2,
    )

    # Tap 1: key goes down.
    mgr._handle_tap_to_toggle(True, blocked=False)
    time.sleep(0.02)  # let the daemon thread running the callback finish
    assert mgr._is_toggled_recording, "First tap did not start toggled recording."
    assert events == ["press"], f"Expected exactly one 'press' event, got {events}"

    # Key stays held for a moment (should be a no-op every tick, not a second edge).
    for _ in range(5):
        mgr._handle_tap_to_toggle(True, blocked=False)
    assert mgr._is_toggled_recording, "Recording stopped while the key was still held down."
    assert events == ["press"], f"Holding the key fired extra events: {events}"

    logger.success("PASS: a single tap starts recording and it stays started while held/idle.")


def test_tap_to_toggle_ignores_bounce_glitch() -> None:
    logger.info("--- Testing TAP_TO_TOGGLE: a same-press bounce glitch is not a second tap ---")
    events = []
    mgr = HotkeyManager(
        on_press=lambda: events.append("press"),
        on_release=lambda: events.append("release"),
        tap_toggle_debounce_sec=0.2,
    )

    mgr._handle_tap_to_toggle(True, blocked=False)  # tap 1: start
    time.sleep(0.02)
    assert mgr._is_toggled_recording

    # Simulate a single-poll glitch: raw signal misreads as briefly up, then down
    # again, well inside the 200ms anti-bounce window (this is the exact shape of
    # the reported bug - a key-bounce or one missed GetAsyncKeyState read).
    mgr._handle_tap_to_toggle(False, blocked=False)
    mgr._handle_tap_to_toggle(True, blocked=False)
    time.sleep(0.02)

    assert mgr._is_toggled_recording, "A bounce glitch inside the debounce window incorrectly stopped recording."
    assert events == ["press"], f"Bounce glitch fired an extra event: {events}"
    logger.success("PASS: a bounce glitch within the anti-bounce window did not toggle recording off.")

    # Now a genuine second tap, after the anti-bounce window has elapsed.
    time.sleep(0.25)
    mgr._handle_tap_to_toggle(False, blocked=False)  # genuine release
    mgr._handle_tap_to_toggle(True, blocked=False)   # genuine second tap
    time.sleep(0.02)

    assert not mgr._is_toggled_recording, "A genuine second tap (after the debounce window) did not stop recording."
    assert events == ["press", "release"], f"Expected press then release, got {events}"
    logger.success("PASS: a genuine second tap (after the debounce window) stopped recording.")


def test_transform_hotkey_accepts_either_alternate_chord() -> None:
    """Alt+C and Alt+P must both fire the same 'polish' transform - one
    id -> multiple alternate chords (src/engine.py's POLISH_HOTKEY_ALIAS)."""
    logger.info("--- Testing per-transform hotkeys: 'alt+c|alt+p' fires on either chord ---")
    from src.engine import _build_transform_hotkeys

    class _FakeDB:
        def list_transforms(self, active_only=True):
            return [{"id": "polish", "shortcut": "alt+c", "active": 1}]

    hotkeys = _build_transform_hotkeys(_FakeDB())
    assert hotkeys["polish"] == "alt+c|alt+p", f"Alias was not appended: {hotkeys}"

    events = []
    mgr = HotkeyManager(
        on_transform_press=lambda tid: events.append(("press", tid)),
        on_transform_release=lambda tid: events.append(("release", tid)),
        transform_hotkeys=hotkeys,
    )
    vk_alternatives = mgr._transform_vks["polish"]
    assert len(vk_alternatives) == 2, f"Expected 2 alternate chords, got {vk_alternatives}"

    from src.hotkey_manager import parse_hotkey_to_vks
    alt_c_down = mgr._all_vks_down(parse_hotkey_to_vks("alt+c"))
    alt_p_down = mgr._all_vks_down(parse_hotkey_to_vks("alt+p"))
    assert alt_c_down is False and alt_p_down is False  # neither physically held during a test run
    assert parse_hotkey_to_vks("alt+c") in vk_alternatives
    assert parse_hotkey_to_vks("alt+p") in vk_alternatives
    logger.success("PASS: 'polish' accepts both Alt+C and Alt+P as alternate chords.")


def test_parse_hotkey_alternatives() -> None:
    logger.info("--- Testing parse_hotkey_alternatives(): '|'-separated chords parse independently ---")
    from src.hotkey_manager import parse_hotkey_to_vks

    alts = parse_hotkey_alternatives("alt+c|alt+p")
    assert alts == [parse_hotkey_to_vks("alt+c"), parse_hotkey_to_vks("alt+p")], alts
    assert parse_hotkey_alternatives("alt+c") == [parse_hotkey_to_vks("alt+c")]
    logger.success("PASS: '|'-separated hotkey alternatives parse into independent VK sets.")


def run_all_tests() -> None:
    logger.info("=========================================================")
    logger.info("   WISPERNO TRIGGER MECHANIC TEST SUITE")
    logger.info("=========================================================")
    test_tap_to_toggle_stays_listening_until_second_tap()
    test_tap_to_toggle_ignores_bounce_glitch()
    test_parse_hotkey_alternatives()
    test_transform_hotkey_accepts_either_alternate_chord()
    logger.success("=========================================================")
    logger.success(" ALL TRIGGER MECHANIC TESTS PASSED!")
    logger.success("=========================================================")


if __name__ == "__main__":
    run_all_tests()
