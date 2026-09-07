"""
Native Win32 Push-to-Talk Hotkey Manager for Wisperno.
Uses a high-precision background polling thread with GetAsyncKeyState for 100% reliable hotkey capture.
Supports two trigger mechanics (HOLD_TO_TALK / TAP_TO_TOGGLE) and runtime hotkey/mode rebinding.
"""

import ctypes
import threading
import time
from typing import Callable, Dict, List, Optional, Set
from loguru import logger

# Win32 Virtual Key Codes
VK_MAP = {
    "ctrl": 0x11,
    "control": 0x11,
    "shift": 0x10,
    "alt": 0x12,
    "space": 0x20,
    "comma": 0xBC,
    ",": 0xBC,
    "caps_lock": 0x14,
    "scroll_lock": 0x91,
    "insert": 0x2D,
    "right_alt": 0xA5,
    "win": 0x5B,  # VK_LWIN
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
    "esc": 0x1B,
    "tab": 0x09,
}

# Add standard letters A-Z (0x41 .. 0x5A)
for char_code in range(ord("a"), ord("z") + 1):
    VK_MAP[chr(char_code)] = 0x41 + (char_code - ord("a"))

# Add digits 0-9 (0x30 .. 0x39)
for digit in range(10):
    VK_MAP[str(digit)] = 0x30 + digit

user32 = ctypes.windll.user32


def parse_hotkey_to_vks(hotkey_str: str) -> Set[int]:
    """Parse hotkey string like 'ctrl+b' into set of Win32 virtual key codes."""
    vks = set()
    for part in hotkey_str.lower().split("+"):
        part = part.strip()
        if not part:
            continue
        if part in VK_MAP:
            vks.add(VK_MAP[part])
        elif len(part) == 1 and part.isalnum():
            vks.add(ord(part.upper()))
        else:
            logger.warning(f"Unknown key in hotkey: '{part}'")
    return vks


def parse_hotkey_alternatives(hotkey_str: str) -> List[Set[int]]:
    """Parse '|'-separated alternate chords (e.g. 'alt+c|alt+p') into a list of
    VK sets, any one of which can trigger the same action - lets a single
    transform accept more than one muscle-memory shortcut."""
    return [parse_hotkey_to_vks(alt) for alt in hotkey_str.split("|") if alt.strip()]


class HotkeyManager:
    """
    High-reliability Win32 hotkey manager using GetAsyncKeyState polling.
    Supports HOLD_TO_TALK (press/release) and TAP_TO_TOGGLE (tap to start, tap to stop)
    trigger mechanics, plus runtime rebinding of the hotkey and trigger mode.
    """

    def __init__(
        self,
        tap_toggle_hotkey: str = "ctrl+alt",
        tap_toggle_enabled: bool = True,
        hold_to_talk_hotkey: str = "ctrl+space",
        hold_to_talk_enabled: bool = False,
        cycle_mode_hotkey: str = "ctrl+shift+b",
        settings_hotkey: Optional[str] = None,
        dashboard_hotkey: Optional[str] = None,
        live_transcribe_hotkey: Optional[str] = None,
        on_press: Optional[Callable[[], None]] = None,
        on_release: Optional[Callable[[], None]] = None,
        on_mode_cycle: Optional[Callable[[str], None]] = None,
        on_settings_open: Optional[Callable[[], None]] = None,
        on_dashboard_toggle: Optional[Callable[[], None]] = None,
        on_live_transcribe_toggle: Optional[Callable[[], None]] = None,
        transform_hotkeys: Optional[Dict[str, str]] = None,
        on_transform_press: Optional[Callable[[str], None]] = None,
        on_transform_release: Optional[Callable[[str], None]] = None,
        available_modes: Optional[List[str]] = None,
        initial_mode: str = "polish",
        poll_interval_sec: float = 0.01,  # 10ms polling interval - worst-case detection latency stays under 20ms
        debounce_sec: float = 0.05,  # ghost-trigger guard: chord must stay down this long before firing
        tap_toggle_debounce_sec: float = 0.2,  # min gap between TAP_TO_TOGGLE fires (key-bounce guard)
        max_recording_sec: float = 600.0,  # safety auto-stop if a key-up event is ever missed (10 min ceiling)
    ):
        # Two fully independent dictation triggers - each with its own chord,
        # its own always-on behavior (tap-toggle vs hold-to-talk), and its own
        # enable switch. Both call the same on_press/on_release callbacks;
        # `_active_trigger` (set below) stops one from starting a second
        # recording while the other's is already in flight.
        self.tap_toggle_str = tap_toggle_hotkey
        self.tap_toggle_enabled = tap_toggle_enabled
        self.hold_to_talk_str = hold_to_talk_hotkey
        self.hold_to_talk_enabled = hold_to_talk_enabled
        self.cycle_mode_str = cycle_mode_hotkey
        self.settings_hotkey_str = settings_hotkey
        self.dashboard_hotkey_str = dashboard_hotkey
        self.live_transcribe_hotkey_str = live_transcribe_hotkey
        self.on_press_callback = on_press
        self.on_release_callback = on_release
        self.on_mode_cycle_callback = on_mode_cycle
        self.on_settings_open_callback = on_settings_open
        self.on_dashboard_toggle_callback = on_dashboard_toggle
        self.on_live_transcribe_toggle_callback = on_live_transcribe_toggle
        self.on_transform_press_callback = on_transform_press
        self.on_transform_release_callback = on_transform_release
        self.poll_interval = poll_interval_sec
        self.debounce_sec = debounce_sec
        self.tap_toggle_debounce_sec = tap_toggle_debounce_sec
        self.max_recording_sec = max_recording_sec

        self.available_modes = available_modes or ["polish", "prompt_engineer", "bullets", "raw"]
        self.current_mode_idx = (
            self.available_modes.index(initial_mode) if initial_mode in self.available_modes else 0
        )

        self._tap_vks = parse_hotkey_to_vks(self.tap_toggle_str)
        self._hold_vks = parse_hotkey_to_vks(self.hold_to_talk_str)
        self._cycle_vks = parse_hotkey_to_vks(self.cycle_mode_str)
        self._settings_vks = parse_hotkey_to_vks(self.settings_hotkey_str) if self.settings_hotkey_str else set()
        self._dashboard_vks = parse_hotkey_to_vks(self.dashboard_hotkey_str) if self.dashboard_hotkey_str else set()
        self._live_transcribe_vks = (
            parse_hotkey_to_vks(self.live_transcribe_hotkey_str) if self.live_transcribe_hotkey_str else set()
        )
        # Per-transform hotkeys (e.g. Alt+C -> "polish", Alt+X -> "prompt_engineer"):
        # each held like a dictation hotkey but tagged with which transform to
        # apply, so an arbitrary (DB-driven, user-creatable) number of these can
        # coexist without one fixed-name trigger per transform.
        self._transform_vks: Dict[str, List[Set[int]]] = (
            {tid: parse_hotkey_alternatives(hk) for tid, hk in transform_hotkeys.items() if hk} if transform_hotkeys else {}
        )
        self._is_transform_down: Dict[str, bool] = {tid: False for tid in self._transform_vks}
        self._transform_active_since: Dict[str, float] = {}

        # Which dictation trigger (if any) currently owns an in-flight
        # recording - blocks the OTHER trigger from also starting one, e.g.
        # if both are enabled and physically overlap.
        self._active_trigger: Optional[str] = None  # "tap" | "hold" | None

        self._is_hold_active = False       # HOLD_TO_TALK: chord currently held
        self._is_tap_edge_down = False     # TAP_TO_TOGGLE: chord was down last tick (for edge detection)
        self._is_toggled_recording = False  # TAP_TO_TOGGLE: recording currently active
        self._is_cycle_down = False
        self._is_settings_down = False
        self._is_dashboard_down = False
        self._is_live_transcribe_down = False
        self._down_since: Dict[str, float] = {}  # debounce bookkeeping, keyed by trigger name
        self._hold_active_since = 0.0
        self._toggle_active_since = 0.0
        self._last_tap_toggle_fire = 0.0  # perf_counter of the last accepted TAP_TO_TOGGLE edge
        self._await_hold_release = False  # after a safety-timeout force-stop, ignore the chord until it's physically released
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    @property
    def current_mode(self) -> str:
        return self.available_modes[self.current_mode_idx]

    # --- Runtime rebinding (safe to call from any thread; attribute swaps are atomic) ---

    def set_tap_toggle_hotkey(self, hotkey_str: str) -> None:
        """Rebind the tap-to-toggle dictation hotkey without restarting the polling thread."""
        self.tap_toggle_str = hotkey_str
        self._tap_vks = parse_hotkey_to_vks(hotkey_str)
        self._is_tap_edge_down = False
        if self._active_trigger == "tap" and self._is_toggled_recording and self.on_release_callback:
            self._is_toggled_recording = False
            self._active_trigger = None
            threading.Thread(target=self.on_release_callback, daemon=True).start()
        self._down_since.pop("tap", None)
        logger.info(f"Hotkey rebound: Tap-to-Toggle -> '{hotkey_str.upper()}'.")

    def set_tap_toggle_enabled(self, enabled: bool) -> None:
        if not enabled and self._active_trigger == "tap" and self._is_toggled_recording and self.on_release_callback:
            self._is_toggled_recording = False
            self._active_trigger = None
            threading.Thread(target=self.on_release_callback, daemon=True).start()
        self.tap_toggle_enabled = enabled
        logger.info(f"Tap-to-Toggle dictation {'enabled' if enabled else 'disabled'}.")

    def set_hold_to_talk_hotkey(self, hotkey_str: str) -> None:
        """Rebind the hold-to-talk dictation hotkey without restarting the polling thread."""
        self.hold_to_talk_str = hotkey_str
        self._hold_vks = parse_hotkey_to_vks(hotkey_str)
        self._is_hold_active = False
        self._await_hold_release = False
        self._down_since.pop("hold", None)
        logger.info(f"Hotkey rebound: Hold-to-Talk -> '{hotkey_str.upper()}'.")

    def set_hold_to_talk_enabled(self, enabled: bool) -> None:
        if not enabled and self._active_trigger == "hold" and self._is_hold_active and self.on_release_callback:
            self._is_hold_active = False
            self._active_trigger = None
            threading.Thread(target=self.on_release_callback, daemon=True).start()
        self.hold_to_talk_enabled = enabled
        logger.info(f"Hold-to-Talk dictation {'enabled' if enabled else 'disabled'}.")

    def set_transform_hotkeys(self, transform_hotkeys: Dict[str, str]) -> None:
        """Replace the whole transform-hotkey map (e.g. after the Transforms tab edits a shortcut)."""
        self._transform_vks = {tid: parse_hotkey_alternatives(hk) for tid, hk in transform_hotkeys.items() if hk}
        self._is_transform_down = {tid: False for tid in self._transform_vks}
        self._transform_active_since = {}
        for tid in list(self._down_since):
            if tid.startswith("transform:"):
                self._down_since.pop(tid, None)
        logger.info(f"Transform hotkeys rebound: {[(t, h.upper()) for t, h in transform_hotkeys.items()]}.")

    def force_stop_active_recording(self) -> None:
        """External stop (the Auto-Stop Silence Detection watchdog already ended
        the recording and ran the pipeline itself) - this only clears whichever
        trigger's own "currently recording" bookkeeping is active, so a later
        physical key-release/tap doesn't misinterpret the chord as still owning
        an in-flight recording and double-fire a redundant stop."""
        if self._active_trigger == "hold":
            self._is_hold_active = False
        elif self._active_trigger == "tap":
            self._is_toggled_recording = False
        self._active_trigger = None

    def _is_vk_down(self, vk: int) -> bool:
        """Check if virtual key is currently physically held down (high bit of GetAsyncKeyState)."""
        return bool(user32.GetAsyncKeyState(vk) & 0x8000)

    def _all_vks_down(self, vks: Set[int]) -> bool:
        """Check if all virtual keys in set are currently, physically held down simultaneously."""
        if not vks:
            return False
        for vk in vks:
            if not self._is_vk_down(vk):
                return False
        return True

    def _debounced(self, key: str, raw_active: bool) -> bool:
        """
        Require `raw_active` to hold continuously for `debounce_sec` before reporting True.
        Filters momentary power-state transitions / focus glitches that could otherwise
        cause ghost activation. Release is reported immediately (no debounce on the way down).
        """
        if not raw_active:
            self._down_since.pop(key, None)
            return False
        since = self._down_since.get(key)
        if since is None:
            self._down_since[key] = time.perf_counter()
            return False
        return (time.perf_counter() - since) >= self.debounce_sec

    def _handle_hold_to_talk(self, hold_active: bool, blocked: bool) -> None:
        if not hold_active:
            self._await_hold_release = False  # physical release confirmed; clear any pending cooldown

        if hold_active and not blocked and not self._await_hold_release:
            if not self._is_hold_active:
                self._is_hold_active = True
                self._active_trigger = "hold"
                self._hold_active_since = time.perf_counter()
                logger.info(f"[HOTKEY] Pressed {self.hold_to_talk_str.upper()} -> Recording started...")
                if self.on_press_callback:
                    threading.Thread(target=self.on_press_callback, daemon=True).start()
            elif time.perf_counter() - self._hold_active_since > self.max_recording_sec:
                logger.warning(
                    f"[HOTKEY] Safety timeout: {self.hold_to_talk_str.upper()} has been held for over "
                    f"{self.max_recording_sec:.0f}s (missed key-up?) - force-stopping."
                )
                self._is_hold_active = False
                self._active_trigger = None
                self._await_hold_release = True
                if self.on_release_callback:
                    threading.Thread(target=self.on_release_callback, daemon=True).start()
        elif self._is_hold_active and not hold_active:
            self._is_hold_active = False
            self._active_trigger = None
            logger.info(f"[HOTKEY] Released {self.hold_to_talk_str.upper()} -> Processing audio pipeline...")
            if self.on_release_callback:
                threading.Thread(target=self.on_release_callback, daemon=True).start()

    def _handle_tap_to_toggle(self, tap_raw_down: bool, blocked: bool) -> None:
        """
        Edge-detects a tap from the RAW (undebounced) key state, not the shared
        hold-to-talk debounce. That debounce requires the chord to read as down
        for a continuous debounce_sec (50ms) before it reports True, and reports
        False the instant even a single poll misses - which is exactly the wrong
        shape for a quick tap: a genuine tap is often *shorter* than the debounce
        window, and any one-tick read glitch mid-hold makes `_is_tap_edge_down`
        drop and immediately re-rise, firing a second toggle for one physical
        press (this was the reported "immediately stopped listening" bug - the
        second, spurious toggle followed the first close enough to look instant).
        Bounce protection here instead rate-limits the FIRE, not the press: two
        edges within tap_toggle_debounce_sec of each other are treated as one.
        """
        if self._is_toggled_recording and time.perf_counter() - self._toggle_active_since > self.max_recording_sec:
            logger.warning(
                f"[HOTKEY] Safety timeout: toggled recording has been active for over "
                f"{self.max_recording_sec:.0f}s - force-stopping."
            )
            self._is_toggled_recording = False
            self._active_trigger = None
            self._is_tap_edge_down = tap_raw_down
            if self.on_release_callback:
                threading.Thread(target=self.on_release_callback, daemon=True).start()
            return

        # Edge-detect the tap on the raw signal; ignore repeats while held down.
        rising_edge = tap_raw_down and not self._is_tap_edge_down and not blocked
        self._is_tap_edge_down = tap_raw_down
        if not rising_edge:
            return

        now = time.perf_counter()
        if now - self._last_tap_toggle_fire < self.tap_toggle_debounce_sec:
            return  # key-bounce guard: too soon after the last accepted tap
        self._last_tap_toggle_fire = now

        self._is_toggled_recording = not self._is_toggled_recording
        if self._is_toggled_recording:
            self._active_trigger = "tap"
            self._toggle_active_since = time.perf_counter()
            logger.info(f"[HOTKEY] Tapped {self.tap_toggle_str.upper()} -> Recording started (toggle)...")
            if self.on_press_callback:
                threading.Thread(target=self.on_press_callback, daemon=True).start()
        else:
            self._active_trigger = None
            logger.info(f"[HOTKEY] Tapped {self.tap_toggle_str.upper()} -> Processing audio pipeline (toggle)...")
            if self.on_release_callback:
                threading.Thread(target=self.on_release_callback, daemon=True).start()

    def _polling_loop(self) -> None:
        """Continuous 10ms polling loop for instant hotkey tracking."""
        logger.info(
            f"Win32 GetAsyncKeyState polling thread started "
            f"[Tap-to-Toggle: '{self.tap_toggle_str.upper()}' ({'on' if self.tap_toggle_enabled else 'off'}), "
            f"Hold-to-Talk: '{self.hold_to_talk_str.upper()}' ({'on' if self.hold_to_talk_enabled else 'off'}), "
            f"Cycle: '{self.cycle_mode_str.upper()}']."
        )

        while not self._stop_event.is_set():
            try:
                cycle_active = self._debounced("cycle", self._all_vks_down(self._cycle_vks))
                settings_active = self._debounced(
                    "settings", self._all_vks_down(self._settings_vks) if self._settings_vks else False
                )
                dashboard_active = self._debounced(
                    "dashboard", self._all_vks_down(self._dashboard_vks) if self._dashboard_vks else False
                )
                live_transcribe_active = self._debounced(
                    "live_transcribe",
                    self._all_vks_down(self._live_transcribe_vks) if self._live_transcribe_vks else False,
                )
                tap_raw_down = self._all_vks_down(self._tap_vks) if self.tap_toggle_enabled else False
                hold_raw_down = self._all_vks_down(self._hold_vks) if self.hold_to_talk_enabled else False
                # HOLD_TO_TALK still uses the press-debounce (guards against a ghost
                # activation being held "on" by noise); TAP_TO_TOGGLE uses tap_raw_down
                # directly - see _handle_tap_to_toggle's docstring for why.
                hold_active = self._debounced("hold", hold_raw_down)

                # 1. Check Cycle Mode (e.g. Ctrl+Shift+B)
                if cycle_active:
                    if not self._is_cycle_down:
                        self._is_cycle_down = True
                        self.current_mode_idx = (self.current_mode_idx + 1) % len(self.available_modes)
                        new_mode = self.current_mode
                        logger.info(f"[HOTKEY] Cycled Mode -> [{new_mode.upper()}]")
                        if self.on_mode_cycle_callback:
                            threading.Thread(target=self.on_mode_cycle_callback, args=(new_mode,), daemon=True).start()
                else:
                    self._is_cycle_down = False

                # 2. Check Open Settings (e.g. Ctrl+,)
                if settings_active:
                    if not self._is_settings_down:
                        self._is_settings_down = True
                        logger.info("[HOTKEY] Opening Settings & Dictionary window.")
                        if self.on_settings_open_callback:
                            threading.Thread(target=self.on_settings_open_callback, daemon=True).start()
                else:
                    self._is_settings_down = False

                # 2b. Check Toggle Dashboard (e.g. Ctrl+Shift+H)
                if dashboard_active:
                    if not self._is_dashboard_down:
                        self._is_dashboard_down = True
                        logger.info("[HOTKEY] Toggling desktop dashboard window.")
                        if self.on_dashboard_toggle_callback:
                            threading.Thread(target=self.on_dashboard_toggle_callback, daemon=True).start()
                else:
                    self._is_dashboard_down = False

                # 2c. Check Toggle Live Transcription (e.g. Ctrl+Shift+L)
                if live_transcribe_active:
                    if not self._is_live_transcribe_down:
                        self._is_live_transcribe_down = True
                        logger.info("[HOTKEY] Toggling Live Transcription session.")
                        if self.on_live_transcribe_toggle_callback:
                            threading.Thread(target=self.on_live_transcribe_toggle_callback, daemon=True).start()
                else:
                    self._is_live_transcribe_down = False

                # 2d. Check per-Transform hotkeys (e.g. Alt+C / Alt+X / Alt+V) - hold to
                # dictate with that transform, same shape as PTT but per transform_id.
                any_transform_active = False
                for tid, vk_alternatives in self._transform_vks.items():
                    active = self._debounced(
                        f"transform:{tid}", any(self._all_vks_down(vks) for vks in vk_alternatives)
                    )
                    if active:
                        any_transform_active = True
                    if active and not self._is_transform_down.get(tid):
                        self._is_transform_down[tid] = True
                        self._transform_active_since[tid] = time.perf_counter()
                        logger.info(f"[HOTKEY] Transform '{tid}' pressed -> Recording started...")
                        if self.on_transform_press_callback:
                            threading.Thread(target=self.on_transform_press_callback, args=(tid,), daemon=True).start()
                    elif not active and self._is_transform_down.get(tid):
                        self._is_transform_down[tid] = False
                        logger.info(f"[HOTKEY] Transform '{tid}' released -> Processing audio pipeline...")
                        if self.on_transform_release_callback:
                            threading.Thread(target=self.on_transform_release_callback, args=(tid,), daemon=True).start()
                    elif active and time.perf_counter() - self._transform_active_since.get(tid, 0) > self.max_recording_sec:
                        logger.warning(f"[HOTKEY] Safety timeout: transform '{tid}' held over {self.max_recording_sec:.0f}s - force-stopping.")
                        self._is_transform_down[tid] = False
                        if self.on_transform_release_callback:
                            threading.Thread(target=self.on_transform_release_callback, args=(tid,), daemon=True).start()

                # 3. Tap-to-Toggle / Hold-to-Talk - two fully independent triggers.
                # Both are blocked while cycling mode or opening settings/dashboard, or
                # while a transform hotkey is down; each additionally blocks the OTHER
                # dictation trigger while it owns an in-flight recording.
                blocked_other = (
                    cycle_active or self._is_cycle_down or settings_active or dashboard_active
                    or live_transcribe_active or any_transform_active
                )
                if self.tap_toggle_enabled:
                    self._handle_tap_to_toggle(tap_raw_down, blocked_other or self._active_trigger == "hold")
                if self.hold_to_talk_enabled:
                    self._handle_hold_to_talk(hold_active, blocked_other or self._active_trigger == "tap")

            except Exception as e:
                logger.error(f"Error in hotkey polling loop: {e}")

            time.sleep(self.poll_interval)

    def start(self) -> None:
        """Start the background polling worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the background polling worker thread."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
        logger.info("Win32 hotkey manager stopped.")
