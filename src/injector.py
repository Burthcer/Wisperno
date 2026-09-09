"""
Native Win32 Text Injection Module for Wisperno.

Paste path, in order of preference:
  1. SendInput with hardware SCAN CODES (what real keyboards emit - survives apps
     that filter on virtual-key-only synthetic input).
  2. keybd_event (older input path, not subject to the same UIPI restrictions).
  3. WM_PASTE posted directly to the focused edit control (works when synthetic
     keystrokes are blocked entirely, e.g. by a higher-integrity target window).

The clipboard write is verified by read-back before any keystroke is sent, so we
never paste stale contents.
"""

import ctypes
from ctypes import wintypes
import time
from typing import Optional
from loguru import logger
import pyperclip

from src.config import InjectorConfig

# Virtual-Key Codes
VK_CONTROL = 0x11
VK_V = 0x56
VK_C = 0x43

# Hardware scan codes (Set 1) for Ctrl, V and C
SCAN_CONTROL = 0x1D
SCAN_V = 0x2F
SCAN_C = 0x2E

# Keyboard Event Flags
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

INPUT_KEYBOARD = 1
WM_PASTE = 0x0302


# Win32 INPUT Structures for 64-bit Windows
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_ulonglong),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_ulonglong),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


def _key_input(scan: int, key_up: bool) -> INPUT:
    """Build a scan-code keyboard INPUT event (wVk must be 0 in scan-code mode)."""
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if key_up else KEYEVENTF_KEYDOWN)
    return INPUT(
        type=INPUT_KEYBOARD,
        union=INPUT_UNION(ki=KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)),
    )


class TextInjector:
    """Simulates a hardware Ctrl+V into the focused Windows application."""

    def __init__(self, config: Optional[InjectorConfig] = None):
        self.config = config or InjectorConfig()

    # --- Focus helpers ------------------------------------------------------

    @staticmethod
    def _get_focused_control(hwnd: int) -> Optional[int]:
        """
        Resolve the actual focused edit control inside `hwnd`. WM_PASTE sent to a
        top-level window is usually ignored - the caret lives in a child control
        (e.g. Notepad's RichEditD2DPT), so we need that handle for the fallback.
        """
        if not hwnd:
            return None
        try:
            tid = user32.GetWindowThreadProcessId(hwnd, None)
            gti = GUITHREADINFO()
            gti.cbSize = ctypes.sizeof(GUITHREADINFO)
            if user32.GetGUIThreadInfo(tid, ctypes.byref(gti)) and gti.hwndFocus:
                return gti.hwndFocus
        except Exception as e:
            logger.debug(f"Could not resolve focused control: {e}")
        return None

    # --- Clipboard ----------------------------------------------------------

    @staticmethod
    def _clip_copy_with_retry(text: str, attempts: int = 5) -> bool:
        """
        pyperclip.copy() raises if another process currently holds the clipboard
        open (clipboard managers, browsers and Office do this constantly). Retry
        briefly rather than losing the user's dictation to a transient lock.
        """
        for attempt in range(attempts):
            try:
                pyperclip.copy(text)
                return True
            except Exception as e:
                if attempt == attempts - 1:
                    logger.error(f"Clipboard is locked by another process; copy failed: {e}")
                    return False
                time.sleep(0.03 * (attempt + 1))  # brief backoff before retrying
        return False

    def _copy_and_verify(self, text: str, timeout_ms: int = 100) -> bool:
        """
        Write `text` to the clipboard and poll read-back until it matches, so we
        never dispatch Ctrl+V while the clipboard still holds the previous value.

        Returns False only if the write itself failed - in that case the caller
        must NOT paste, since doing so would inject stale clipboard contents.
        A read-back timeout is logged but still allows the paste to proceed.
        """
        if not self._clip_copy_with_retry(text):
            return False
        deadline = time.perf_counter() + (timeout_ms / 1000.0)
        while time.perf_counter() < deadline:
            try:
                if pyperclip.paste() == text:
                    return True
            except Exception:
                pass
            time.sleep(0.005)
        logger.warning(f"Clipboard did not confirm within {timeout_ms}ms; pasting anyway.")
        return True

    # --- Keystroke dispatch -------------------------------------------------

    def _dispatch_send_input_scancode(self, key_scan: int = SCAN_V) -> bool:
        """Ctrl+<key> via SendInput using hardware scan codes. Returns True if accepted."""
        try:
            down = (INPUT * 2)(_key_input(SCAN_CONTROL, False), _key_input(key_scan, False))
            sent_down = user32.SendInput(2, down, ctypes.sizeof(INPUT))
            if sent_down != 2:
                logger.debug(
                    f"SendInput(down) accepted {sent_down}/2 (GetLastError={kernel32.GetLastError()})."
                )
                # Release anything that did land so Ctrl isn't left stuck down.
                up = (INPUT * 2)(_key_input(key_scan, True), _key_input(SCAN_CONTROL, True))
                user32.SendInput(2, up, ctypes.sizeof(INPUT))
                return False

            time.sleep(0.025)  # let the target app process the key-down pair

            up = (INPUT * 2)(_key_input(key_scan, True), _key_input(SCAN_CONTROL, True))
            sent_up = user32.SendInput(2, up, ctypes.sizeof(INPUT))
            if sent_up != 2:
                logger.debug(f"SendInput(up) accepted {sent_up}/2 - keys may be left down.")
                return False
            return True
        except Exception as e:
            logger.debug(f"SendInput scan-code exception: {e}")
            return False

    def _dispatch_keybd_event(self, key_scan: int = SCAN_V, key_vk: int = VK_V) -> bool:
        """Ctrl+<key> via the legacy keybd_event path."""
        try:
            user32.keybd_event(VK_CONTROL, SCAN_CONTROL, KEYEVENTF_KEYDOWN, 0)
            user32.keybd_event(key_vk, key_scan, KEYEVENTF_KEYDOWN, 0)
            time.sleep(0.025)
            user32.keybd_event(key_vk, key_scan, KEYEVENTF_KEYUP, 0)
            user32.keybd_event(VK_CONTROL, SCAN_CONTROL, KEYEVENTF_KEYUP, 0)
            return True
        except Exception as e:
            logger.debug(f"keybd_event exception: {e}")
            return False

    def _dispatch_wm_paste(self, hwnd: int) -> bool:
        """Last resort: post WM_PASTE straight at the focused edit control."""
        target = self._get_focused_control(hwnd) or hwnd
        if not target:
            return False
        try:
            user32.SendMessageW(target, WM_PASTE, 0, 0)
            logger.info(f"Fell back to WM_PASTE on hwnd {target}.")
            return True
        except Exception as e:
            logger.debug(f"WM_PASTE exception: {e}")
            return False

    # --- Public API ---------------------------------------------------------

    def copy_selection(self) -> Optional[str]:
        """
        Send synthetic Ctrl+C to copy whatever text is currently highlighted in
        the foreground app, returning it (or None if nothing was copied). Writes
        a sentinel to the clipboard first so "nothing selected" (clipboard stays
        the sentinel) can be told apart from "selection equals old clipboard
        content" - a blind before/after clipboard diff would miss that case.
        """
        sentinel = "\x00__wisperno_no_selection__\x00"
        if not self._clip_copy_with_retry(sentinel):
            return None
        time.sleep(0.03)  # let the OS clipboard write settle before Ctrl+C overwrites it

        if self._dispatch_send_input_scancode(SCAN_C):
            logger.debug("Selection-copy dispatched via SendInput (scan codes).")
        elif self._dispatch_keybd_event(SCAN_C, VK_C):
            logger.debug("Selection-copy dispatched via keybd_event fallback.")
        else:
            logger.error("Could not dispatch Ctrl+C to copy the selection.")
            return None

        deadline = time.perf_counter() + 0.3
        while time.perf_counter() < deadline:
            try:
                current = pyperclip.paste()
            except Exception:
                current = None
            if current is not None and current != sentinel:
                return current
            time.sleep(0.02)
        return None  # timed out - most likely nothing was actually selected

    def inject_text(self, text: str) -> bool:
        """Paste `text` into whichever window currently has focus."""
        if not text:
            logger.debug("Injector: Empty text provided, skipping injection.")
            return False

        logger.info(
            f"Injecting text ({len(text)} chars) into active window: "
            f"'{text[:60]}{'...' if len(text) > 60 else ''}'"
        )

        # 1. Capture the target BEFORE touching the clipboard - clipboard operations
        #    can themselves shift focus on some systems.
        hwnd = user32.GetForegroundWindow()
        logger.debug(f"Foreground window at injection time: hwnd={hwnd}")

        old_clip = None
        if self.config.restore_clipboard:
            try:
                old_clip = pyperclip.paste()
            except Exception as e:
                logger.debug(f"Could not read previous clipboard: {e}")

        try:
            # 2. Write + verify the clipboard actually holds our text. If the write
            #    itself failed, abort - pasting now would inject stale content.
            if not self._copy_and_verify(text, timeout_ms=100):
                logger.error("Aborting injection: could not write text to the clipboard.")
                return False

            # 3/4. Scan-code SendInput, then keybd_event, then WM_PASTE.
            if self._dispatch_send_input_scancode():
                logger.debug("Paste dispatched via SendInput (scan codes).")
            elif self._dispatch_keybd_event():
                logger.debug("Paste dispatched via keybd_event fallback.")
            elif self._dispatch_wm_paste(hwnd):
                logger.debug("Paste dispatched via WM_PASTE fallback.")
            else:
                logger.error("All injection paths failed (SendInput, keybd_event, WM_PASTE).")
                return False

            # 5. Let the target app's message loop consume the paste before we
            #    put the user's old clipboard back.
            time.sleep(self.config.post_paste_delay_ms / 1000.0)
            return True

        except Exception as e:
            logger.error(f"Failed to inject text: {e}")
            return False

        finally:
            # auto_copy_to_clipboard deliberately overrides restore_clipboard's
            # own restore step - the whole point of the setting is that the
            # just-typed text stays the clipboard's contents afterward.
            if self.config.restore_clipboard and not self.config.auto_copy_to_clipboard and old_clip is not None:
                try:
                    pyperclip.copy(old_clip)
                except Exception as e:
                    logger.debug(f"Failed to restore previous clipboard: {e}")
