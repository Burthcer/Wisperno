"""
Single-Instance Guard for Wisperno.

Uses a Win32 named mutex so a second launch (e.g. an impatient double-click while
the first copy is still loading models) can never start a second engine. Multiple
copies would otherwise collide over the microphone stream, VRAM, and the global
GetAsyncKeyState hotkey poller.
"""

import ctypes
import os
import sys
from typing import Optional

from loguru import logger

kernel32 = ctypes.windll.kernel32
user32 = ctypes.windll.user32

MUTEX_NAME = r"Local\Wisperno_SingleInstance_Mutex_9921"
ERROR_ALREADY_EXISTS = 183

OVERLAY_WINDOW_TITLE = "Wisperno Overlay"

# The handle must outlive acquire() - if it is garbage collected the OS releases
# the mutex and the guard silently stops working. Held at module scope for the
# lifetime of the process.
_mutex_handle: Optional[int] = None


def _focus_existing_instance() -> bool:
    """Bring the already-running instance's pill to the foreground, if we can find it."""
    hwnd = user32.FindWindowW(None, OVERLAY_WINDOW_TITLE)
    if not hwnd:
        return False
    try:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        # Flash it so the user's eye is drawn to the pill they already have running.
        user32.FlashWindow(hwnd, True)
        return True
    except Exception:
        return False


def _notify_already_running() -> None:
    """Tell the user why this launch did nothing (the whole point of the guard)."""
    try:
        # MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST
        user32.MessageBoxW(
            None,
            "Wisperno is already running.\n\n"
            "Look for the pill at the bottom of your screen, or the Wisperno icon "
            "in your system tray (bottom-right, next to the clock).",
            "Wisperno",
            0x00000000 | 0x00000040 | 0x00010000 | 0x00040000,
        )
    except Exception:
        pass


def acquire(notify: bool = True) -> None:
    """
    Claim the single-instance mutex. If another instance already holds it, draw
    attention to that instance and exit this process immediately.

    Must be called before any model, GUI, or audio-device initialization.

    Feedback is deliberately non-blocking when the running instance's pill can be
    found on screen: flashing the pill the user is already looking at is enough,
    and a modal dialog would leave this duplicate process alive until dismissed.
    The modal is reserved for the case where no pill can be found - only then is
    the user genuinely left with no explanation.
    """
    global _mutex_handle

    _mutex_handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    last_error = kernel32.GetLastError()

    if _mutex_handle and last_error == ERROR_ALREADY_EXISTS:
        logger.warning("Another Wisperno instance is already running - exiting this one.")
        focused = _focus_existing_instance()
        if notify and not focused and not os.environ.get("WISPERNO_QUIET_DUPLICATE"):
            _notify_already_running()
        logger.info(f"Existing instance {'flashed on screen' if focused else 'not found on screen'}; exiting.")
        sys.exit(0)

    if not _mutex_handle:
        # Could not create the mutex at all - log it but do not block startup, since
        # refusing to run over a mutex failure is worse than running unguarded.
        logger.warning(f"Could not create single-instance mutex (GetLastError={last_error}); continuing unguarded.")
        return

    logger.info("Single-instance mutex acquired.")


def is_held() -> bool:
    """True if this process currently owns the single-instance mutex."""
    return _mutex_handle is not None


def release() -> None:
    """Release the mutex (Windows also does this automatically on process exit)."""
    global _mutex_handle
    if _mutex_handle:
        try:
            kernel32.ReleaseMutex(_mutex_handle)
            kernel32.CloseHandle(_mutex_handle)
        except Exception:
            pass
        _mutex_handle = None


def kill_orphaned_instances(exclude_self: bool = True) -> int:
    """
    Recovery helper: terminate other running Wisperno.exe processes.

    NOT called during normal startup - the mutex is the startup guard. If a fresh
    launch killed whatever was already running, an impatient double-click would
    tear down a healthy instance and restart the whole multi-second model load,
    which is the exact failure this module exists to prevent.

    Use it to clean up genuinely orphaned/zombie processes:
        python -m src.single_instance --cleanup

    Returns the number of processes terminated.
    """
    import subprocess

    own_pid = kernel32.GetCurrentProcessId() if exclude_self else -1
    killed = 0

    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Wisperno.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in result.stdout.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            if pid == own_pid:
                continue
            subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], capture_output=True, timeout=15)
            logger.info(f"Terminated orphaned Wisperno process PID {pid}.")
            killed += 1
    except Exception as e:
        logger.error(f"Orphan cleanup failed: {e}")

    return killed


if __name__ == "__main__":
    if "--cleanup" in sys.argv:
        count = kill_orphaned_instances()
        print(f"Terminated {count} orphaned Wisperno process(es).")
    else:
        # Self-check: acquiring twice in one process is fine (same owner); the real
        # cross-process behavior is exercised by tests/verify_final_app.py.
        handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
        err = kernel32.GetLastError()
        assert handle, "CreateMutexW returned NULL"
        print(f"PASS: mutex created (handle={handle}, GetLastError={err}, already_exists={err == ERROR_ALREADY_EXISTS}).")
        kernel32.CloseHandle(handle)
