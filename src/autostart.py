"""
Windows Startup Registration for Wisperno.
Adds/removes a HKCU Run key so the app can launch automatically on login.
"""

import sys
import winreg
from loguru import logger

RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_NAME = "Wisperno"


def _launch_command() -> str:
    """
    Command to register: the frozen .exe path, or `pythonw main.py` when
    running from source - always with --background, so a login-triggered
    launch starts minimized to the pill/tray instead of popping the dashboard
    in the user's face before they've touched the keyboard.
    """
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --background'
    from src.config import get_base_dir

    pythonw = str(get_base_dir() / ".venv" / "Scripts" / "pythonw.exe")
    main_py = str(get_base_dir() / "main.py")
    return f'"{pythonw}" "{main_py}" --background'


def is_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_READ) as key:
            winreg.QueryValueEx(key, APP_NAME)
            return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def enable() -> bool:
    try:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_WRITE) as key:
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, _launch_command())
        logger.info(f"Autostart enabled: {_launch_command()}")
        return True
    except OSError as e:
        logger.error(f"Failed to enable autostart: {e}")
        return False


def disable() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_WRITE) as key:
            winreg.DeleteValue(key, APP_NAME)
        logger.info("Autostart disabled.")
        return True
    except FileNotFoundError:
        return True  # Already disabled
    except OSError as e:
        logger.error(f"Failed to disable autostart: {e}")
        return False


def set_enabled(enabled: bool) -> bool:
    return enable() if enabled else disable()


def _demo() -> None:
    """Self-check: enable, verify, disable, verify - uses the real HKCU Run key."""
    was_enabled_before = is_enabled()

    assert enable() is True
    assert is_enabled() is True, "Run key was not created."

    assert disable() is True
    assert is_enabled() is False, "Run key was not removed."

    if was_enabled_before:
        enable()  # restore prior state

    print("PASS: autostart enable/disable round-trip verified against the real registry.")


if __name__ == "__main__":
    _demo()
