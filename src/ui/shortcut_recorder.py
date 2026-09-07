"""
KeySequenceRecorder: a button that captures a full hotkey chord (modifiers +
key) in one press-and-release, used by the Settings dashboard to rebind
Wisperno's global hotkeys.
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton, QWidget

# Maps a Qt key code to the token vocabulary src.hotkey_manager.VK_MAP expects.
_SPECIAL_KEY_NAMES = {
    Qt.Key.Key_Space: "space",
    Qt.Key.Key_CapsLock: "caps_lock",
    Qt.Key.Key_ScrollLock: "scroll_lock",
    Qt.Key.Key_Insert: "insert",
    Qt.Key.Key_Comma: "comma",
    Qt.Key.Key_Escape: "esc",
    Qt.Key.Key_Tab: "tab",
    **{getattr(Qt.Key, f"Key_F{i}"): f"f{i}" for i in range(1, 13)},
}

_MODIFIER_KEYS = (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta)
_MODIFIER_ORDER = (Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Shift, Qt.Key.Key_Meta)
_MOD_KEY_TOKEN = {
    Qt.Key.Key_Control: "ctrl",
    Qt.Key.Key_Alt: "alt",
    Qt.Key.Key_Shift: "shift",
    Qt.Key.Key_Meta: "win",
}

_MODIFIER_DISPLAY = [
    (Qt.KeyboardModifier.ControlModifier, "Ctrl"),
    (Qt.KeyboardModifier.AltModifier, "Alt"),
    (Qt.KeyboardModifier.ShiftModifier, "Shift"),
    (Qt.KeyboardModifier.MetaModifier, "Win"),
]


def _key_to_token(key: int) -> str:
    if key in _SPECIAL_KEY_NAMES:
        return _SPECIAL_KEY_NAMES[key]
    if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
        return chr(key).lower()
    if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
        return chr(key)
    return ""


def modifiers_to_parts(mods) -> list:
    """Order-stable Ctrl/Alt/Shift/Win prefix list for a QKeyEvent's modifiers() bitmask."""
    return [token.lower() for flag, token in _MODIFIER_DISPLAY if mods & flag]


class KeySequenceRecorder(QPushButton):
    """
    Click to arm, then press a key chord (e.g. Ctrl+Alt+V) in one go - the
    button grabs the keyboard for the duration so it reliably receives every
    key regardless of Qt's normal focus-chain routing, shows the modifiers
    live as they're held (e.g. "Ctrl + Alt + ..."), and on the first
    non-modifier key formats the full chord into the canonical string
    src.hotkey_manager expects, reporting it via `sequence_captured`.
    Escape cancels and restores the previous value.
    """

    sequence_captured = Signal(str)

    def __init__(self, initial_hotkey: str, parent: Optional[QWidget] = None):
        super().__init__(initial_hotkey.upper(), parent)
        self.hotkey_str = initial_hotkey.lower()
        self._recording = False
        self._held_mod_keys: set = set()  # modifiers currently down this recording session
        self._max_held_mods: set = set()  # union seen so far (captures the intended chord even if released staggered)
        self._non_modifier_pressed = False
        self.clicked.connect(self._start_recording)

    def _start_recording(self) -> None:
        if self._recording:
            return
        self._recording = True
        self._held_mod_keys.clear()
        self._max_held_mods.clear()
        self._non_modifier_pressed = False
        self.setText("Recording shortcut... (Press keys)")
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        self.grabKeyboard()

    def keyPressEvent(self, event) -> None:
        if not self._recording:
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key.Key_Escape:
            self._finish_recording(self.hotkey_str)  # cancel: restore previous value
            return

        if event.key() in _MODIFIER_KEYS:
            self._held_mod_keys.add(event.key())
            self._max_held_mods.add(event.key())
            # Not a complete chord yet - show progress so the user sees the
            # modifiers registering instead of a static "Recording..." label.
            parts = modifiers_to_parts(event.modifiers()) or ["Recording shortcut..."]
            self.setText(" + ".join(p.capitalize() for p in parts) + " + ...")
            return

        self._non_modifier_pressed = True
        parts = modifiers_to_parts(event.modifiers())
        key_token = _key_to_token(event.key())
        if not key_token:
            return  # unrecognized key - keep waiting rather than commit garbage
        parts.append(key_token)

        self._finish_recording("+".join(parts))

    def keyReleaseEvent(self, event) -> None:
        if not self._recording or event.key() not in _MODIFIER_KEYS:
            super().keyReleaseEvent(event)
            return

        self._held_mod_keys.discard(event.key())
        # A pure-modifier chord (e.g. Ctrl+Alt, no terminating key) finalizes
        # once every modifier that was held together is released, as long as
        # no real key was pressed in between and at least two modifiers were
        # actually held at once - a single bare Ctrl tap is not a shortcut.
        if not self._held_mod_keys and not self._non_modifier_pressed and len(self._max_held_mods) >= 2:
            parts = [_MOD_KEY_TOKEN[k] for k in _MODIFIER_ORDER if k in self._max_held_mods]
            self._finish_recording("+".join(parts))

    def _finish_recording(self, hotkey_str: str) -> None:
        self._recording = False
        self._held_mod_keys.clear()
        self._max_held_mods.clear()
        self.releaseKeyboard()
        self.hotkey_str = hotkey_str
        self.setText(hotkey_str.upper())
        self.sequence_captured.emit(hotkey_str)
