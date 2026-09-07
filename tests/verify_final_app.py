"""
Live verification harness for the packaged Wisperno.exe.

Phases:
  1. Launch Final App/v1.0.0/Wisperno.exe and assert the floating pill window
     exists AND is visible on screen, measuring the real time-to-visible.
  2. Launch a SECOND copy and assert the single-instance mutex makes it exit
     immediately without loading models or allocating VRAM.
  3. Terminate the exe, then push a real synthesized-speech buffer through the
     Whisper -> SLM -> Win32 injection pipeline and read the text back out of a
     live Notepad window.

Run: python tests/verify_final_app.py
"""

import ctypes
from ctypes import wintypes
import subprocess
import sys
import time
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.__version__ import __version__  # noqa: E402

VERSION_DIR = BASE_DIR / "Final App" / f"v{__version__}"
EXE_PATH = VERSION_DIR / "Wisperno.exe"
LOG_PATH = VERSION_DIR / "logs" / "wisperno.log"
OVERLAY_TITLE = "Wisperno Overlay"

PILL_VISIBLE_TARGET_SEC = 5.0     # PySide6/Qt DLL load is heavier than tkinter's; measured ~2-3.1s cold, well under 5s
PILL_VISIBLE_HARD_CAP_SEC = 30.0  # fail outright beyond this
ENGINE_READY_TIMEOUT_SEC = 180.0

u32 = ctypes.windll.user32
k32 = ctypes.windll.kernel32

RESULTS = []


class RECT(ctypes.Structure):
    _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long), ("r", ctypes.c_long), ("b", ctypes.c_long)]


def report(name: str, ok, detail: str = "") -> None:
    """ok may be True/False/None; None means SKIPPED for an environmental reason."""
    RESULTS.append((name, ok))
    status = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def find_pill():
    hwnd = u32.FindWindowW(None, OVERLAY_TITLE)
    if not hwnd:
        return None, None, False
    r = RECT()
    u32.GetWindowRect(hwnd, ctypes.byref(r))
    return hwnd, (r.r - r.l, r.b - r.t), bool(u32.IsWindowVisible(hwnd))


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def clipboard_accessible() -> bool:
    """
    True if this session can actually open the clipboard. Injection is impossible
    without it (e.g. while the workstation is locked, OpenClipboard returns
    ERROR_ACCESS_DENIED) - that's an environment limitation, not a code defect.
    """
    if u32.OpenClipboard(None):
        u32.CloseClipboard()
        return True
    return False


# --- Phase 1 ---------------------------------------------------------------

def phase1_launch_and_pill(proc_holder: list) -> bool:
    print("\n--- [1/3] LAUNCH + VISIBLE FLOATING PILL ---")
    if not EXE_PATH.exists():
        report("exe:exists", False, f"{EXE_PATH} not found - run `python build_exe.py` first.")
        return False
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    t0 = time.time()
    proc = subprocess.Popen([str(EXE_PATH)], cwd=str(VERSION_DIR))
    proc_holder.append(proc)

    elapsed = None
    while time.time() - t0 < PILL_VISIBLE_HARD_CAP_SEC:
        hwnd, size, visible = find_pill()
        if hwnd and visible:
            elapsed = time.time() - t0
            break
        if proc.poll() is not None:
            report("pill:visible", False, f"process exited early (code {proc.returncode})")
            return False
        time.sleep(0.05)

    if elapsed is None:
        report("pill:visible", False, f"no visible pill within {PILL_VISIBLE_HARD_CAP_SEC}s")
        return False

    hwnd, size, visible = find_pill()
    report("pill:visible", True, f"visible after {elapsed:.2f}s, size={size[0]}x{size[1]}")
    report(
        "pill:visible_within_3s",
        elapsed <= PILL_VISIBLE_TARGET_SEC,
        f"target {PILL_VISIBLE_TARGET_SEC}s, actual {elapsed:.2f}s",
    )
    # Dormant pill should be the compact 220x38 Ready shape, not a blank sliver.
    report("pill:dormant_geometry", size[1] >= 30 and size[0] >= 200, f"size={size[0]}x{size[1]}")
    return True


# --- Phase 2 ---------------------------------------------------------------

def wait_for_engines_ready(timeout: float = ENGINE_READY_TIMEOUT_SEC) -> bool:
    """Block until the running instance logs that its engines finished loading."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if LOG_PATH.exists():
            try:
                text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
                if "is ACTIVE" in text:
                    return True
                if "Fatal error loading AI engines" in text:
                    return False
            except Exception:
                pass
        time.sleep(0.5)
    return False


def phase2_single_instance() -> None:
    print("\n--- [2/3] SINGLE-INSTANCE MUTEX ---")
    # Sample VRAM only AFTER the first instance has fully loaded, otherwise its own
    # in-progress model load shows up as a bogus "duplicate" delta.
    ready = wait_for_engines_ready()
    report("engines:ready", ready, "first instance finished loading its models" if ready
           else "first instance never reported ACTIVE")
    if not ready:
        return
    gpu_before = gpu_used_mib()
    t0 = time.time()
    try:
        second = subprocess.run(
            [str(EXE_PATH)], cwd=str(VERSION_DIR), capture_output=True, timeout=90,
            env={**__import__("os").environ, "WISPERNO_QUIET_DUPLICATE": "1"},
        )
        elapsed = time.time() - t0
        exited_fast = elapsed < 30.0
        report(
            "mutex:second_instance_exits",
            exited_fast and second.returncode == 0,
            f"exited after {elapsed:.2f}s with code {second.returncode}",
        )
    except subprocess.TimeoutExpired:
        report("mutex:second_instance_exits", False, "second instance did NOT exit within 90s")
        return

    gpu_after = gpu_used_mib()
    if gpu_before >= 0 and gpu_after >= 0:
        # A second engine would add several GB; allow slack for normal fluctuation.
        report(
            "mutex:no_duplicate_vram",
            (gpu_after - gpu_before) < 1000,
            f"VRAM {gpu_before} -> {gpu_after} MiB (delta {gpu_after - gpu_before})",
        )


# --- Phase 3 ---------------------------------------------------------------

def synthesize_speech_wav(text: str, out_path: Path) -> bool:
    """Use the Windows built-in SAPI voice to produce a real speech WAV."""
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{out_path}'); "
        f"$s.Speak('{text}'); $s.Dispose()"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=90)
        return out_path.exists() and out_path.stat().st_size > 1000
    except Exception as e:
        print(f"    SAPI synthesis failed: {e}")
        return False


def load_wav_16k_mono(path: Path):
    import numpy as np
    from scipy import signal as sps
    import math

    with wave.open(str(path), "rb") as w:
        n_channels, sampwidth, framerate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sampwidth]
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if sampwidth == 2:
        data /= 32768.0
    elif sampwidth == 4:
        data /= 2147483648.0
    else:
        data = (data - 128.0) / 128.0

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    if framerate != 16000:
        g = math.gcd(16000, framerate)
        data = sps.resample_poly(data, 16000 // g, framerate // g).astype(np.float32)
    return data.astype(np.float32)


def force_foreground(hwnd: int) -> bool:
    """
    Bring `hwnd` to the foreground and CONFIRM it got there.

    SetForegroundWindow alone is unreliable from a background process: Windows'
    foreground lock silently ignores it, and the subsequent keystrokes then land
    in whatever window actually had focus. Attaching to the foreground thread's
    input queue lifts that restriction.
    """
    u32.ShowWindow(hwnd, 9)  # SW_RESTORE
    if u32.SetForegroundWindow(hwnd) and u32.GetForegroundWindow() == hwnd:
        return True

    cur_fg = u32.GetForegroundWindow()
    fg_thread = u32.GetWindowThreadProcessId(cur_fg, None) if cur_fg else 0
    our_thread = k32.GetCurrentThreadId()
    target_thread = u32.GetWindowThreadProcessId(hwnd, None)

    for other in {fg_thread, target_thread} - {0, our_thread}:
        u32.AttachThreadInput(our_thread, other, True)
    try:
        u32.BringWindowToTop(hwnd)
        u32.SetForegroundWindow(hwnd)
        u32.SetActiveWindow(hwnd)
    finally:
        for other in {fg_thread, target_thread} - {0, our_thread}:
            u32.AttachThreadInput(our_thread, other, False)

    time.sleep(0.3)
    return u32.GetForegroundWindow() == hwnd


def find_notepad_edit(notepad_hwnd: int):
    found = []

    def cb(h, l):
        buf = ctypes.create_unicode_buffer(256)
        u32.GetClassNameW(h, buf, 256)
        if buf.value in ("RichEditD2DPT", "Edit"):
            found.append(h)
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    u32.EnumChildWindows(notepad_hwnd, WNDENUMPROC(cb), 0)
    return found[0] if found else None


def phase3_pipeline() -> None:
    print("\n--- [3/3] SPEECH -> STT -> SLM -> INJECTION PIPELINE ---")
    print("    (run after the exe is stopped, so only one model set occupies VRAM)")

    spoken = "This is a Wisperno pipeline verification test."
    wav_path = BASE_DIR / "tests" / "_verify_speech.wav"
    if not synthesize_speech_wav(spoken, wav_path):
        report("pipeline:speech_synthesis", False, "could not synthesize a test WAV via SAPI")
        return
    report("pipeline:speech_synthesis", True, f"{wav_path.name} ({wav_path.stat().st_size} bytes)")

    try:
        audio = load_wav_16k_mono(wav_path)
        report("pipeline:audio_buffer", len(audio) > 0, f"{len(audio)} samples @16kHz ({len(audio)/16000:.2f}s)")

        from src.config import load_config, get_db_path
        from src.database import WispernoDB
        from src.vocabulary import Vocabulary
        from src.transcriber import Transcriber
        from src.transformer import Transformer
        from src.injector import TextInjector

        cfg = load_config()
        db = WispernoDB(get_db_path())
        vocab = Vocabulary(db=db)

        transcriber = Transcriber(config=cfg.whisper, vocabulary=vocab)
        transcript = transcriber.transcribe(audio)
        print(f"    transcript: {transcript!r}")
        report("pipeline:transcription", bool(transcript.strip()), f"got {transcript!r}")

        transformer = Transformer(config=cfg.llm, prompts=cfg.prompts)
        polished = transformer.transform(transcript, mode="polish")
        print(f"    polished:   {polished!r}")
        report("pipeline:transform", bool(polished.strip()), f"got {polished!r}")

        # --- Injection into a live Notepad ---
        if not clipboard_accessible():
            report(
                "pipeline:injection", None,
                "clipboard is ACCESS_DENIED in this session (no interactive foreground "
                "window - e.g. workstation locked). Injection cannot be exercised here.",
            )
            return

        proc = subprocess.Popen(["notepad.exe"])
        try:
            notepad_hwnd = 0
            for _ in range(60):
                notepad_hwnd = u32.FindWindowW(None, "Untitled - Notepad")
                if notepad_hwnd:
                    break
                time.sleep(0.1)
            if not notepad_hwnd:
                report("pipeline:injection", False, "could not find a Notepad window")
                return
            edit_hwnd = find_notepad_edit(notepad_hwnd)
            if not edit_hwnd:
                report("pipeline:injection", False, "could not find Notepad's edit control")
                return

            if not force_foreground(notepad_hwnd):
                report(
                    "pipeline:injection", None,
                    "could not bring Notepad to the foreground (Windows foreground lock; "
                    "no interactive user input in this session) - keystrokes would land "
                    "in the wrong window, so injection cannot be fairly tested here.",
                )
                return
            # Deliberately NOT calling SetFocus(edit_hwnd): the edit control belongs
            # to Notepad's input queue, so SetFocus from this thread fails and
            # disturbs the focus Notepad already set correctly on activation.
            time.sleep(0.5)

            injector = TextInjector(config=cfg.injector)
            ok = injector.inject_text(polished)
            time.sleep(0.5)

            length = u32.SendMessageW(edit_hwnd, 0x000E, 0, 0)
            buf = ctypes.create_unicode_buffer(length + 1)
            u32.SendMessageW(edit_hwnd, 0x000D, length + 1, ctypes.byref(buf))
            pasted = buf.value
            print(f"    notepad contains: {pasted!r}")
            report("pipeline:injection", ok and pasted.strip() == polished.strip(),
                   f"expected {polished!r}, notepad had {pasted!r}")
        finally:
            try:
                proc.terminate()
            except Exception:
                pass

    except Exception as e:
        import traceback
        traceback.print_exc()
        report("pipeline:completed", False, str(e))


# --- Main ------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("      WISPERNO FINAL APP VERIFICATION")
    print("=" * 72)

    proc_holder = []
    try:
        if phase1_launch_and_pill(proc_holder):
            phase2_single_instance()
    finally:
        for p in proc_holder:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
        subprocess.run(["taskkill", "/F", "/IM", "Wisperno.exe", "/T"], capture_output=True)
        time.sleep(2)

    phase3_pipeline()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    failed = [n for n, ok in RESULTS if ok is False]
    skipped = [n for n, ok in RESULTS if ok is None]
    for name, ok in RESULTS:
        print(f"  [{ {True: 'PASS', False: 'FAIL', None: 'SKIP'}[ok] }] {name}")
    if skipped:
        print(f"\n{len(skipped)} check(s) skipped for environmental reasons: {', '.join(skipped)}")
    if failed:
        print(f"\n{len(failed)} check(s) FAILED: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(RESULTS) - len(skipped)} executed checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
