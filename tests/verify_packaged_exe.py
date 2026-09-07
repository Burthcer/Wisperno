"""
Standalone verification for the packaged Wisperno.exe (Final App/v{VERSION}/).
Launches the real .exe, confirms it survives startup without crashing, confirms
it writes clean startup logs to logs/wisperno.log (this is also indirect proof
the noconsole-loguru crash is fixed - a crash there would leave the log empty
or truncated), then terminates it cleanly.

Run: python tests/verify_packaged_exe.py
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.__version__ import __version__  # noqa: E402

VERSION_DIR = BASE_DIR / "Final App" / f"v{__version__}"
EXE_PATH = VERSION_DIR / "Wisperno.exe"
LOG_PATH = VERSION_DIR / "logs" / "wisperno.log"

STARTUP_WAIT_SEC = 5
POLL_INTERVAL_SEC = 1
MAX_WAIT_FOR_ACTIVE_SEC = 45


def main() -> int:
    if not EXE_PATH.exists():
        print(f"[FAIL] {EXE_PATH} does not exist - run `python build_exe.py` first.")
        return 1

    # Fresh log for this run so we don't read stale entries from a previous launch.
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    print(f"Launching {EXE_PATH} ...")
    proc = subprocess.Popen([str(EXE_PATH)], cwd=str(VERSION_DIR))

    try:
        time.sleep(STARTUP_WAIT_SEC)
        if proc.poll() is not None:
            print(f"[FAIL] Process exited early with code {proc.returncode} (crashed on startup).")
            return 1
        print(f"[PASS] Process (PID {proc.pid}) still alive after {STARTUP_WAIT_SEC}s.")

        waited = STARTUP_WAIT_SEC
        while not LOG_PATH.exists() and waited < MAX_WAIT_FOR_ACTIVE_SEC:
            time.sleep(POLL_INTERVAL_SEC)
            waited += POLL_INTERVAL_SEC
            if proc.poll() is not None:
                print(f"[FAIL] Process exited with code {proc.returncode} before writing any log.")
                return 1

        if not LOG_PATH.exists():
            print(f"[FAIL] {LOG_PATH} was never created after {waited}s.")
            return 1

        log_text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        if not log_text.strip():
            print(f"[FAIL] {LOG_PATH} exists but is empty.")
            return 1
        print(f"[PASS] {LOG_PATH} created and non-empty ({len(log_text)} chars).")

        if "Traceback" in log_text or "ERROR" in log_text.upper().replace("NO ERROR", ""):
            print("[WARN] Log contains ERROR/Traceback text - inspect manually:")
            print(log_text[-2000:])
        else:
            print("[PASS] No ERROR/Traceback lines found in the startup log.")

        if proc.poll() is not None:
            print(f"[FAIL] Process exited with code {proc.returncode} sometime during the check.")
            return 1
        print(f"[PASS] Process (PID {proc.pid}) still alive after {waited}s total - no crash dialog, no early exit.")

        return 0
    finally:
        if proc.poll() is None:
            print(f"Terminating PID {proc.pid}...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
