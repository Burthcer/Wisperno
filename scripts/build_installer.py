"""
Compiles installer/setup.iss into Final App/Wisperno-Setup-v{VERSION}.exe via
Inno Setup's command-line compiler (ISCC.exe).

Requires `python build_exe.py` to have already produced Final App/v{VERSION}/
- this script does not invoke PyInstaller itself, it packages that output.

Final App/v{VERSION}/models/ is a junction (see build_exe.py) to this dev
machine's whole models/ cache - every LLM preset ever downloaded, not just
the one actually configured. Installing that verbatim would ship several
presets nobody selected. So this script first stages a trimmed copy (app +
runtime + the Whisper weights + ONLY the active LLM preset) via
package_portable.py's own, already-tested _copy_exe_and_runtime()/
_copy_models()/_write_clean_config() helpers, and points Inno at that
instead of the raw build output.

Usage: python scripts/build_installer.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.__version__ import __version__  # noqa: E402
from package_portable import _copy_exe_and_runtime, _copy_models, _write_clean_config  # noqa: E402

STAGING_DIR = BASE_DIR / "installer" / "_staging"

WINGET_ID = "JRSoftware.InnoSetup"
# Common install locations, in the order worth checking - PATH first (covers
# a manual install the user already put there), then both Program Files
# variants Inno's own installer defaults to for a machine-wide install, then
# the per-user location winget actually used on this machine.
ISCC_CANDIDATES = [
    r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    r"C:\Program Files\Inno Setup 6\ISCC.exe",
    str(Path.home() / "AppData" / "Local" / "Programs" / "Inno Setup 6" / "ISCC.exe"),
]


def find_iscc() -> str:
    on_path = shutil.which("ISCC.exe") or shutil.which("ISCC")
    if on_path:
        return on_path
    for candidate in ISCC_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return ""


def install_inno_setup() -> str:
    print(f"Inno Setup not found - installing via winget ({WINGET_ID})...")
    result = subprocess.run(
        ["winget", "install", "--id", WINGET_ID, "-e",
         "--accept-source-agreements", "--accept-package-agreements"],
    )
    if result.returncode != 0:
        print("\n[ERROR] winget install failed. Install Inno Setup 6 manually from "
              "https://jrsoftware.org/isdl.php and re-run this script.")
        sys.exit(1)
    iscc = find_iscc()
    if not iscc:
        print("\n[ERROR] winget reported success but ISCC.exe still can't be found.")
        sys.exit(1)
    return iscc


def stage_installer_source() -> Path:
    version_dir = BASE_DIR / "Final App" / f"v{__version__}"
    if not version_dir.exists():
        print(f"[ERROR] '{version_dir}' not found - run `python build_exe.py` first.")
        sys.exit(1)

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    print("Staging installer source (app + runtime + active model preset only)...")
    _copy_exe_and_runtime(STAGING_DIR)
    _copy_models(STAGING_DIR)
    _write_clean_config(STAGING_DIR)
    return STAGING_DIR


def main() -> None:
    staging_dir = stage_installer_source()

    iscc = find_iscc() or install_inno_setup()
    print(f"Using Inno Setup compiler: {iscc}")

    print("=" * 70)
    print(f" Compiling Wisperno-Setup-v{__version__}.exe")
    print("=" * 70)
    result = subprocess.run(
        [iscc, f"/DMyAppVersion={__version__}", f"/DSourceDir={staging_dir}", str(BASE_DIR / "installer" / "setup.iss")],
        cwd=str(BASE_DIR / "installer"),
    )
    if result.returncode != 0:
        print("\nInno Setup compilation FAILED. See output above.")
        sys.exit(result.returncode)

    setup_exe = BASE_DIR / "Final App" / f"Wisperno-Setup-v{__version__}.exe"
    if not setup_exe.exists():
        print(f"\n[ERROR] Compiler reported success but '{setup_exe}' was not found.")
        sys.exit(1)

    # Disk spanning (see setup.iss) puts the actual payload in sibling
    # Wisperno-Setup-v{VERSION}-N.bin volume(s), not the launcher exe itself -
    # both must ship together, so total size is measured across all of them.
    volumes = [setup_exe] + sorted(setup_exe.parent.glob(f"Wisperno-Setup-v{__version__}-*.bin"))
    total_gb = sum(f.stat().st_size for f in volumes) / 1e9

    print("Cleaning up the staging directory...")
    shutil.rmtree(staging_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f" Installer built: {setup_exe.name} + {len(volumes) - 1} volume(s) ({total_gb:.2f} GB total)")
    for f in volumes:
        print(f"   {f}")
    print(" All files above must ship together in the same folder.")
    print("=" * 70)


if __name__ == "__main__":
    main()
