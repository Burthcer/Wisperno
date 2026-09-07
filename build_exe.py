"""
Versioned build script for producing a standalone Wisperno.exe via PyInstaller.

Usage:
    python build_exe.py

Runs `pyinstaller wisperno.spec`, then assembles the result into:

    Final App/v{VERSION}/
        Wisperno.exe
        models/            (junction to the source tree's models/ - avoids
                             duplicating multi-GB model files on every build)
        config/
            config.yaml
            dictionary.json
        assets/icons/
        README.txt

VERSION is read from src/__version__.py.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYINSTALLER_DIST_DIR = BASE_DIR / "dist" / "Wisperno"

sys.path.insert(0, str(BASE_DIR))
from src.__version__ import __version__  # noqa: E402


def run_pyinstaller() -> None:
    print("=" * 70)
    print(" Building Wisperno.exe with PyInstaller (this can take several minutes)")
    print("=" * 70)
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "wisperno.spec", "--noconfirm", "--clean"],
        cwd=str(BASE_DIR),
    )
    if result.returncode != 0:
        print("\nPyInstaller build FAILED. See output above.")
        sys.exit(result.returncode)
    if not PYINSTALLER_DIST_DIR.exists():
        print(f"\nBuild reported success but '{PYINSTALLER_DIST_DIR}' was not found.")
        sys.exit(1)


def link_or_copy_models(target_dir: Path) -> None:
    """Prefer a directory junction to the source models/ folder (multi-GB, no need to duplicate)."""
    source_models = BASE_DIR / "models"
    target_models = target_dir / "models"
    if target_models.exists():
        return
    if not source_models.exists():
        target_models.mkdir(parents=True, exist_ok=True)
        return

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(target_models), str(source_models)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Linked models/ -> {source_models} (directory junction).")
    else:
        print(f"Could not create a junction ({result.stderr.strip()}); copying models/ instead (this is slow).")
        shutil.copytree(source_models, target_models)


def stop_running_instances() -> None:
    """
    Terminate any running Wisperno.exe before touching the output folder.

    A running instance holds logs/wisperno.log open, which makes rmtree() fail with
    PermissionError (WinError 32) and leaves a half-wiped distribution behind.
    """
    result = subprocess.run(
        ["taskkill", "/F", "/IM", "Wisperno.exe", "/T"], capture_output=True, text=True
    )
    if "SUCCESS" in result.stdout:
        print("Stopped a running Wisperno.exe before rebuilding.")
        time.sleep(2)  # let Windows release the file handles


def assemble_final_app() -> Path:
    version_dir = BASE_DIR / "Final App" / f"v{__version__}"
    stop_running_instances()
    if version_dir.exists():
        try:
            shutil.rmtree(version_dir)
        except PermissionError as e:
            print(f"\nCould not clear '{version_dir}': {e}")
            print("Something still holds a file open there - close Wisperno and retry.")
            sys.exit(1)
    version_dir.mkdir(parents=True)

    for item in PYINSTALLER_DIST_DIR.iterdir():
        dest = version_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    config_dir = version_dir / "config"
    config_dir.mkdir(exist_ok=True)
    shutil.copy2(BASE_DIR / "config" / "config.yaml", config_dir / "config.yaml")
    shutil.copy2(BASE_DIR / "config" / "dictionary.json", config_dir / "dictionary.json")

    icons_dir = version_dir / "assets" / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)
    src_icons = BASE_DIR / "assets" / "icons"
    if src_icons.exists():
        for icon in src_icons.iterdir():
            if icon.is_file():
                shutil.copy2(icon, icons_dir / icon.name)

    (version_dir / "logs").mkdir(exist_ok=True)

    init_database(version_dir)  # seeds %APPDATA%\Wisperno\wisperno.db, not version_dir - see docstring
    link_or_copy_models(version_dir)

    (version_dir / "README.txt").write_text(
        f"Wisperno v{__version__}\n"
        "========================\n\n"
        "Run Wisperno.exe. It runs silently in the background with a tray icon\n"
        "(bottom-right of your taskbar) and a floating status pill.\n\n"
        f"Default hotkeys:\n"
        f"  Ctrl+Alt        Hold to talk (or tap, if Tap-to-Toggle is enabled)\n"
        f"  Ctrl+Shift+B    Cycle transformation mode\n"
        f"  Ctrl+,          Open the Dashboard (History / Dictionary / Snippets / Styles / Settings)\n\n"
        "Model weights live in models/ next to this exe - if that folder is empty,\n"
        "run `python download_models.py` from the source checkout and copy the\n"
        "resulting models/ folder here.\n\n"
        "Right-click the tray icon for mode/trigger selection, the dashboard, and exit.\n",
        encoding="utf-8",
    )

    return version_dir


def init_database(version_dir: Path) -> None:
    """
    Create wisperno.db with its schema (and migrate config/dictionary.json into
    it) so first launch doesn't pay that cost. Seeded at %APPDATA%\\Wisperno\\
    wisperno.db - the app's one real runtime location (src/config.py:get_db_path)
    - NOT version_dir, which would be a stray, never-read copy now that history
    survives independently of which build folder the user is running.
    """
    sys.path.insert(0, str(BASE_DIR))
    from src.database import WispernoDB
    from src.config import get_db_path

    db_path = get_db_path()
    db = WispernoDB(db_path)
    db.migrate_dictionary_json(version_dir / "config" / "dictionary.json")
    db.close()
    print(f"Initialized database at {db_path}.")


def main() -> None:
    run_pyinstaller()
    version_dir = assemble_final_app()
    print("\n" + "=" * 70)
    print(f" Build complete: {version_dir}\\Wisperno.exe")
    print("=" * 70)


if __name__ == "__main__":
    main()
