"""
Assembles a fully self-contained, offline-ready Wisperno bundle for
deployment on a machine that doesn't have this source checkout: the compiled
exe + its PyInstaller runtime deps, Whisper's own weights, ONLY the
currently-active LLM preset's weights (not every tier - a full copy of all
four would roughly double the archive for presets nobody's actually running),
a clean relative-path config.yaml, and a fresh, empty database (no personal
history/dictionary/snippet customization baked in from this dev machine).

Requires `python build_exe.py` to have already produced Final App/v{VERSION}/
- this script does not invoke PyInstaller itself, it repackages that output.

Usage: python scripts/package_portable.py [--no-zip]
    --no-zip    Assemble Final App/Wisperno-Portable/ only, skip the .zip
                (the folder alone is directly runnable - useful while
                iterating, since the zip step is the slow part on multi-GB
                model weights).
"""

import shutil
import sys
import time
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.__version__ import __version__  # noqa: E402
from src.config import load_config  # noqa: E402

FINAL_APP_DIR = BASE_DIR / "Final App"
BUILT_VERSION_DIR = FINAL_APP_DIR / f"v{__version__}"
BUILT_CONFIG_PATH = BUILT_VERSION_DIR / "config" / "config.yaml"
PORTABLE_DIR = FINAL_APP_DIR / "Wisperno-Portable"
PORTABLE_ZIP = FINAL_APP_DIR / f"Wisperno-v{__version__}-Portable.zip"

# faster-whisper's own HuggingFace-cache-style layout under models/ (see
# src/transcriber.py's download_root) - copied whole; shutil.copytree's
# default symlinks=False resolves any internal cache symlinks to real file
# content, so the bundle never depends on symlinks surviving a move to
# another machine/filesystem.
WHISPER_CACHE_DIRNAME = "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"


def _copy_exe_and_runtime(dest: Path) -> None:
    if not BUILT_VERSION_DIR.exists():
        print(f"[ERROR] '{BUILT_VERSION_DIR}' not found - run `python build_exe.py` first.")
        sys.exit(1)
    for item in BUILT_VERSION_DIR.iterdir():
        # models/ is a junction in the build output (see build_exe.py) - handled
        # separately below with a REAL copy, since a junction only resolves on
        # this machine. logs/ starts empty either way, no need to carry it over.
        if item.name in ("models", "logs"):
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _copy_models(dest: Path) -> None:
    src_models = BASE_DIR / "models"
    dest_models = dest / "models"
    dest_models.mkdir(parents=True, exist_ok=True)

    whisper_cache = src_models / WHISPER_CACHE_DIRNAME
    if whisper_cache.exists():
        shutil.copytree(whisper_cache, dest_models / WHISPER_CACHE_DIRNAME, dirs_exist_ok=True)
        print(f"  Bundled Whisper weights ({whisper_cache.name}).")
    else:
        print(
            f"  [WARNING] Whisper weights not found at '{whisper_cache}' - "
            "the portable build will need to download them on first run."
        )

    # Root-cause fix: "active preset" MUST mean the preset actually selected
    # in the BUILT app (Final App/v{VERSION}/config/config.yaml - what the
    # Settings UI reads/writes and what a real install carries forward), not
    # this source checkout's own config/config.yaml. Those two drifted apart
    # in practice (dev switches presets live in the running app; the source
    # tree's config.yaml sits untouched) and staging from the wrong one
    # shipped an installer whose config.yaml pointed at a model file that
    # was never actually copied in - Whisper loaded fine, the LLM silently
    # had nothing to load, and the app quietly fell back to zero-LLM raw
    # transcript (measured: ~1.1-1.8GB VRAM instead of the preset's real
    # ~3.2-3.5GB, with the "not found" reason logged but easy to miss).
    config = load_config(BUILT_CONFIG_PATH)
    active_llm_filename = Path(config.llm.model_path).name
    active_llm_src = src_models / active_llm_filename
    if not active_llm_src.exists():
        print(
            f"  [ERROR] Active LLM model '{active_llm_filename}' (preset '{config.model_preset}', "
            f"from '{BUILT_CONFIG_PATH}') not found at '{active_llm_src}'. Refusing to ship a build "
            "whose configured model can't be loaded - download the model or switch presets first."
        )
        sys.exit(1)
    shutil.copy2(active_llm_src, dest_models / active_llm_filename)
    size_gb = active_llm_src.stat().st_size / 1e9
    print(f"  Bundled active LLM preset ('{config.model_preset}'): {active_llm_filename} ({size_gb:.1f} GB).")


def _write_clean_config(dest: Path) -> None:
    """Copied from the BUILT app's own config (Final App/v{VERSION}/config/),
    not this source checkout's config/ - see _copy_models()'s comment above,
    same root cause: the built app's config is the one that actually reflects
    which preset/model is active, and it must match what _copy_models() just
    staged into models/."""
    dest_config_dir = dest / "config"
    dest_config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILT_CONFIG_PATH, dest_config_dir / "config.yaml")
    shutil.copy2(BUILT_VERSION_DIR / "config" / "dictionary.json", dest_config_dir / "dictionary.json")


def _write_empty_database(dest: Path) -> None:
    """A fresh WispernoDB (schema + default transforms/inactive example
    snippets only) - deliberately NOT a copy of this dev machine's own
    %APPDATA% database, which has real dictation history in it."""
    from src.database import WispernoDB

    db_path = dest / "wisperno.db"
    db_path.unlink(missing_ok=True)
    db = WispernoDB(db_path)
    db.close()
    print(f"  Seeded a fresh, empty database template at '{db_path.name}'.")


def build_portable_bundle() -> Path:
    if PORTABLE_DIR.exists():
        shutil.rmtree(PORTABLE_DIR)
    PORTABLE_DIR.mkdir(parents=True)

    print("Copying Wisperno.exe + PyInstaller runtime dependencies...")
    _copy_exe_and_runtime(PORTABLE_DIR)

    print("Copying model weights (Whisper + the currently active LLM preset only)...")
    _copy_models(PORTABLE_DIR)

    print("Writing a clean, relative-path config...")
    _write_clean_config(PORTABLE_DIR)

    print("Seeding an empty database template...")
    _write_empty_database(PORTABLE_DIR)

    (PORTABLE_DIR / "logs").mkdir(exist_ok=True)
    return PORTABLE_DIR


def zip_bundle(source_dir: Path) -> Path:
    """ZIP_STORED (no compression), not ZIP_DEFLATED - the bundle is almost
    entirely already-compressed binary data (GGUF weights, CTranslate2
    tensors, DLLs), where DEFLATE buys back only a percent or two of size at
    the cost of a genuinely long compression pass over several GB. Python's
    zipfile transparently uses ZIP64 extensions for files/archives over the
    legacy 4GB limit, so a multi-GB bundle like this one is not a problem."""
    print(f"Archiving to {PORTABLE_ZIP.name} (uncompressed - model weights don't compress further)...")
    if PORTABLE_ZIP.exists():
        PORTABLE_ZIP.unlink()
    t0 = time.perf_counter()
    with zipfile.ZipFile(PORTABLE_ZIP, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, Path(source_dir.name) / path.relative_to(source_dir))
    elapsed = time.perf_counter() - t0
    print(f"  Archived in {elapsed:.0f}s.")
    return PORTABLE_ZIP


def main() -> None:
    skip_zip = "--no-zip" in sys.argv
    print("=" * 70)
    print(f" Packaging Wisperno v{__version__} Portable Bundle")
    print("=" * 70)

    bundle_dir = build_portable_bundle()
    bundle_size_gb = sum(f.stat().st_size for f in bundle_dir.rglob("*") if f.is_file()) / 1e9
    print(f"\nBundle folder: {bundle_dir} ({bundle_size_gb:.1f} GB)")

    if skip_zip:
        print("\n--no-zip passed - skipping archive step.")
        print("=" * 70)
        return

    zip_path = zip_bundle(bundle_dir)
    zip_size_gb = zip_path.stat().st_size / 1e9
    print("\n" + "=" * 70)
    print(f" Portable folder: {bundle_dir} ({bundle_size_gb:.1f} GB)")
    print(f" Portable archive: {zip_path} ({zip_size_gb:.1f} GB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
