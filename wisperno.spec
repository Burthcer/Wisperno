# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for Wisperno.

Produces a --onedir bundle (Wisperno.exe + a dist folder of DLLs/assets) rather
than --onefile: --onefile re-extracts its entire payload to a temp dir on every
launch, which is a poor fit for a multi-GB CUDA/torch/ctranslate2 dependency tree.

Build with:
    pyinstaller wisperno.spec --noconfirm --clean
or:
    python build_exe.py

Model weights (Whisper + .gguf LLMs) are intentionally NOT bundled - the app
resolves them from a `models/` folder next to the executable at runtime (see
src/config.py:get_base_dir(), which is frozen-aware via sys.frozen/sys.executable).
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

block_cipher = None

# Native CUDA/ML shared libraries these packages load via ctypes/dlopen at
# runtime, which PyInstaller's static import analysis can't discover on its own.
binaries = []
binaries += collect_dynamic_libs("llama_cpp")
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("torch")  # transcriber.py/transformer.py borrow torch's bundled CUDA DLLs (see SKILL.md pitfall 5)

datas = []
datas += collect_data_files("faster_whisper")
datas += [("config/config.yaml", "config"), ("config/dictionary.json", "config")]
datas += [("assets/icons/wisperno.ico", "assets/icons"), ("assets/icons/wisperno.png", "assets/icons")]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "src.config",
        "src.audio_recorder",
        "src.transcriber",
        "src.transformer",
        "src.injector",
        "src.hotkey_manager",
        "src.vocabulary",
        "src.autostart",
        "src.single_instance",
        "src.database",
        "src.workers",
        "src.engine",
        "src.system_monitor",
        "psutil",
        "pynvml",
        "src.ui.theme",
        "src.ui.floating_pill",
        "src.ui.main_window",
        "src.ui.settings_tab",
        "src.ui.shortcut_recorder",
        "src.ui.history_tab",
        "src.ui.transforms_tab",
        "src.ui.dictionary_tab",
        "src.ui.snippets_tab",
        "src.ui.icons",
        "src.ui.tray",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "sqlite3",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Wisperno",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # --noconsole: no terminal window for the daemon
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icons/wisperno.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Wisperno",
)
