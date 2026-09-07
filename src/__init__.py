"""
Wisperno - Offline Desktop Dictation & Contextual Transformation Daemon.
"""

import os
import sys

# Ensure Windows finds CUDA runtime DLLs bundled in PyTorch (cublas64_12.dll, cudart64_12.dll, etc.)
if sys.platform == "win32":
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.path.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

__version__ = "1.0.0"
