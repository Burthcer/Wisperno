"""
Live hardware telemetry (VRAM + RAM) for the System Usage tab.

Per-process VRAM does NOT come from pynvml's nvmlDeviceGetComputeRunningProcesses:
verified directly on this GeForce RTX 4070 Laptop that NVML finds our own PID in
the compute-process list but reports `usedGpuMemory=None` for it - a known NVML
limitation on consumer (non-Tesla/Quadro) GPUs under WDDM, not a bug in this code.
Instead this reads the same "GPU Process Memory\\Dedicated Usage" performance
counter Windows Task Manager itself uses, via the low-level PDH API (no pywin32
dependency - just ctypes + pdh.dll, both already present on every Windows box).
Whole-GPU total/used still comes from pynvml, which IS accurate at that level.
"""

import ctypes as ct
import os
import sys
from loguru import logger
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception as e:
    logger.debug(f"pynvml unavailable, GPU total/used telemetry will report zero: {e}")
    _NVML_OK = False

PDH_FMT_LARGE = 0x00000400
PDH_MORE_DATA = 0x800007D2


class _PdhCounterValue(ct.Structure):
    class _U(ct.Union):
        _fields_ = [
            ("longValue", ct.c_long),
            ("doubleValue", ct.c_double),
            ("largeValue", ct.c_longlong),
            ("AnsiStringValue", ct.c_char_p),
            ("WideStringValue", ct.c_wchar_p),
        ]
    _fields_ = [("CStatus", ct.c_ulong), ("u", _U)]
    _anonymous_ = ("u",)


class _PdhCounterItem(ct.Structure):
    _fields_ = [("szName", ct.c_wchar_p), ("FmtValue", _PdhCounterValue)]


class _ProcessVramReader:
    """
    Wraps a persistent PDH query against the wildcard "GPU Process Memory"
    counter set, re-collected on each snapshot() call rather than reopened -
    PdhOpenQuery/PdhAddCounter are comparatively expensive (~ms) and this may
    be polled every couple of seconds for the life of the app.
    """

    def __init__(self):
        self._pdh = None
        self._h_query = None
        self._h_counter = None
        if sys.platform != "win32":
            return
        try:
            pdh = ct.WinDLL("pdh.dll")
            h_query = ct.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, ct.byref(h_query)) != 0:
                return
            h_counter = ct.c_void_p()
            status = pdh.PdhAddEnglishCounterW(
                h_query, r"\GPU Process Memory(*)\Dedicated Usage", 0, ct.byref(h_counter)
            )
            if status != 0:
                pdh.PdhCloseQuery(h_query)
                return
            self._pdh, self._h_query, self._h_counter = pdh, h_query, h_counter
        except Exception as e:
            logger.debug(f"PDH GPU Process Memory counter unavailable: {e}")

    def read_mb(self, pid: int) -> float:
        if self._pdh is None:
            return 0.0
        try:
            if self._pdh.PdhCollectQueryData(self._h_query) != 0:
                return 0.0
            buf_size = ct.c_ulong(0)
            item_count = ct.c_ulong(0)
            # PDH_STATUS is a signed LONG but its error codes (incl. PDH_MORE_DATA)
            # are defined as unsigned hex literals - ctypes' default signed c_int
            # return type makes a raw `!= PDH_MORE_DATA` comparison always true.
            # Mask both sides back to unsigned 32-bit before comparing.
            status = self._pdh.PdhGetFormattedCounterArrayW(
                self._h_counter, PDH_FMT_LARGE, ct.byref(buf_size), ct.byref(item_count), None
            ) & 0xFFFFFFFF
            if status != PDH_MORE_DATA or buf_size.value == 0:
                return 0.0
            buf = ct.create_string_buffer(buf_size.value)
            status = self._pdh.PdhGetFormattedCounterArrayW(
                self._h_counter, PDH_FMT_LARGE, ct.byref(buf_size), ct.byref(item_count), buf
            ) & 0xFFFFFFFF
            if status != 0:
                return 0.0
            items = ct.cast(buf, ct.POINTER(_PdhCounterItem))
            prefix = f"pid_{pid}_"
            total_bytes = sum(
                items[i].FmtValue.largeValue
                for i in range(item_count.value)
                if items[i].szName and items[i].szName.startswith(prefix) and items[i].FmtValue.largeValue
            )
            return total_bytes / 1024 / 1024
        except Exception as e:
            logger.debug(f"PDH GPU Process Memory read failed: {e}")
            return 0.0

    def close(self) -> None:
        if self._pdh is not None:
            try:
                self._pdh.PdhCloseQuery(self._h_query)
            except Exception:
                pass


class SystemMonitor:
    def __init__(self):
        self._pid = os.getpid()
        self._gpu_handle = None
        if _NVML_OK:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                logger.debug(f"No NVIDIA GPU found for telemetry: {e}")
        self._proc_vram = _ProcessVramReader()

    def snapshot(self) -> dict:
        """Returns MB-denominated gpu_total/gpu_used/app_vram and ram_total/ram_used/app_ram."""
        ram = psutil.virtual_memory()
        app_ram_mb = psutil.Process(self._pid).memory_info().rss / 1024 / 1024

        gpu_total_mb = gpu_used_mb = 0.0
        if self._gpu_handle is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_total_mb = mem.total / 1024 / 1024
                gpu_used_mb = mem.used / 1024 / 1024
            except Exception as e:
                logger.debug(f"GPU total/used telemetry read failed: {e}")

        app_vram_mb = self._proc_vram.read_mb(self._pid)

        gpu_util_pct = 0
        gpu_temp_c = None
        if self._gpu_handle is not None:
            try:
                gpu_util_pct = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu
            except Exception as e:
                logger.debug(f"GPU utilization read failed: {e}")
            try:
                gpu_temp_c = pynvml.nvmlDeviceGetTemperature(self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception as e:
                logger.debug(f"GPU temperature read failed: {e}")

        # cpu_percent(interval=None) reports the delta since the LAST call on
        # this process - meaningless (always 0.0) on the very first call ever
        # made, which is fine here since this is polled every 2s for the life
        # of the app (system_usage_tab.py's QTimer), not read once.
        cpu_util_pct = psutil.cpu_percent(interval=None)
        try:
            thread_count = psutil.Process(self._pid).num_threads()
        except Exception as e:
            logger.debug(f"Thread count read failed: {e}")
            thread_count = 0

        return {
            "gpu_total_mb": gpu_total_mb,
            "gpu_used_mb": gpu_used_mb,
            "app_vram_mb": app_vram_mb,
            "gpu_util_pct": gpu_util_pct,
            "gpu_temp_c": gpu_temp_c,
            "ram_total_mb": ram.total / 1024 / 1024,
            "ram_used_mb": ram.used / 1024 / 1024,
            "app_ram_mb": app_ram_mb,
            "cpu_util_pct": cpu_util_pct,
            "thread_count": thread_count,
        }

    def close(self) -> None:
        self._proc_vram.close()


def _demo():
    m = SystemMonitor()
    snap = m.snapshot()
    assert snap["ram_total_mb"] > 0, "RAM total should always be reportable"
    assert snap["app_ram_mb"] > 0, "This process should show nonzero RSS"
    assert "gpu_util_pct" in snap and "gpu_temp_c" in snap
    print("system_monitor self-check OK:", snap)
    m.close()


if __name__ == "__main__":
    _demo()
