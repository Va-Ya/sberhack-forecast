import ctypes
import os


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in
        ["total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended"]
    ]


state = MemoryStatus()
state.length = ctypes.sizeof(state)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))
print({"cpus": os.cpu_count(), "ram_gb": state.total / 2**30, "available_gb": state.available / 2**30}, flush=True)
